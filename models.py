from typing import Any, Optional, Tuple, Union
from transformers import (
    AutoTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaPreTrainedModel,
)
import sqlite3
from sentence_transformers import SentenceTransformer, util
from torch.nn import CrossEntropyLoss, Linear
import torch
from utils.db import query_database
from utils.fsp import get_vote_frame_info, FrameSemanticParsingOutput
from utils.partition import partition_predictions
from utils.fsp import VoteFSPDataset, build_model_inputs
from torch.utils.data import DataLoader


class UnifiedFrameSemanticParser(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        self.fe_identifier = Linear(config.hidden_size, 2)
        self.frame_identifier = Linear(config.hidden_size, 2)

        self.frame_loss = CrossEntropyLoss()
        self.fe_loss = CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        fe_start_positions: Optional[torch.LongTensor] = None,
        fe_end_positions: Optional[torch.LongTensor] = None,
        target_labels: Optional[torch.LongTensor] = None,
        target_spans: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], FrameSemanticParsingOutput]:
        r"""
        fe_start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        fe_end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        target_labels (`torch.LongTensor` of shape `(batch_size, seq_len)`, *optional*):
            Labels for the targets used during frame identification, 1 = the frame matches the target
                0 = frame doesn't match the target, -100 = ignore/not a target
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        fe_logits = self.fe_identifier(sequence_output)  # [seq_len, 2]

        start_fe_logits, end_fe_logits = fe_logits.split(
            1, dim=-1
        )  # -> # [seq_len,] * 2
        start_fe_logits = start_fe_logits.squeeze(-1).contiguous()
        end_fe_logits = end_fe_logits.squeeze(-1).contiguous()

        # Mask out irrelevant tokens
        pooled_sequence = sequence_output.clone()
        for i in range(target_spans.shape[0]):
            pooled_sequence[i, : target_spans[i][0], :] = 0
            pooled_sequence[i, target_spans[i][1] + 1 :, :] = 0

        # average pool the token spans
        pooled_sequence = pooled_sequence.mean(dim=1)  # [b, h]

        frame_logits = self.frame_identifier(pooled_sequence)  # [b, 2]

        total_loss = None

        if (
            fe_start_positions is not None
            and fe_end_positions is not None
            and target_labels is not None
        ):
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            start_loss = self.fe_loss(start_fe_logits, fe_start_positions)
            end_loss = self.fe_loss(end_fe_logits, fe_end_positions)

            frame_loss = self.frame_loss(frame_logits, target_labels)

            total_loss = (start_loss + end_loss) / 2 + frame_loss

        if not return_dict:
            output = (start_fe_logits, end_fe_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return FrameSemanticParsingOutput(
            loss=total_loss,
            start_fe_logits=start_fe_logits,
            end_fe_logits=end_fe_logits,
            frame_logits=frame_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class VoteCandidateTargetIdentifier:
    def __init__(self):
        self.vote_lus = set(["vote", "voted", "voting", "votes"])

    def lookup_lus(self, sentence):
        sent_words = sentence.lower().split(" ")

        candidate_spans = []

        _index = 0
        for word in sent_words:
            if word in self.vote_lus:
                candidate_spans.append((_index, _index + len(word)))

            _index += len(word) + 1

        return candidate_spans

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.lookup_lus(*args, **kwds)


class FrameParser:
    def __init__(self, args):
        self.device = args.device
        self.model = UnifiedFrameSemanticParser(
            config=RobertaConfig.from_pretrained("roberta-base")
        )
        self.model.load_state_dict(torch.load(args.fsp_model, map_location=self.device))
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
        self.target_id = VoteCandidateTargetIdentifier()
        self.vote_info = get_vote_frame_info(args.vote_file)
        self.label_map = {
            "Agent": 0,
            "Frequency": 1,
            "Issue": 2,
            "Place": 3,
            "Position": 4,
            "Side": 5,
            "Support_rate": 6,
            "Time": 7,
            0: "Agent",
            1: "Frequency",
            2: "Issue",
            3: "Place",
            4: "Position",
            5: "Side",
            6: "Support_rate",
            7: "Time",
        }

    def __call__(self, claim: str) -> tuple((str, dict)):
        return self.frame_semantic_parse(claim)

    def partition_to_char_pos(self, partition, input_ids):
        fe_pos = {x: {"start": -1, "end": -1} for x in self.vote_info["fes"]}

        for i in range(len(self.vote_info["fes"])):
            partition_tokens = partition.partition == i
            # Skip if partition is empty
            if partition_tokens.sum() == 0:
                continue

            start_idx, end_idx = partition_tokens.nonzero().flatten()[[0, -1]]

            fe_pos[self.vote_info["fes"][i]] = {
                "start": input_ids.token_to_chars(start_idx).start,
                "end": input_ids.token_to_chars(end_idx).end,
            }

        return fe_pos

    def partition_to_str(self, partition, input_ids):
        str_parts = {x: "" for x in self.vote_info["fes"]}
        sep_index = (input_ids == self.tokenizer.sep_token_id).nonzero()[0]

        for i in range(len(self.vote_info["fes"])):
            partition_tokens = partition.partition == i
            str_parts[self.vote_info["fes"][i]] = self.tokenizer.decode(
                input_ids[:sep_index][partition_tokens], skip_special_tokens=True
            ).strip()

        return str_parts

    def frame_semantic_parse(self, claim: str) -> tuple((str, dict)):
        # Get frame and frame elements from claim
        # Input: claim (str)
        # Output: frame (str), fes (dict)

        fes = {
            "sentence": None,
            "Agent": None,
            "Issue": None,
            "Side": None,
            "Frequency": None,
            "Position": None,
            "Support_rate": None,
            "Place": None,
            "Time": None,
        }  # These values can be changed based on the outputs of the model

        # Get targets
        claim_targets = self.target_id(claim)

        if len(claim_targets) == 0:
            return False, []

        input_samples = [
            {"sentence": claim, "target_span": x, "fe_spans": []} for x in claim_targets
        ]

        tok_sent = self.tokenizer(claim)

        model_inputs = build_model_inputs(
            input_samples, self.tokenizer, self.vote_info, label=0
        )

        input_dataset = VoteFSPDataset(model_inputs, self.device, self.tokenizer)
        input_dataloader = DataLoader(input_dataset, batch_size=1)

        frame_pred = None
        partitions = []
        with torch.no_grad():
            for i, batch in enumerate(input_dataloader):
                batch = {k: v.squeeze(0) for k, v in batch.items()}
                model_output = self.model(**batch)
                sep_index = (
                    batch["input_ids"] == self.tokenizer.sep_token_id
                ).nonzero()[0][-1]

                frame_pred = (
                    model_output.frame_logits.argmax(dim=-1).cpu() == 1
                ).sum() >= 4

                best_partition = partition_predictions(
                    sep_index,
                    model_output.start_fe_logits.cpu(),
                    model_output.end_fe_logits.cpu(),
                    batch["target_spans"].cpu(),
                )

                partitions.append(self.partition_to_char_pos(best_partition, tok_sent))

        return frame_pred, partitions


class BillFinder:
    """Bill finding class. Given an issue, find the bill that the congressmember with the given BioGuide ID voted on."""

    def __init__(self, db: sqlite3.Connection, embedder_model):
        self.db = db
        self.bills = None
        self.bill_file = "./data/bill_embeddings.pkl"
        self.embedder = SentenceTransformer(embedder_model)
        self.embedder.eval()
        self.load_bills()

    def compute_corpus_embeddings(self):
        # get all ids and bill descriptions from the Bills table

        all_bills = query_database(
            "SELECT bills.BillID, bills.BillDescription FROM bills WHERE bills.BillID IN (SELECT DISTINCT(billId) FROM Rollcalls);",
            self.db,
        )
        ids = []
        corpus = []
        for t in all_bills:
            ids.append(t[0])
            corpus.append(t[1])
        # compute embeddings for all bills and return dictionary with ids as keys and embeddings as values
        corpus_embeddings = self.embedder.encode(corpus, convert_to_tensor=True)
        corpus_embeddings_dict = {}
        for i in range(len(ids)):
            assert isinstance(corpus_embeddings[i], torch.Tensor)
            # assert corpus_embeddings[i].shape == (768,)
            corpus_embeddings_dict[ids[i]] = corpus_embeddings[i]
        return corpus_embeddings_dict

    def load_bills(self):
        # Create embeddings every time C2S is initialized
        self.bills = self.compute_corpus_embeddings()

    def search(self, issue: str, bioguide_id: str, num_bills=1) -> Union[str, str]:
        """Search for the bill that the congressmember with the given BioGuide ID voted on.

        Args:
            issue (str): Issue FE from frame-semantic parser

        Returns:
            Union[str, str]: Tuple of congress number and bill ID
        """

        # get the ids of the bills voted on by the given congress person
        query = f"SELECT bills.BillID FROM bills WHERE bills.BillID in (SELECT DISTINCT(BillID) FROM Rollcalls WHERE MemberID = '{bioguide_id}');"
        bills_tuple = query_database(query, self.db)

        if len(bills_tuple) == 0:
            return None

        bills_id = []
        for t in bills_tuple:
            bills_id.append(t[0])

        # get the embeddings of the retrieved bills
        bills_embeddings = []

        for id in bills_id:
            bills_embeddings.append(self.bills.get(id))

        query_embedding = self.embedder.encode(issue, convert_to_tensor=True)

        hits = util.semantic_search(
            query_embedding,
            bills_embeddings,
            top_k=num_bills,
            score_function=util.dot_score,
        )
        hits = hits[0]

        res = []
        for h in hits:
            res.append(bills_id[h["corpus_id"]])

        return res
