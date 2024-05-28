import torch
from typing import Optional, Tuple
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import xml.etree.ElementTree as ET

class FrameSemanticParsingOutput:
    loss: Optional[torch.FloatTensor] = None
    frame_logits: Optional[torch.FloatTensor] = None
    start_fe_logits: torch.FloatTensor = None
    end_fe_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    def __init__(self, loss=None, start_fe_logits=None, end_fe_logits=None, frame_logits=None, hidden_states=None, attentions=None):
        self.frame_logits = frame_logits
        self.loss = loss
        self.start_fe_logits = start_fe_logits
        self.end_fe_logits = end_fe_logits
        self.hidden_states = hidden_states
        self.attentions = attentions
    
    def __repr__(self):
        return f"FrameSemanticParsingOutput:\n\tLoss: {self.loss}\n\tFrame Logits: {self.frame_logits}\n\tStart Logits: {self.start_fe_logits}\n\tEnd Logits: {self.end_fe_logits}\n\tHidden States: {self.hidden_states}\n\tAttentions: {self.attentions}\n"
    
    def __str__(self):
        return self.__repr__()
    
class VoteFSPDataset(Dataset):
    def __init__(self, dataset, device, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_ids = pad_sequence([x["input_ids"] for x in self.dataset[idx]], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence([x["attention_mask"] for x in self.dataset[idx]], batch_first=True)
        target_spans = torch.tensor([x["target_span"] for x in self.dataset[idx]])
        fe_start_positions = torch.tensor([x["fe_span"][0] for x in self.dataset[idx]])
        fe_end_positions = torch.tensor([x["fe_span"][1] for x in self.dataset[idx]])
        target_labels = torch.tensor([x["target_label"] for x in self.dataset[idx]])
        
        return {"input_ids":input_ids.to(self.device), 
                "attention_mask":attention_mask.to(self.device),
                "fe_start_positions":fe_start_positions.to(self.device), 
                "fe_end_positions":fe_end_positions.to(self.device),
                "target_labels":target_labels.to(self.device), 
                "target_spans":target_spans.to(self.device)}

def build_model_inputs(train_vote_samples, tokenizer, vote_frame_info, label=1):
    train_raw_dataset = []

    for sample in train_vote_samples:
        sample_inputs = []
        
        sample_toks = tokenizer.encode_plus(sample['sentence'], return_tensors="pt")
        target_span = (sample_toks.char_to_token(sample["target_span"][0]), sample_toks.char_to_token(sample["target_span"][1]-1))
        sample_fes = {x:(0, 0) for x in vote_frame_info["fes"]}
        
        for sample_fe in sample["fe_spans"]:
            fe_name = sample_fe[-1]
            fe_span = (sample_toks.char_to_token(sample_fe[0]), sample_toks.char_to_token(sample_fe[1]))
            sample_fes[fe_name] = fe_span
        
        for fe_name, span in sample_fes.items():
            fe_toks = tokenizer.encode_plus(f"{sample['sentence']}{tokenizer.sep_token}Vote{tokenizer.sep_token}{fe_name}", return_tensors="pt")
            sample_inputs.append({"input_ids":fe_toks["input_ids"].squeeze(), "attention_mask":fe_toks["attention_mask"].squeeze(), "target_span":target_span, "target_label":label, "fe_span":span, "fe_name":fe_name})
        
        train_raw_dataset.append(sample_inputs)
        
    return train_raw_dataset

def clean_sentence(sentence):
    return sentence.strip("\n")#.replace(" n't ", "n't ").replace(" 's ", "'s ").replace(" ' ", "' ").replace(" 're ", "'re ").replace(" 've ", "'ve ").replace(" 'm ", "'m ").replace(" 'll ", "'ll ").replace(" 'd ", "'d ")

def get_vote_frame_info(frame_file):
    frame_root = ET.parse(frame_file).getroot()
    fes = [x.attrib["name"] for x in frame_root.findall("{http://framenet.icsi.berkeley.edu}FE")]
    return {"name":frame_root.attrib["name"], "fes":fes, "num_elements":len(fes)}