import spacy
import os
from models import FrameParser, BillFinder
from utils.db import connect_to_db, query_database
from utils.fsp import clean_sentence
from utils.bill import lookup_bill
from utils.agent import lookup_agent
from dotenv import load_dotenv
from openai import OpenAI
from pprint import pprint
import json

load_dotenv()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
api_key = os.getenv("OPENAI_API_KEY")
proj_id = os.getenv("OPENAI_PROJ")
org_id = os.getenv("OPENAI_ORG")

openai_client = OpenAI(organization=org_id, project=proj_id)
string_for_api = """
Given the following factual claim, bill summary, and vote on the bill, evaluate whether the content of the bill summary and the voting record align with the given claim. You may consider factors such as the main objectives of the bill and unintended or implicit consequences. Your task is to determine if the information provided in the bill summary and the voting record supports or refutes the given factual claim. Return your explanation and one of the following labels in JSON format.
Bill Summary:
{summary}
Vote: {vote_type}
Claim: {claim}
Labels:
Supports - The vote on this bill directly or indirectly supports the claim.
Refutes - The vote on this bill explicitly refutes the claim.
Irrelevant - The vote on this bill is not relevant to the claim at all.
"""


class Claim2SQL:
    def __init__(self, args, embedder_model) -> None:
        self.fsp = FrameParser(args)
        self.db = connect_to_db(args.db)
        self.nlp = spacy.load("en_core_web_sm")
        self.bill_search_model = BillFinder(self.db, embedder_model)
        self.build_member_txt()

    def __del__(self):
        self.db.close()

    def build_member_txt(self):
        rows = query_database("SELECT * from Members;", self.db)
        with open("members.txt", "w") as f:
            # Write each row to the file
            for row in rows:
                f.write(",".join([str(value) for value in row]) + "\n")

    def get_everything(self, claim: str, bills_to_return: int = 5):
        to_return = {
            "input_sentence": claim,
            "frame_elements": None,
            "bills": None,
            "member_id": None,
            "congress_member": None,
        }

        claim = clean_sentence(claim)

        # Parse claim, get frame/FEs
        is_vote_frame, fes = self.fsp(claim)
        fes = fes[0]
        print(fes)

        if (
            not is_vote_frame
            or fes["Agent"] == None
            or fes["Issue"] == None
            or fes == []
        ):
            print("Not a vote frame or no agent/issue")
            return to_return

        to_return["frame_elements"] = fes

        agent_fe = claim[fes["Agent"]["start"] : fes["Agent"]["end"]]
        issue_fe = claim[fes["Issue"]["start"] : fes["Issue"]["end"]]

        # find agent
        agent = lookup_agent(agent_fe, self.db, self.nlp)

        if not agent:
            print(f"No agent found for {agent_fe}")
            return to_return

        agent_retrieved = agent.pop()
        to_return["member_id"] = agent_retrieved
        to_return_cong_member = query_database(
            f"SELECT Name from Members WHERE BioGuideID = '{agent_retrieved}';", self.db
        )
        if not to_return_cong_member:
            return to_return
        to_return["congress_member"] = to_return_cong_member[0][0]

        bills = lookup_bill(
            issue_fe,
            self.bill_search_model,
            agent_retrieved,
            bills_to_return,
        )

        if not bills:
            print(f"No bills found for {issue_fe}")
            return to_return

        bill_details = []
        for i, bill in enumerate(bills):
            res = query_database(
                f"SELECT BillCongress, BillType, BillNumber, BillSummary from bills WHERE BillID = '{bill}';",
                self.db,
            )
            id = f"{res[0][0]} {res[0][1]} {res[0][2]}"
            summary = res[0][3]
            vote_type = query_database(
                f"SELECT VoteType from Rollcalls WHERE BillID = '{bill}' AND MemberID = '{agent_retrieved}';",
                self.db,
            )[0][0]

            if False:
                openai_response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "Follow the user input exactly"},
                        {
                            "role": "user",
                            "content": string_for_api.format(
                                summary=summary, vote_type=vote_type, claim=claim
                            ),
                        },
                    ],
                )

                alignment, alignment_explanation = (
                    json.loads(openai_response.choices[0].message.content)["Label"],
                    json.loads(openai_response.choices[0].message.content)[
                        "Explanation"
                    ],
                )
            else:
                alignment, alignment_explanation = (
                    "None",
                    "Currently, we only retrieve alignments for the top 5 bills.",
                )

            bill_details.append(
                {
                    "bill_title": id,
                    "bill_summary": summary,
                    "vote_type": vote_type,
                    "alignment": alignment,
                    "alignment_explanation": alignment_explanation,
                }
            )

        to_return["bills"] = bill_details

        # print(to_return)
        return to_return
