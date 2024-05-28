import argparse
import spacy
from utils.db import connect_to_db
from utils.agent import lookup_agent

def test_parse_name(args):
    nlp = spacy.load("en_core_web_sm")

    db = connect_to_db(args.db)

    test_cases = {
        "Patrick Murphy": 21319,
        "Congressman Joe Heck": 21151,
        "Debbie Wasserman Schultz": 20504,
        "U.S. Sen. Johnny Isakson": 29909,
        "GOP Rep. Joe Heck of Nevada": 21151,
        "U.S. Rep. Bruce Poliquin": 21524,
        "Miami Congressman Carlos Curbelo": 21512,
        "Colorado congressional candidate Morgan Carroll": None,
        "Sen. Johnny Isakson": 29909,
        "Democratic U.S. Senate candidate Russ Feingold": 49309,
        "Russ Feingold": 49309,
        "Democrat Stephen Webber": None,
        "Judge Neil Gorsuch": None,
        "Then-Senator Obama": 40502,
        "Sen. Schumer": 14858,
        "U.S. Sen. Chuck Schumer": 14858,
        "Republican Rep. Todd Rokita of Indiana": 21131,
        "John Smith": None,
    }
    # FUTURE: we should also support "he" in the future, must use coreferece resolution

    for agent, icpsr in test_cases.items():
        pred = lookup_agent(agent, db, nlp)
        print(
            f"Testing name: {agent} -> {pred} (expected {icpsr}) --- {'SUCCESS' if icpsr in pred else 'FAIL'}"
        )
