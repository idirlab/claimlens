import sqlite3
from nicknames import NickNamer
from utils.db import query_database


def lookup_agent(agent: str, db: sqlite3.Connection, nlp) -> str:
    """Given an agent, find the BioGuide ID of the agent in the database.

    Args:
        agent (str): string containing the name of the agent, taken from frame-semantic parser,
        db (sqlite3.Connection): database connection
        nlp (spacy.Language): Spacy NLP model for finding person names in agent

    Returns:
        str: BioGuide ID of the agent
    """

    candidate_ids = set()
    agent_names = []

    # Split agent
    for tok in nlp(agent):
        if tok.ent_type_ == "PERSON":
            agent_names.append(tok.text)

    if len(agent_names) == 0:
        return None

    last_name = agent_names.pop()

    nn = NickNamer()
    for name in agent_names:
        for nick in nn.canonicals_of(name):
            if nick not in agent_names:
                agent_names.append(nick)

    if len(agent_names) == 1:
        query = f"select bioguideId from members where upper(name) like upper('%{agent_names[0]}%') and upper(name) like upper('%{last_name}%')"

    elif len(agent_names) == 0:
        query = f"select bioguideId from members where upper(name) like upper('%{last_name}%')"

    else:
        query = f"select bioguideId from members where"
        for i, part in enumerate(agent_names):
            if i == 0:  # don't start with an or
                query += f"(upper(name) like upper('%{part}%') or"
            elif i == len(agent_names) - 1:  # end of first and middle name queries
                query += f" upper(name) like upper('%{part}%'))"
            else:
                query += f" upper(name) like upper('%{part}%') or"

        query += f" and upper(name) like upper('%{last_name}%')"
        query += (
            " order by bioguideId desc;"  # return most recent congressmembers first
        )

    results = query_database(query, db)

    for result in results:
        candidate_ids.add(result[0])

    if len(results) == 0:
        candidate_ids.add(None)

    return candidate_ids
