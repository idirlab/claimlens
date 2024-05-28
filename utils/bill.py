from typing import Union


def lookup_bill(
    issue: str, bill_search_model, bioguide_id: str, num_bills=5
) -> Union[str, str]:
    """Given an issue, find the bill ID of the bill that the congressmember with the given BioGuide ID voted on.

    Args:
        issue (str): Issue FE from frame-semantic parser
        db (sqlite3.Connection): DB connection
        bill_search_model (_type_): _description_
        bioguide_id (str): bioguide ID of the agent
        num_bills (int, optional): number of bills to return. Defaults to 1.

    Returns:
        int: _description_
    """

    # Search for bill
    result = bill_search_model.search(issue, bioguide_id, num_bills)

    # Return bill ID
    return result
