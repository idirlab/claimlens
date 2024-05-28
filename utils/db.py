import sqlite3

def generate_query(fes: dict, bill: int, agent: str) -> str:
    # Generate the query based on the FEs
    # Input: fes (dict)
    #        bill (int)
    #        agent (str)
    # Output: SQL query

    # Our input contains at least an agent and an issue, so we dont need to check for those
    if bill != None:
        # Example template for missing position
        if fes["Position"] == None:
            query = f"SELECT * FROM rollcalls JOIN votes ON rollcalls.id = votes.rollcall \
                id JOIN members ON votes.member id = members.id WHERE members.bioname = '{agent}' \
                    AND rollcalls.bill_id = {bill}"

        else:
            query = f"SELECT * FROM Votes WHERE memberId = '{agent}' and billID = '{bill}' ORDER BY congressNumber DESC;"

        return query

    # assert False, "No supported query found for input." # Remove this once we have sufficient template coverage
    return None


def query_database(query: str, db: sqlite3.Connection):
    # Query voteview database with given query
    # Input: query (str)
    #        db (sqlite3.Connection)
    # Output: result of query

    cur = db.cursor()
    cur.execute(query)

    return cur.fetchall()


def connect_to_db(db_file: str) -> sqlite3.Connection:
    return sqlite3.connect(db_file)


def load_congressmembers(db: sqlite3.Connection) -> dict:
    congressmembers = {}

    results = query_database("select bioguideid, name from members;", db)

    for bioguideid, name in results:
        congressmembers[name] = bioguideid

    return congressmembers
