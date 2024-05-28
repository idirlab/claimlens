from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import wikipediaapi
import requests
import httpx


from claim2sql import Claim2SQL


class ArgConfig:
    def __init__(
        self,
        db="./data/congress.db",
        fsp_model="./models/vote-fsp",
        device="cpu",
        vote_file="./data/Vote.xml",
        test_name_parse=False,
    ):
        self.db = db
        self.fsp_model = fsp_model
        self.device = device
        self.vote_file = vote_file
        self.test_name_parse = test_name_parse


# Set hostname to idir.uta.edu/claimlens/ for deployment
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

args = ArgConfig()
claim2sql = Claim2SQL(args, "msmarco-distilbert-base-tas-b")

templates = Jinja2Templates(directory="templates")

wiki = wikipediaapi.Wikipedia("idir_lab", "en")


def get_default_request_vars():
    return {
        "static_url_path": "https://idir.uta.edu/claimlens/static/",
        "base_url": "https://idir.uta.edu/claimlens/",
    }


def get_member_image_url(member_name):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": member_name,
        "prop": "pageimages",
        "pithumbsize": 300,  # Set the desired thumbnail size (e.g., 300 pixels)
    }
    response = requests.get(url, params=params)
    data = response.json()
    page = next(iter(data["query"]["pages"].values()))

    if "thumbnail" not in page:
        # If no thumbnail is available, use a default image
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/No_image_available.svg/300px-No_image_available.svg.png"
    else:
        if "original" in page["thumbnail"]:
            # If the "original" size is available, use it for higher quality
            image_url = page["thumbnail"]["original"]
        else:
            # Otherwise, use the larger thumbnail size requested
            image_url = page["thumbnail"]["source"]

    return image_url


@app.get("/", response_class=HTMLResponse)
async def read_root(
    request: Request, default_vars: dict = Depends(get_default_request_vars)
):
    return templates.TemplateResponse(
        "index.html", {"request": request, **default_vars}
    )


@app.get("/fact-check", response_class=HTMLResponse)
async def read_fact_check(
    request: Request, default_vars: dict = Depends(get_default_request_vars)
):
    return templates.TemplateResponse(
        "fact-check.html", {"request": request, **default_vars}
    )


@app.get("/submit", response_class=HTMLResponse)
async def submit_text(
    query: str, request: Request, default_vars: dict = Depends(get_default_request_vars)
):
    # Call API to retrieve model inputs
    model_outputs = claim2sql.get_everything(query, 10)

    # Define frame element definitions
    fe_definitions = {
        "Agent": "The conscious entity, generally a person, that performs the voting decision on an Issue.",
        "Issue": "The matter which the Agent has a positive or negative opinion about and either votes for or votes against.",
        "Side": "The conscious entity, generally a person, that performs the voting decision on an Issue together with the Agent.",
        "Position": "This FE identifies the position that the Agent takes on an Issue. The Agent either votes for or against.",
        "Frequency": "The number of times that the Agent made the same voting decision on an Issue.",
        "Time": "This FE identifies the time when the Agent performs the voting decision.",
        "Place": "The location where the voting decision took place.",
        "Support_rate": "The percentage of Agent's total votes, in which the Agent took the same position with a Side for a specific period of time.",
    }

    # Build html for frame elements
    prev_end = 0
    fe_html = ""
    for fe, span in sorted(
        model_outputs["frame_elements"].items(), key=lambda x: x[1]["start"]
    ):
        if span["start"] == -1 or span["end"] == -1:
            continue

        if prev_end != span["start"]:
            fe_html += f"<span class='non-fe'>{model_outputs['input_sentence'][prev_end:span['start']]}</span>"
        fe_html += f"<span data-content='{fe_definitions[fe]}' class='fe-{fe.lower()}'>{model_outputs['input_sentence'][span['start']:span['end']]}</span>"

        prev_end = span["end"]
    fe_html += f"<span>{model_outputs['input_sentence'][prev_end:]}</span>"

    # Get agent wikipedia page
    agent_page = wiki.page(model_outputs["congress_member"])
    agent_summary = agent_page.summary.split("\n")[0]
    agent_url = agent_page.fullurl

    # Get congress member image
    image_url = get_member_image_url(model_outputs["congress_member"])

    # Render results page
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "model_outputs": model_outputs,
            "agent_summary": agent_summary,
            "agent_url": agent_url,
            "fe_html": fe_html,
            "image_url": image_url,
            **default_vars,
        },
    )


@app.get("/api", response_class=JSONResponse)
async def root(claim: str, model_name: str):
    model_outputs = claim2sql.get_everything(claim, 10)
    return model_outputs
