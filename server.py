import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from core.atlasmind import AtlasMind, normalize_issue
from dconfig import EmbeddingsConfig
from core.models import ChartSpec, QueryRequest, QueryResponse
from config.jira_config import load_active_profile
from settings import EMBEDDING_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_atlasmind: AtlasMind | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _atlasmind
    logger.info("Starting up — seeding pgvector databases...")
    config = EmbeddingsConfig(model_name=EMBEDDING_MODEL)
    _atlasmind = AtlasMind(config)
    _atlasmind.run()
    logger.info("Ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="aMind JQL Generator", lifespan=lifespan)


# -- Helpers ----------------------------------------------------------

_DEFAULT_DISPLAY_FIELDS = ["status", "assignee", "created"]

_FIELD_HINTS: list[tuple[list[str], str]] = [
    (["assignee", "assigned to"],   "assignee"),
    (["reporter", "reported by"],   "reporter"),
    (["status"],                    "status"),
    (["priority"],                  "priority"),
    (["type", "issuetype"],         "issuetype"),
    (["sprint"],                    "sprint"),
    (["label"],                     "labels"),
    (["epic"],                      "epic_link"),
    (["point", "story point"],      "story_points"),
    (["created"],                   "created"),
    (["updated", "modified"],       "updated"),
    (["due"],                       "duedate"),
    (["resolved", "resolution"],    "resolutiondate"),
    (["parent"],                    "parent"),
    (["comment"],                   "comments"),
    (["description", "detail"],     "description"),
    (["effort", "days", "day", "completion time", "time to close",
       "took", "longest", "duration", "cycle time"], "effort_days"),
    (["effort hour", "hours", "hour"], "effort_hours"),
    (["age", "open for", "how long open"], "age_days"),
]


def _detect_display_fields(query: str) -> list[str]:
    q = query.lower()
    fields: list[str] = []
    for keywords, field in _FIELD_HINTS:
        if any(kw in q for kw in keywords):
            if field not in fields:
                fields.append(field)
    for f in _DEFAULT_DISPLAY_FIELDS:
        if f not in fields:
            fields.append(f)
    return fields


def _extract_filters(issues: list[dict]) -> dict[str, list[str]]:
    """Build filter facets from normalised issues for frontend filter dropdowns."""
    facet_fields = ["status", "issuetype", "priority", "assignee", "labels"]
    facets: dict[str, set] = {f: set() for f in facet_fields}
    for issue in issues:
        for field in facet_fields:
            value = issue.get(field)
            if value is None:
                continue
            if isinstance(value, list):
                facets[field].update(v for v in value if v)
            else:
                facets[field].add(value)
    return {k: sorted(v) for k, v in facets.items() if v}


def _build_response(llm_result, jira_result: dict | None, query: str = "") -> QueryResponse:
    """Build a uniform QueryResponse from LLM and optional Jira results."""
    profile = load_active_profile()
    profile_name = profile.get("name", "default")
    jira_base_url = profile.get("jira_url", "")

    chart_spec = None
    if llm_result.chart_spec:
        try:
            chart_spec = ChartSpec(**llm_result.chart_spec)
        except Exception:
            pass

    if jira_result is None:
        return QueryResponse(
            type="general",
            profile=profile_name,
            jira_base_url=jira_base_url,
            answer=llm_result.answer,
            chart_spec=chart_spec,
        )

    normalised = [normalize_issue(r) for r in jira_result.get("raw_issues", [])]
    return QueryResponse(
        type="jql",
        profile=profile_name,
        jira_base_url=jira_base_url,
        answer=llm_result.answer,
        jql=jira_result.get("jql"),
        total=jira_result.get("total", 0),
        shown=jira_result.get("shown", 0),
        examined=jira_result.get("shown", 0),
        display_fields=_detect_display_fields(query),
        issues=normalised,
        chart_spec=chart_spec,
        filters=_extract_filters(normalised),
    )


def _error_response(message: str) -> dict:
    """Return a general-type response carrying an error message for the frontend."""
    profile = load_active_profile()
    return QueryResponse(
        type="general",
        profile=profile.get("name", "default"),
        jira_base_url=profile.get("jira_url", ""),
        answer=f"Error: {message}",
    ).model_dump()


# -- Endpoints --------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/query")
async def query_get(
    q: str = Query(..., description="Natural language Jira query"),
):
    """Translate a natural language query to JQL and return Jira issues (GET)."""
    if _atlasmind is None:
        raise HTTPException(status_code=503, detail="Model not initialised.")
    try:
        llm_result, jira_result = await _atlasmind.generate_jql(q)
        return _build_response(llm_result, jira_result, query=q).model_dump()
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        return _error_response(str(exc))


@app.post("/query")
async def query_post(request: QueryRequest):
    """Translate a natural language query to JQL and return Jira issues (POST)."""
    if _atlasmind is None:
        raise HTTPException(status_code=503, detail="Model not initialised.")
    try:
        llm_result, jira_result = await _atlasmind.generate_jql(request.query)
        return _build_response(llm_result, jira_result, query=request.query).model_dump()
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        return _error_response(str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
