import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from core.atlasmind import AtlasMind, normalize_issue
from core.field_resolver import ResolvedIntentFields
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
_llm_backend: str = os.getenv("LLM_BACKEND", "ollama")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _atlasmind
    logger.info("Starting up — seeding pgvector databases... (backend: %s)", _llm_backend)
    config = EmbeddingsConfig(model_name=EMBEDDING_MODEL)
    _atlasmind = AtlasMind(config, llm_backend=_llm_backend)
    _atlasmind.run()
    logger.info("Ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="aMind JQL Generator", lifespan=lifespan)


# -- Helpers ----------------------------------------------------------

def _build_display_fields(resolved: ResolvedIntentFields) -> list[str]:
    """Build the ordered display field list for the frontend.

    Standard fields are always first, resolved to their canonical display names
    via FieldResolver. Intent fields proposed by the LLM are appended after.

    Args:
        resolved: Intent fields resolved from the LLM response.

    Returns:
        List of display name strings for the frontend to use as column headers.
    """
    if _atlasmind and _atlasmind.field_resolver:
        standard_names = _atlasmind.field_resolver.display_names_for_ids(
            _atlasmind.standard_field_ids
        )
    else:
        standard_names = list(_atlasmind.standard_field_ids) if _atlasmind else []
    return standard_names + resolved.display_names


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


def _build_response(llm_result, jira_result: dict | None) -> QueryResponse:
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

    resolved: ResolvedIntentFields = jira_result.get(
        "resolved_intent_fields", ResolvedIntentFields()
    )
    extra_fields = resolved.as_extra_fields()
    requested_ids: set[str] = (
        set(_atlasmind.standard_field_ids) | set(resolved.field_ids)
        if _atlasmind else set()
    )
    normalised = [
        normalize_issue(r, extra_fields=extra_fields, requested_ids=requested_ids)
        for r in jira_result.get("raw_issues", [])
    ]

    return QueryResponse(
        type="jql",
        profile=profile_name,
        jira_base_url=jira_base_url,
        answer=llm_result.answer,
        jql=jira_result.get("jql"),
        total=jira_result.get("total", 0),
        shown=jira_result.get("shown", 0),
        examined=jira_result.get("shown", 0),
        display_fields=_build_display_fields(resolved),
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
        return _build_response(llm_result, jira_result).model_dump()
    except ValueError as exc:
        logger.error("Query failed: %s", exc)
        return _error_response(str(exc))
    except Exception as exc:
        logger.exception("Query failed (unexpected): %s", exc)
        return _error_response(str(exc))


@app.post("/query")
async def query_post(request: QueryRequest):
    """Translate a natural language query to JQL and return Jira issues (POST)."""
    if _atlasmind is None:
        raise HTTPException(status_code=503, detail="Model not initialised.")
    try:
        llm_result, jira_result = await _atlasmind.generate_jql(request.query)
        return _build_response(llm_result, jira_result).model_dump()
    except ValueError as exc:
        logger.error("Query failed: %s", exc)
        return _error_response(str(exc))
    except Exception as exc:
        logger.exception("Query failed (unexpected): %s", exc)
        return _error_response(str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
