import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query

from core.atlasmind import AtlasMind, normalize_issue, _FIELD_ID_TO_OUTPUT_KEY
from core.vllm_client import VllmUnavailable
from core.field_resolver import ExtraField, ResolvedIntentFields
from dconfig import EmbeddingsConfig
from core.models import ChartSpec, QueryRequest, QueryResponse, ServerMeta
from core.client_events import ClientEvent, ClientEventType, EventAck
import core.client_events as client_events
from config.jira_config import load_active_jira_profile
from core.jira_auth import jira_token_dep, jira_url_dep
from settings import EMBEDDING_MODEL, GROQ_MODEL, OLLAMA_MODEL, CLAUDE_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_atlasmind: AtlasMind | None = None
_llm_backend: str = os.getenv("LLM_BACKEND", "ollama")
_server_meta: ServerMeta | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _atlasmind, _server_meta
    logger.info("Starting up — seeding pgvector databases... (backend: %s)", _llm_backend)
    config = EmbeddingsConfig(model_name=EMBEDDING_MODEL)
    _atlasmind = AtlasMind(config, llm_backend=_llm_backend)
    _atlasmind.run()
    if _atlasmind.llm_backend == "groq":
        _meta_model_name = f"Groq: {GROQ_MODEL}"
    elif _atlasmind.llm_backend == "vllm":
        _meta_model_name = f"vLLM: {_atlasmind.llm_client.model}"
    elif _atlasmind.llm_backend == "claude":
        _meta_model_name = f"Claude: {CLAUDE_MODEL}"
    else:
        _meta_model_name = f"Ollama: {OLLAMA_MODEL}"
    _server_meta = ServerMeta(
        model_name=_meta_model_name,
        llm_backend=_atlasmind.llm_backend,
        llm_timeout=_atlasmind.llm_client.timeout,
    )
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
    profile = load_active_jira_profile()
    profile_name = profile.name
    jira_base_url = profile.jira_url

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
            answer=llm_result.answer or "No response generated. Please try rephrasing your query.",
            chart_spec=chart_spec,
            meta=_server_meta,
        )

    resolved: ResolvedIntentFields = jira_result.get(
        "resolved_intent_fields", ResolvedIntentFields()
    )
    extra_fields = resolved.as_extra_fields()
    if _atlasmind and _atlasmind.field_resolver:
        intent_ids = set(resolved.field_ids)
        for fid in _atlasmind.standard_field_ids:
            if fid not in _FIELD_ID_TO_OUTPUT_KEY and fid not in intent_ids:
                display = _atlasmind.field_resolver._id_to_name.get(fid, fid)
                extra_fields.append(ExtraField(field_id=fid, display_name=display))
    requested_ids: set[str] = (
        set(_atlasmind.standard_field_ids) | set(resolved.field_ids)
        if _atlasmind else set()
    )
    normalised = [
        normalize_issue(r, extra_fields=extra_fields, requested_ids=requested_ids)
        for r in jira_result.get("raw_issues", [])
    ]

    total = jira_result.get("total", 0)
    shown = jira_result.get("shown", 0)

    base_answer = llm_result.answer or ""
    if total == 0:
        count_note = "No results found."
    elif shown < total:
        count_note = f"Found {total} result(s); showing {shown}."
    else:
        count_note = f"Found {total} result(s)."
    answer = f"{base_answer} {count_note}".strip() if base_answer else count_note

    return QueryResponse(
        type="jql",
        profile=profile_name,
        jira_base_url=jira_base_url,
        answer=answer,
        jql=jira_result.get("jql"),
        total=total,
        shown=shown,
        examined=shown,
        display_fields=_build_display_fields(resolved),
        issues=normalised,
        chart_spec=chart_spec,
        filters=_extract_filters(normalised),
        meta=_server_meta,
    )


def _error_response(message: str) -> dict:
    """Return a general-type response carrying an error message for the frontend."""
    profile = load_active_jira_profile()
    return QueryResponse(
        type="general",
        profile=profile.name,
        jira_base_url=profile.jira_url,
        answer=f"Error: {message}",
    ).model_dump()


# -- Endpoints --------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta", response_model=ServerMeta)
def meta():
    """Return server metadata. Frontend should call this once on bootup."""
    if _server_meta is None:
        raise HTTPException(status_code=503, detail="Server not initialised.")
    return _server_meta


@app.get("/query")
async def query_get(
    q:          str = Query(...,  description="Natural language Jira query"),
    request_id: str = Query(None, description="Client-generated UUID for cancel support"),
    jira_token: str | None = Depends(jira_token_dep),
    jira_url:   str | None = Depends(jira_url_dep),
):
    """Translate a natural language query to JQL and return Jira issues (GET)."""
    if _atlasmind is None:
        raise HTTPException(status_code=503, detail="Model not initialised.")

    logger.info("[cancel] GET /query request_id=%r", request_id)
    token = client_events.register(request_id) if request_id else None
    if not request_id:
        logger.warning("[cancel] GET /query has no request_id — cancel will not work for this query")

    task = asyncio.create_task(_atlasmind.generate_jql(q, jira_token=jira_token, jira_url=jira_url))
    if token:
        token.attach(task)

    try:
        llm_result, jira_result = await task
        return _build_response(llm_result, jira_result).model_dump()
    except asyncio.CancelledError:
        logger.info("Query cancelled by client: request_id=%s", request_id)
        return _error_response("Query cancelled.")
    except VllmUnavailable as exc:
        logger.error("Query failed — vLLM unavailable: %s", exc)
        return _error_response("LLM service is temporarily unavailable. Please try again later.")
    except ValueError as exc:
        logger.error("Query failed: %s", exc)
        return _error_response(str(exc))
    except Exception as exc:
        logger.exception("Query failed (unexpected): %s", exc)
        return _error_response(str(exc))
    finally:
        if request_id:
            client_events.unregister(request_id)


@app.post("/query")
async def query_post(
    request: QueryRequest,
    jira_token: str | None = Depends(jira_token_dep),
    jira_url:   str | None = Depends(jira_url_dep),
):
    """Translate a natural language query to JQL and return Jira issues (POST)."""
    if _atlasmind is None:
        raise HTTPException(status_code=503, detail="Model not initialised.")

    # Register the cancel token immediately — before any async work — so a
    # cancel event arriving during LLM classification still finds the token.
    logger.info("[cancel] POST /query request_id=%r", request.request_id)
    token = client_events.register(request.request_id) if request.request_id else None
    if not request.request_id:
        logger.warning("[cancel] POST /query has no request_id — cancel will not work for this query")

    task = asyncio.create_task(_atlasmind.generate_jql(request.query, jira_token=jira_token, jira_url=jira_url))
    if token:
        token.attach(task)

    try:
        llm_result, jira_result = await task
        return _build_response(llm_result, jira_result).model_dump()
    except asyncio.CancelledError:
        logger.info("Query cancelled by client: request_id=%s", request.request_id)
        return _error_response("Query cancelled.")
    except VllmUnavailable as exc:
        logger.error("Query failed — vLLM unavailable: %s", exc)
        return _error_response("LLM service is temporarily unavailable. Please try again later.")
    except ValueError as exc:
        logger.error("Query failed: %s", exc)
        return _error_response(str(exc))
    except Exception as exc:
        logger.exception("Query failed (unexpected): %s", exc)
        return _error_response(str(exc))
    finally:
        if request.request_id:
            client_events.unregister(request.request_id)


@app.post("/event", response_model=EventAck)
async def post_event(event: ClientEvent):
    """Receive a frontend event (e.g. cancel) for an in-flight query.

    The frontend sends this with the same request_id used in POST /query.
    Currently handled events:
        cancel    — cancels the running query task
        heartbeat — acknowledged, no server action
    """
    if event.event == ClientEventType.CANCEL:
        found = client_events.cancel(event.request_id)
        return EventAck(
            request_id=event.request_id,
            accepted=found,
            detail="cancellation requested" if found else "no active query for this request_id",
        )

    if event.event == ClientEventType.HEARTBEAT:
        return EventAck(request_id=event.request_id, accepted=True, detail="ok")

    return EventAck(request_id=event.request_id, accepted=False, detail="unhandled event type")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
