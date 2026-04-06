import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel
from document_processor import DocumentProcessor
from rag.jira_field_embeddings import Jira_Field_Embeddings
from rag.jql_embeddings import JQL_Embeddings
from core.ollama_client import OllamaClient
from dconfig import EmbeddingsConfig
from config.jira_config import load_active_profile, get_data_dir
from settings import DEFAULT_ANNOTATION_FILE, JIRA_FIELDS_FILENAME, SYSTEM_PROMPT_FILE, MAX_RESULTS

logger = logging.getLogger(__name__)

_LIMIT_RE = re.compile(r"\b(?:top|first|last|limit|show|list|get|fetch)?\s*(\d+)\s*(?:issues?|tickets?|results?|items?)?\b", re.IGNORECASE)
_JQL_LIMIT_RE = re.compile(r"\s+LIMIT\s+\d+\s*$", re.IGNORECASE)
_MAX_ALLOWED = 500


def _parse_limit(query: str) -> int:
    """Extract a numeric result limit from the user query, capped at _MAX_ALLOWED."""
    match = _LIMIT_RE.search(query)
    if match:
        return min(int(match.group(1)), _MAX_ALLOWED)
    return MAX_RESULTS


class JqlResponse(BaseModel):
    jql: str | None
    chart_spec: dict[str, Any] | None
    answer: str


def normalize_issue(jira_issue: dict) -> dict:
    """Flatten a raw Jira API issue into a snake_case dict for the frontend."""
    fields = jira_issue.get("fields", {})
    sprint = None
    sprint_field = fields.get("customfield_10020")
    if isinstance(sprint_field, list) and sprint_field:
        sprint = sprint_field[-1].get("name") if isinstance(sprint_field[-1], dict) else None

    return {
        "key":            jira_issue.get("key"),
        "summary":        fields.get("summary"),
        "description":    fields.get("description"),
        "status":         (fields.get("status") or {}).get("name"),
        "issuetype":      (fields.get("issuetype") or {}).get("name"),
        "priority":       (fields.get("priority") or {}).get("name"),
        "assignee":       (fields.get("assignee") or {}).get("displayName"),
        "reporter":       (fields.get("reporter") or {}).get("displayName"),
        "story_points":   fields.get("story_points") or fields.get("customfield_10016"),
        "epic_link":      fields.get("customfield_10014"),
        "parent":         (fields.get("parent") or {}).get("key"),
        "sprint":         sprint,
        "created":        fields.get("created"),
        "updated":        fields.get("updated"),
        "resolutiondate": fields.get("resolutiondate"),
        "duedate":        fields.get("duedate"),
        "labels":         fields.get("labels", []),
        "comments":       [
            {
                "author":  (c.get("author") or {}).get("displayName"),
                "body":    c.get("body"),
                "created": c.get("created"),
            }
            for c in (fields.get("comment") or {}).get("comments", [])
        ],
    }


class AtlasMind:
    def __init__(self, embedconfig: EmbeddingsConfig):
        self.embedconfig = embedconfig
        self.ollama_client = OllamaClient()
        self.document_processor = DocumentProcessor(embedconfig=embedconfig)
        self.system_prompt_dir = Path(SYSTEM_PROMPT_FILE)

        # Both embedding classes share the same DocumentProcessor so the
        # SentenceTransformer model is only loaded once.
        self.jql_embeddings = JQL_Embeddings(embedconfig, self.document_processor)
        self.jira_field_embeddings = Jira_Field_Embeddings(embedconfig, self.document_processor)

    def run(self):
        self.ollama_client.test_connection()
        self.jql_embeddings.run(Path(DEFAULT_ANNOTATION_FILE))

        profile = load_active_profile()
        fields_file = get_data_dir(profile["jira_url"]) / JIRA_FIELDS_FILENAME
        self.jira_field_embeddings.run(fields_file)

    async def _build_prompt(self, query: str) -> str:
        """Build a RAG-grounded prompt combining system instructions with retrieved context.

        Prepends the system prompt (which defines the JSON output contract and
        mode-switching logic) with retrieved Jira field vocabulary and semantically
        similar JQL examples. The context section only provides data — it does not
        restate output format rules already covered by the system prompt.

        Args:
            query: The user's natural language query string.

        Returns:
            str: system_prompt + RAG context + user request, ready to send to Ollama.
        """
        model = self.document_processor._model

        # search_sample_jql_embeddings_db is synchronous; search_jira_fields is async
        jql_examples, _ = self.jql_embeddings.search_sample_jql_embeddings_db(query, model)
        jira_fields, _ = await self.jira_field_embeddings.search_jira_fields(query, model)

        system_prompt = self.system_prompt_dir.read_text(encoding="utf-8")

        # rows: (id, field_id, field_name, field_type, is_custom, description, distance)
        # description (row[5]) is built by _build_description and already includes
        # allowed values, JQL clause names, and field type — use it directly.
        fields_block = "\n".join(
            f"  - {row[5]}"
            for row in jira_fields
        )

        # rows: (id, annotation, jql, distance)
        examples_block = "\n\n".join(
            f"  -- {row[1]}\n  {row[2]}"
            for row in jql_examples
        )

        context = (
            "\n\n"
            "## Available Jira Fields\n"
            "Use only these field IDs when building JQL — do not invent fields.\n"
            f"{fields_block}\n\n"
            "## JQL Rules\n"
            "1. Use only field IDs and allowed values listed above — do not invent fields or values.\n"
            "2. Do not use placeholder values like 'ProjectName' or 'USERNAME'.\n"
            "3. If no specific project is mentioned, omit the project filter.\n"
            "4. Do NOT use date arithmetic between two fields — JQL does not support it.\n"
            "   INVALID: resolutiondate >= created + 20d\n"
            "   INVALID: resolutiondate - created > 20d\n"
            "   CORRECT: resolution IS NOT EMPTY ORDER BY resolutiondate DESC\n"
            "   (duration filtering is handled externally — omit it from JQL)\n"
            "5. Do NOT append LIMIT — result count is controlled externally.\n"
            "6. Always end with ORDER BY unless the user specifies otherwise.\n\n"
            "## Similar JQL Examples\n"
            f"{examples_block}\n\n"
            "## User Request\n"
            f"{query}\n"
        )

        return system_prompt + context

    async def _execute_query(self, jql: str, max_results: int = MAX_RESULTS) -> dict:
        """Execute a JQL query against the active Jira instance.

        Args:
            jql: Valid JQL string produced by generate_jql().

        Returns:
            dict with keys: jql, raw_issues, total, shown.
        """
        profile = load_active_profile()
        base_url = profile["jira_url"].rstrip("/")
        email = profile.get("email", "")
        token = profile.get("token", "")
        auth = (email, token) if email and token else None

        url = f"{base_url}/rest/api/2/search"
        params = {
            "jql":        jql,
            "maxResults": max_results,
            "fields":     (
                "summary,description,status,assignee,reporter,priority,issuetype,"
                "created,updated,labels,parent,story_points,customfield_10016,"
                "customfield_10014,customfield_10020,resolutiondate,duedate,comment"
            ),
        }

        logger.info("Executing JQL against %s: %s", base_url, jql)
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    url, params=params, auth=auth,
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # Extract the error message from the Jira response body when available
            jira_error = ""
            try:
                body = exc.response.json()
                messages = body.get("errorMessages", [])
                errors = body.get("errors", {})
                jira_error = "; ".join(messages + list(errors.values()))
            except Exception:
                pass
            msg = jira_error or str(exc)
            logger.warning("Jira API error (HTTP %s): %s", exc.response.status_code, msg)
            raise ValueError(f"Jira rejected the JQL: {msg}") from exc
        except httpx.HTTPError as exc:
            logger.warning("Jira REST API call failed: %s", exc)
            raise ValueError(f"Jira connection failed: {exc}") from exc

        payload    = response.json()
        raw_issues = payload.get("issues", [])
        total      = payload.get("total", 0)
        logger.info("Jira returned %d / %d issues", len(raw_issues), total)
        return {"jql": jql, "raw_issues": raw_issues, "total": total, "shown": len(raw_issues)}

    async def generate_jql(self, query: str) -> tuple[JqlResponse, dict | None]:
        """Generate a JQL query (or general answer) from a natural language request.

        Builds a RAG-grounded prompt, sends it to Ollama, parses the JSON response,
        and executes the JQL against the active Jira instance when one is produced.

        Args:
            query: The user's natural language query string.

        Returns:
            tuple of:
                - JqlResponse — LLM result (jql, chart_spec, answer).
                - dict | None — Jira result from _execute_query(), or None for general answers.

        Raises:
            ValueError: If Ollama returns a response that cannot be parsed as JSON.
        """
        prompt = await self._build_prompt(query)
        raw = await self.ollama_client.generate_jql(prompt)
        logger.debug("Ollama raw response: %s", raw)

        try:
            data = json.loads(raw, strict=False)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Ollama response is not valid JSON: {raw!r}") from exc

        llm_result = JqlResponse(**data)

        logger.info("LLM generated output: %s", llm_result)
        jira_result = None
        if llm_result.jql:
            clean_jql = _JQL_LIMIT_RE.sub("", llm_result.jql).strip()
            if clean_jql != llm_result.jql:
                logger.info("JQL after LIMIT strip: %s", clean_jql)
            jira_result = await self._execute_query(clean_jql, max_results=_parse_limit(query))

        return llm_result, jira_result
