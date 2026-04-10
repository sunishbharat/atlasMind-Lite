import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx
from document_processor import DocumentProcessor
from rag.jira_field_embeddings import Jira_Field_Embeddings
from rag.jql_embeddings import JQL_Embeddings
from core.ollama_client import OllamaClient
from core.groq_client import GroqClient
from core.router import QueryRouter
from core.field_resolver import ExtraField, FieldResolver, ResolvedIntentFields
from core.models import JqlResponse
from dconfig import EmbeddingsConfig
from config.jira_config import load_active_profile, get_data_dir
from jira.jira_compute import enrich_issue
from settings import (
    DEFAULT_ANNOTATION_FILE,
    JIRA_FIELDS_FILENAME,
    MAX_INTENT_FIELDS,
    MAX_RESULTS,
    ROUTER_PROMPT_FILE,
    STANDARD_FIELD_IDS,
    SYSTEM_PROMPT_FILE,
)

logger = logging.getLogger(__name__)

# Requires a qualifying keyword before OR after the number to avoid matching
# sprint IDs, issue keys, and other bare numbers (e.g. "sprint 224").
# Groups: (prefix-qualified N) | (suffix-qualified N)
_LIMIT_RE = re.compile(
    r"\b(?:top|first|last|limit|show|list|get|fetch)\s+(\d+)\b"
    r"|\b(\d+)\s+(?:issues?|tickets?|results?|items?)\b",
    re.IGNORECASE,
)
# Strip quotes from purely-numeric values in JQL (e.g. Sprint in ('224') → Sprint in (224)).
_JQL_QUOTED_NUMBER_RE = re.compile(r"""(['"])(\d+)\1""")
_JQL_LIMIT_RE = re.compile(r"\s+LIMIT\s+\d+\s*$", re.IGNORECASE)
_JQL_ARITHMETIC_ORDER_RE = re.compile(r"\s+ORDER\s+BY\s+\S+\s*[-+]\s*.*$", re.IGNORECASE)
# Matches field-to-field date arithmetic in WHERE conditions, e.g.
#   AND resolutiondate > created + 20d
#   AND created + 5w < resolutiondate
# Does NOT match valid relative dates like: AND created > -20d (right side starts with -)
_JQL_FIELD_ARITH_COND_RE = re.compile(
    r"\s+AND\s+"
    r"(?:"
    r"\w+\s*(?:>=?|<=?|!=?)\s*[a-zA-Z]\w*\s*[-+]\s*\d+[a-zA-Z]+"   # field op field+Nd
    r"|"
    r"\w+\s*[-+]\s*\d+[a-zA-Z]+\s*(?:>=?|<=?|!=?)\s*\w+"            # field+Nd op field
    r"|"
    r"\w+\s*[-+]\s*\w+\s*(?:>=?|<=?|!=?)\s*\d+[a-zA-Z]+"            # field-field op Nd
    r")",
    re.IGNORECASE,
)
_MAX_ALLOWED = 500

# Matches AND field = 'value' or AND field != 'value' with quoted values.
_JQL_AND_EQUALITY_RE = re.compile(
    r"""\s+AND\s+([\w\[\]]+)\s*(!=|=)\s*['"]([^'"]+)['"]""",
    re.IGNORECASE,
)
# Matches AND field IN (...) or AND field NOT IN (...) with quoted values.
_JQL_AND_IN_RE = re.compile(
    r"""\s+AND\s+([\w\[\]]+)\s+(NOT\s+IN|IN)\s*\(([\s\S]*?)\)""",
    re.IGNORECASE,
)


def _validate_jql_values(jql: str, allowed: dict[str, list[str]]) -> str:
    """Strip JQL conditions that reference values not in a field's allowed set.

    Only fields present in ``allowed`` are validated — fields without discrete
    options (dates, free-text, numbers) are passed through unchanged.

    Handles:
    - ``AND field = 'value'``   — strip entire condition if value is invalid
    - ``AND field != 'value'``  — strip entire condition if value is invalid
    - ``AND field IN (...)``    — drop invalid values; strip condition if all invalid
    - ``AND field NOT IN (...)``— drop invalid values; strip condition if all invalid

    Args:
        jql:     JQL string (post-processed by earlier strip steps).
        allowed: {field_id: [canonical_value, ...]} from fetch_allowed_values().

    Returns:
        JQL with invalid-value conditions removed.
    """
    if not allowed:
        return jql

    normed: dict[str, set[str]] = {
        fid.lower(): {v.lower() for v in vals}
        for fid, vals in allowed.items()
    }

    def _check_equality(m: re.Match) -> str:
        field = m.group(1)
        field_key = field.lower()
        value = m.group(3)
        if field_key not in normed:
            return m.group(0)
        if value.lower() in normed[field_key]:
            return m.group(0)
        logger.warning(
            "JQL: value %r is not valid for field %r — stripping condition "
            "(allowed sample: %s)",
            value, field, list(allowed.get(field_key, []))[:5],
        )
        return ""

    def _check_in(m: re.Match) -> str:
        field = m.group(1)
        field_key = field.lower()
        if field_key not in normed:
            return m.group(0)
        in_keyword = m.group(2)
        # Parse values whether quoted or unquoted: 'Bug', "Story", Epic
        raw_values = [v.strip().strip("'\"") for v in m.group(3).split(',') if v.strip()]
        valid = [v for v in raw_values if v.lower() in normed[field_key]]
        invalid = [v for v in raw_values if v.lower() not in normed[field_key]]
        if invalid:
            logger.warning(
                "JQL: invalid values for field %r %s clause — dropping: %s",
                field, in_keyword.upper(), invalid,
            )
        if not valid:
            logger.warning(
                "JQL: all values invalid for field %r %s clause — stripping condition",
                field, in_keyword.upper(),
            )
            return ""
        valid_str = ", ".join(valid)
        return f" AND {field} {in_keyword} ({valid_str})"

    result = _JQL_AND_EQUALITY_RE.sub(_check_equality, jql)
    result = _JQL_AND_IN_RE.sub(_check_in, result)
    return result.strip()


def _parse_limit(query: str) -> int:
    """Extract a numeric result limit from the user query, capped at _MAX_ALLOWED."""
    match = _LIMIT_RE.search(query)
    if match:
        n = next(g for g in match.groups() if g is not None)
        return min(int(n), _MAX_ALLOWED)
    return MAX_RESULTS


# Maps Jira field_id → the key used in the normalized issue dict.
# Used to filter normalize_issue output to only requested fields.
_FIELD_ID_TO_OUTPUT_KEY: dict[str, str] = {
    "key":               "key",
    "summary":           "summary",
    "description":       "description",
    "status":            "status",
    "issuetype":         "issuetype",
    "priority":          "priority",
    "assignee":          "assignee",
    "reporter":          "reporter",
    "customfield_10016": "story_points",
    "customfield_10014": "epic_link",
    "parent":            "parent",
    "customfield_10020": "sprint",
    "created":           "created",
    "updated":           "updated",
    "resolutiondate":    "resolutiondate",
    "duedate":           "duedate",
    "labels":            "labels",
    "comment":           "comments",
}


_SPRINT_TOSTRING_NAME_RE = re.compile(r"\bname=([^,\]]+)")


def _extract_field_value(v: Any) -> Any:
    """Return a human-readable scalar from a raw Jira field entry.

    - dict  → name / value / displayName
    - Sprint toString string → name= segment (e.g. "Usergrid 34")
    - primitive → as-is
    """
    if isinstance(v, dict):
        return v.get("name") or v.get("value") or v.get("displayName")
    if isinstance(v, str):
        m = _SPRINT_TOSTRING_NAME_RE.search(v)
        if m:
            return m.group(1).strip()
    return v


def normalize_issue(
    jira_issue: dict,
    extra_fields: list[ExtraField] | None = None,
    requested_ids: set[str] | None = None,
) -> dict:
    """Flatten a raw Jira API issue into a snake_case dict for the frontend.

    Standard fields are always extracted. Any extra_fields resolved from intent
    are extracted generically: object types yield their .name/.value/.displayName,
    lists are flattened to a list of names, primitives are passed through as-is.

    Args:
        jira_issue: Raw issue dict from the Jira REST API.
        extra_fields: Optional list of ExtraField resolved from LLM intent_fields.
        requested_ids: When provided, output is filtered to only include fields
            whose field_id is in this set (plus ``key`` which is always included).
            Fields not in requested_ids are omitted rather than returned as null.

    Returns:
        Flat dict with only the requested fields (and any extra_fields appended).
    """
    fields = jira_issue.get("fields", {})

    sprint = None
    sprint_field = fields.get("customfield_10020")
    if isinstance(sprint_field, list) and sprint_field:
        sprint = sprint_field[-1].get("name") if isinstance(sprint_field[-1], dict) else None

    issue: dict[str, Any] = {
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
        "comments": [
            {
                "author":  (c.get("author") or {}).get("displayName"),
                "body":    c.get("body"),
                "created": c.get("created"),
            }
            for c in (fields.get("comment") or {}).get("comments", [])
        ],
    }
    issue.update(enrich_issue(fields))

    for ef in (extra_fields or []):
        if ef.display_name in issue:
            logger.debug(
                "intent_field %r skipped — key already exists in normalized issue",
                ef.display_name,
            )
            continue
        raw = fields.get(ef.field_id)
        if isinstance(raw, list):
            issue[ef.display_name] = [_extract_field_value(v) for v in raw]
        else:
            issue[ef.display_name] = _extract_field_value(raw)

    if requested_ids is not None:
        keep: set[str] = {"key"}
        for fid in requested_ids:
            out_key = _FIELD_ID_TO_OUTPUT_KEY.get(fid)
            if out_key:
                keep.add(out_key)
        # Retain computed enrichment fields when the dates they depend on are present.
        if {"created", "resolutiondate"} & requested_ids:
            keep.update(("effort_days", "effort_hours", "age_days"))
        # Always keep extra_field display names — they were explicitly requested.
        for ef in (extra_fields or []):
            keep.add(ef.display_name)
        issue = {k: v for k, v in issue.items() if k in keep}

    return issue


class AtlasMind:
    def __init__(self, embedconfig: EmbeddingsConfig, llm_backend: str = "ollama"):
        self.embedconfig = embedconfig
        self.llm_backend = llm_backend
        self.field_resolver: FieldResolver | None = None
        # Populated in run() after FieldResolver validates against the vector DB.
        # Only these fields are always requested; intent fields are added per query.
        self.standard_field_ids: list[str] = list(STANDARD_FIELD_IDS)
        # {field_id: [allowed_value_strings]} — populated in run() from the vector DB.
        self.allowed_values: dict[str, list[str]] = {}

        if llm_backend == "groq":
            self.llm_client = GroqClient()
        else:
            self.llm_client = OllamaClient()

        self.router = QueryRouter(self.llm_client, Path(ROUTER_PROMPT_FILE))
        self.document_processor = DocumentProcessor(embedconfig=embedconfig)
        self.system_prompt_dir = Path(SYSTEM_PROMPT_FILE)

        self.jql_embeddings = JQL_Embeddings(embedconfig, self.document_processor)
        self.jira_field_embeddings = Jira_Field_Embeddings(embedconfig, self.document_processor)

    def run(self) -> None:
        self.llm_client.test_connection()
        self.jql_embeddings.run(Path(DEFAULT_ANNOTATION_FILE))

        profile = load_active_profile()
        fields_file = get_data_dir(profile["jira_url"]) / JIRA_FIELDS_FILENAME
        self.jira_field_embeddings.run(fields_file)

        # Build FieldResolver from DB mappings — single query covers both
        # intent field name resolution and field ID validation.
        name_to_id, id_to_name = self.jira_field_embeddings.fetch_field_mappings()
        self.field_resolver = FieldResolver.from_db_mappings(
            name_to_id, id_to_name, max_intent_fields=MAX_INTENT_FIELDS
        )

        # Validate configured standard field IDs against the vector DB.
        # Reuses id_to_name fetched above — no second DB query needed.
        known_ids: set[str] = set(id_to_name.keys())
        self.standard_field_ids = self.field_resolver.validate_field_ids(STANDARD_FIELD_IDS, known_ids)

        # Load allowed values for JQL condition validation before API execution.
        self.allowed_values = self.jira_field_embeddings.fetch_allowed_values()

    async def _build_prompt(self, query: str) -> str:
        """Build a RAG-grounded prompt combining system instructions with retrieved context."""
        model = self.document_processor._model

        jql_examples, _ = self.jql_embeddings.search_sample_jql_embeddings_db(query, model)
        jira_fields, _ = await self.jira_field_embeddings.search_jira_fields(query, model)

        system_prompt = self.system_prompt_dir.read_text(encoding="utf-8")

        fields_block = "\n".join(f"  - {row[5]}" for row in jira_fields)
        examples_block = "\n\n".join(
            f"  -- {row[1]}\n  {row[2]}" for row in jql_examples
        )

        context = (
            "\n\n"
            "## Available Jira Fields\n"
            "Populate intent_fields using ONLY the display names listed here, copied verbatim.\n"
            "Do NOT use field IDs (e.g. 'issuetype', 'resolutiondate') — use the display name (e.g. 'Issue Type', 'Resolved').\n"
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

    async def _execute_query(
        self,
        jql: str,
        max_results: int = MAX_RESULTS,
        extra_field_ids: list[str] | None = None,
    ) -> dict:
        """Execute a JQL query against the active Jira instance.

        Args:
            jql: Valid JQL string produced by generate_jql().
            max_results: Maximum number of issues to return.
            extra_field_ids: Additional Jira field IDs to request beyond the base set.

        Returns:
            dict with keys: jql, raw_issues, total, shown.
        """
        profile = load_active_profile()
        base_url = profile["jira_url"].rstrip("/")
        email = profile.get("email", "")
        token = profile.get("token", "")
        auth = (email, token) if email and token else None

        all_fields = self.field_resolver.build_fields_param(
            self.standard_field_ids, extra_field_ids
        ) if self.field_resolver else ",".join(self.standard_field_ids)

        url = f"{base_url}/rest/api/2/search"
        params = {
            "jql":        jql,
            "maxResults": max_results,
            "fields":     all_fields,
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

        payload = response.json()
        raw_issues = payload.get("issues", [])
        total = payload.get("total", 0)
        logger.info("Jira returned %d / %d issues", len(raw_issues), total)
        return {"jql": jql, "raw_issues": raw_issues, "total": total, "shown": len(raw_issues)}

    async def generate_jql(self, query: str) -> tuple[JqlResponse, dict | None]:
        """Generate a JQL query (or general answer) from a natural language request.

        Stage 1: fast route via QueryRouter — classifies as JQL or general answer.
        Stage 2 (JQL path only): full RAG pipeline, LLM call, intent field resolution,
        Jira API execution.

        Args:
            query: The user's natural language query string.

        Returns:
            tuple of:
                - JqlResponse — LLM result (jql, chart_spec, answer, intent_fields).
                - dict | None — Jira result including resolved_intent_fields, or None for general answers.

        Raises:
            ValueError: If the LLM returns a response that cannot be parsed as JSON.
        """
        logger.info("User query: %s", query)
        route = await self.router.route(query)
        if not route.is_jql:
            return JqlResponse(jql=None, chart_spec=None, answer=route.answer), None

        prompt = await self._build_prompt(query)
        raw = await self.llm_client.generate_jql(prompt)
        logger.debug("LLM raw response: %s", raw)

        try:
            data = json.loads(raw, strict=False)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM response is not valid JSON: {raw!r}") from exc

        llm_result = JqlResponse(**data)
        logger.info("LLM generated output: %s", llm_result)

        jira_result = None
        if llm_result.jql:
            clean_jql = _JQL_LIMIT_RE.sub("", llm_result.jql).strip()
            if clean_jql != llm_result.jql:
                logger.info("JQL after LIMIT strip: %s", clean_jql)

            stripped = _JQL_ARITHMETIC_ORDER_RE.sub("", clean_jql).strip()
            if stripped != clean_jql:
                logger.warning(
                    "JQL contained arithmetic in ORDER BY — stripped: %s", clean_jql
                )
                clean_jql = stripped

            stripped = _JQL_FIELD_ARITH_COND_RE.sub("", clean_jql).strip()
            if stripped != clean_jql:
                logger.warning(
                    "JQL contained field-to-field date arithmetic in WHERE (unsupported by Jira) — stripped: %s",
                    clean_jql,
                )
                clean_jql = stripped

            unquoted = _JQL_QUOTED_NUMBER_RE.sub(r"\2", clean_jql)
            if unquoted != clean_jql:
                logger.info("JQL after numeric dequote: %s", unquoted)
                clean_jql = unquoted

            validated = _validate_jql_values(clean_jql, self.allowed_values)
            if validated != clean_jql:
                logger.info("JQL after allowed-values validation: %s", validated)
                clean_jql = validated

            resolved = (
                self.field_resolver.resolve(llm_result.intent_fields)
                if self.field_resolver
                else ResolvedIntentFields()
            )

            jira_result = await self._execute_query(
                jql=clean_jql,
                max_results=_parse_limit(query),
                extra_field_ids=resolved.field_ids or None,
            )
            jira_result["resolved_intent_fields"] = resolved

        return llm_result, jira_result
