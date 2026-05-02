import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from document_processor import DocumentProcessor
from rag.jira_field_embeddings import Jira_Field_Embeddings
from rag.jql_embeddings import JQL_Embeddings
from core.ollama_client import OllamaClient
from core.groq_client import GroqClient
from core.vllm_client import VllmClient, VllmUnavailable
from core.claude_client import ClaudeClient
from core.bedrock_claude_client import BedrockClaudeClient
from core.router import QueryRouter
from core.chart_spec_generator import ChartSpecGenerator
from core.field_resolver import ExtraField, FieldResolver, ResolvedIntentFields
from core.jql_sanitizer import JqlSanitizer, SanitizeResult
from core.models import JqlResponse, RouteResult, TokenUsage
from dconfig import EmbeddingsConfig
from config.jira_config import get_data_dir, load_active_jira_profile
from jira.jira_compute import enrich_issue
from jira.jira_search import JiraSearchClient, JiraSearchRequest
from rag.jira_field_value_embeddings import JiraFieldValueEmbeddings
from settings import (
    CHART_SPEC_PROMPT_FILE,
    DEFAULT_ANNOTATION_FILE,
    JIRA_ALLOWED_VALUES_FILENAME,
    JIRA_FIELDS_FILENAME,
    JQL_MAX_ATTEMPTS,
    JQL_RETRY_FIELD_TEMPLATE,
    JQL_RETRY_FIELDS_TEMPLATE,
    JQL_RETRY_TEMPLATE,
    JQL_RETRY_VALUE_HINTS_TEMPLATE,
    MAX_INTENT_FIELDS,
    MAX_JIRA_RESULTS,
    MAX_RESULTS,
    ROUTER_PROMPT_FILE,
    ROUTER_PROMPT_FILE_OLLAMA,
    STANDARD_FIELD_IDS,
    SYSTEM_PROMPT_FILE,
    VLLM_FALLBACK,
)

logger = logging.getLogger(__name__)

# Requires a qualifying keyword before OR after the number to avoid matching
# sprint IDs, issue keys, and other bare numbers (e.g. "sprint 224").
# Negative lookahead excludes time quantities ("last 20 days", "first 7 weeks").
# Groups: (prefix-qualified N) | (suffix-qualified N)
_LIMIT_RE = re.compile(
    r"\b(?:top|first|last|limit|show|list|get|fetch)\s+(\d+)\b(?!\s+(?:days?|weeks?|months?|years?|hours?|minutes?))"
    r"|\b(\d+)\s*(?:issues?|tickets?|results?|items?)\b",
    re.IGNORECASE,
)
# Used in _handle_raw_query to strip LIMIT before direct JQL execution.
_JQL_LIMIT_RE = re.compile(r"\s+LIMIT\s+\d+", re.IGNORECASE)


_JIRA_ERROR_FIELD_RE = re.compile(
    r"Field '([\w.]+)' does not exist"
    r"|Field '([\w.]+)' is not searchable"
    r"|'(\w+)' is a reserved JQL word",
    re.IGNORECASE,
)

_JIRA_VALUE_ERROR_RE = re.compile(
    r"The value '([^']+)' does not exist for the field '([\w\[\].]+)'",
    re.IGNORECASE,
)

_JIRA_UNSUPPORTED_OP_RE = re.compile(
    r"The field '([\w.]+)' does not support searching for (?:EMPTY values|an empty string)",
    re.IGNORECASE,
)

_JIRA_UNSUPPORTED_OPERATOR_RE = re.compile(
    r"The operator '[^']+' is not supported by the '([\w.]+)' field",
    re.IGNORECASE,
)


def _extract_json_object(raw: str) -> str:
    """Extract the first complete JSON object from raw LLM output.

    Small models often append extra content after the closing brace, or leak
    keys outside the object. Tracks brace depth to find the exact end of the
    first well-formed object rather than relying on rfind('}').
    """
    start = raw.find("{")
    if start == -1:
        return raw
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(raw[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]
    return raw[start:]


_JQL_VALUE_DQUOTE_RE = re.compile(
    r'("jql"\s*:\s*")(.*?)("(?:\s*,\s*"(?:chart_spec|answer|intent_fields)"|}))',
    re.DOTALL,
)


def _repair_jql_quotes(raw: str) -> str:
    """Replace unescaped double quotes inside the jql value with single quotes.

    Small models sometimes emit  status IN ("Done")  instead of  status IN ('Done'),
    which breaks the JSON wrapper. This replaces bare " inside the jql string value
    with ' so json.loads can succeed.
    """
    def _fix_value(m: re.Match) -> str:
        prefix, jql_body, suffix = m.group(1), m.group(2), m.group(3)
        fixed = jql_body.replace('\\"', '\x00').replace('"', "'").replace('\x00', '\\"')
        return prefix + fixed + suffix

    repaired = _JQL_VALUE_DQUOTE_RE.sub(_fix_value, raw, count=1)
    return repaired


def _extract_error_fields(error_msg: str) -> list[str]:
    """Return all invalid field names found in a Jira error message."""
    return [
        next(g for g in m.groups() if g is not None)
        for m in _JIRA_ERROR_FIELD_RE.finditer(error_msg)
    ]


def _extract_value_errors(error_msg: str) -> list[tuple[str, str]]:
    """Return (field, value) pairs for 'value does not exist for field' Jira errors."""
    return [(m.group(2), m.group(1)) for m in _JIRA_VALUE_ERROR_RE.finditer(error_msg)]


def _extract_unsupported_op_fields(error_msg: str) -> list[str]:
    """Return field names from 'does not support searching for EMPTY/empty string' errors."""
    return [m.group(1) for m in _JIRA_UNSUPPORTED_OP_RE.finditer(error_msg)]


def _extract_unsupported_operator_fields(error_msg: str) -> list[str]:
    """Return field names from 'operator X is not supported by field Y' errors."""
    return [m.group(1) for m in _JIRA_UNSUPPORTED_OPERATOR_RE.finditer(error_msg)]


def _strip_field_conditions(jql: str, fields: list[str]) -> str:
    """Remove all WHERE conditions involving the given field IDs from JQL.

    Handles both AND-preceded conditions and leading conditions (first in WHERE).
    """
    result = jql
    for field in fields:
        fp = re.escape(field)
        # AND-preceded condition (middle or trailing)
        result = re.sub(
            rf"\s+AND\s+{fp}\s+(?:NOT\s+IN|IN)\s*\([^)]*\)"
            rf"|\s+AND\s+{fp}\s*(?:!=|=)\s*(?:'[^']*'|\"[^\"]*\"|\S+)",
            "",
            result,
            flags=re.IGNORECASE,
        )
        # Leading condition — strip field+op+value and consume the following AND if present
        result = re.sub(
            rf"^{fp}\s+(?:NOT\s+IN|IN)\s*\([^)]*\)\s*(?:AND\s+)?"
            rf"|^{fp}\s*(?:!=|=)\s*(?:'[^']*'|\"[^\"]*\"|\S+)\s*(?:AND\s+)?",
            "",
            result.strip(),
            flags=re.IGNORECASE,
        )
    return result.strip()






def _truncate_field_desc(desc: str, max_chars: int = 500) -> str:
    """Truncate a field description to max_chars while preserving the JQL clause and field ID.

    Blindly slicing at max_chars can drop the 'Used in JQL as' and 'Field ID' footer,
    which are the parts the LLM needs most for generating correct JQL. Instead, truncate
    only the allowed-values section in the middle and always keep the footer intact.
    """
    if len(desc) <= max_chars:
        return desc
    jql_idx = desc.find(" Used in JQL")
    if jql_idx != -1:
        footer = desc[jql_idx:]
        keep = max_chars - len(footer)
        if keep > 20:
            return desc[:keep].rstrip(", ") + " ..." + footer
    return desc[:max_chars]


def _parse_limit(query: str) -> int:
    """Extract a numeric result limit from the user query.

    If MAX_JIRA_RESULTS is set (non-zero), the parsed value is capped at that limit.
    """
    match = _LIMIT_RE.search(query)
    if match:
        n = int(next(g for g in match.groups() if g is not None))
        return min(n, MAX_JIRA_RESULTS) if MAX_JIRA_RESULTS else n
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
        # Populated in run() — None until startup completes.
        self.jql_sanitizer: JqlSanitizer | None = None
        self.field_value_embeddings: JiraFieldValueEmbeddings | None = None

        self._init_llm_backend(llm_backend)
        self.document_processor = DocumentProcessor(embedconfig=embedconfig)
        self.system_prompt_dir = Path(SYSTEM_PROMPT_FILE)

        self.jql_embeddings = JQL_Embeddings(embedconfig, self.document_processor)
        self.jira_field_embeddings = Jira_Field_Embeddings(embedconfig, self.document_processor)

    def _init_llm_backend(self, backend: str) -> None:
        self.llm_backend = backend
        if backend == "groq":
            self.llm_client = GroqClient()
            self.router = QueryRouter(self.llm_client, Path(ROUTER_PROMPT_FILE))
        elif backend == "vllm":
            self.llm_client = VllmClient()
            self.router = QueryRouter(self.llm_client, Path(ROUTER_PROMPT_FILE))
        elif backend == "claude":
            self.llm_client = ClaudeClient()
            self.router = QueryRouter(self.llm_client, Path(ROUTER_PROMPT_FILE))
        elif backend == "bedrock":
            self.llm_client = BedrockClaudeClient()
            self.router = QueryRouter(self.llm_client, Path(ROUTER_PROMPT_FILE))
        else:
            self.llm_client = OllamaClient()
            self.router = QueryRouter(self.llm_client, Path(ROUTER_PROMPT_FILE_OLLAMA), two_pass=True)
        self.chart_spec_generator = ChartSpecGenerator(self.llm_client, Path(CHART_SPEC_PROMPT_FILE))

    def run(self) -> None:
        try:
            self.llm_client.test_connection()
        except VllmUnavailable as exc:
            logger.warning("vLLM unavailable (%s) — falling back to %s", exc, VLLM_FALLBACK)
            self._init_llm_backend(VLLM_FALLBACK)
            self.llm_client.test_connection()
        self.jql_embeddings.run(Path(DEFAULT_ANNOTATION_FILE))

        profile = load_active_jira_profile()
        fields_file = get_data_dir(profile.jira_url) / JIRA_FIELDS_FILENAME
        self.jira_field_embeddings.run(fields_file)

        # Build FieldResolver from DB mappings — single query covers both
        # intent field name resolution and field ID validation.
        name_to_id, id_to_name = self.jira_field_embeddings.fetch_field_mappings()
        self.field_resolver = FieldResolver.from_db_mappings(
            name_to_id, id_to_name, max_intent_fields=MAX_INTENT_FIELDS,
            fuzzy_field_fn=lambda name: self.jira_field_embeddings.find_similar_field_name(
                name, self.document_processor._model
            ),
        )

        # Validate configured standard field IDs against the vector DB.
        # Reuses id_to_name fetched above — no second DB query needed.
        known_ids: set[str] = set(id_to_name.keys())
        self.standard_field_ids = self.field_resolver.validate_field_ids(STANDARD_FIELD_IDS, known_ids)

        # Load allowed values for JQL condition validation before API execution.
        self.allowed_values = self.jira_field_embeddings.fetch_allowed_values()

        # Seed per-value embeddings and construct the sanitizer.
        av_file = get_data_dir(profile.jira_url) / JIRA_ALLOWED_VALUES_FILENAME
        self.field_value_embeddings = JiraFieldValueEmbeddings(self.embedconfig, self.document_processor)
        self.field_value_embeddings.setup_table()
        self.field_value_embeddings.seed(self.allowed_values, id_to_name, av_file)

        self.jql_sanitizer = JqlSanitizer(
            name_to_id=name_to_id,
            id_to_name=id_to_name,
            allowed_values=self.allowed_values,
            field_value_embeddings=self.field_value_embeddings,
            model=self.document_processor._model,
        )

    async def _build_prompt(self, query: str) -> tuple[str, list[str]]:
        """Build a RAG-grounded prompt combining system instructions with retrieved context.

        Encodes the query once and runs two parallel lookups:
        - jira_fields vector search → top relevant fields and their descriptions
        - jira_field_values search → top-N query-relevant allowed values per field

        The per-field value hints are injected alongside the field description so
        the LLM sees the most contextually relevant values upfront, before generation.
        This is the pre-generation complement to the sanitizer's post-generation
        correction — reducing hallucinations on the first pass rather than fixing them.

        Returns:
            tuple of:
                - full prompt string to send to the LLM
                - list of field IDs from the vector search (fetched back from Jira REST API)
        """
        model = self.document_processor._model

        jql_examples, _ = self.jql_embeddings.search_sample_jql_embeddings_db(query, model)
        jira_fields, _ = await self.jira_field_embeddings.search_jira_fields(query, model)

        # row = (id, field_id, field_name, field_type, is_custom, description, distance)
        rag_field_ids: list[str] = [row[1] for row in jira_fields]

        # Encode the user query once and reuse across all per-field value searches.
        # Offloaded to a thread executor so model.encode does not block the event loop.
        loop = asyncio.get_event_loop()
        query_emb = await loop.run_in_executor(
            None,
            lambda: model.encode(query, normalize_embeddings=True),
        )

        # For each relevant field with discrete allowed values, retrieve the top-N
        # values semantically closest to the user query. These are injected into the
        # field description line so the LLM sees query-ranked values before generation
        # rather than only the first N values from the truncated description.
        value_hints: dict[str, list[str]] = {}
        if self.field_value_embeddings:
            for row in jira_fields:
                field_id = row[1]
                if field_id in self.allowed_values:
                    similar = self.field_value_embeddings.find_similar_values_by_embedding(
                        field_id=field_id,
                        query_embedding=query_emb,
                    )
                    if similar:
                        value_hints[field_id] = [s.value for s in similar]

        if value_hints:
            logger.info(
                "Value hints injected for %d field(s): %s",
                len(value_hints),
                {fid: vals for fid, vals in value_hints.items()},
            )

        system_prompt = self.system_prompt_dir.read_text(encoding="utf-8")

        def _field_line(row: tuple) -> str:
            desc = _truncate_field_desc(row[5])
            hints = value_hints.get(row[1])
            if hints:
                return f"  - {desc} [Query-relevant values: {', '.join(hints)}]"
            return f"  - {desc}"

        fields_block = "\n".join(_field_line(row) for row in jira_fields)
        examples_block = "\n\n".join(
            f"  -- {row[1]}\n  {row[2]}" for row in jql_examples
        )

        context = (
            "\n\n"
            "## Available Jira Fields\n"
            "Populate intent_fields using ONLY the display names listed here, copied verbatim.\n"
            "The display name is the text BEFORE the first colon in each line (e.g. 'Fix Version/s', not 'fixVersion' or 'fixVersions').\n"
            "Do NOT use JQL clause names or field IDs — only the display name before the colon.\n"
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
            "6. Always end with ORDER BY using a field ID from ## Available Jira Fields above — never use issueFunction or any field not listed there.\n\n"
            "## Similar JQL Examples\n"
            f"{examples_block}\n\n"
            "## User Request\n"
            f"{query}\n"
        )

        full_prompt = system_prompt + context
        token_usage = TokenUsage(
            system_tokens=len(system_prompt) // 4,
            fields_tokens=len(fields_block) // 4,
            examples_tokens=len(examples_block) // 4,
            total_tokens=len(full_prompt) // 4,
        )
        logger.info(
            "Prompt sizes — system: %d chars (~%d tokens), fields_block: %d chars (~%d tokens), "
            "examples_block: %d chars (~%d tokens), total: %d chars (~%d tokens)",
            len(system_prompt), token_usage.system_tokens,
            len(fields_block), token_usage.fields_tokens,
            len(examples_block), token_usage.examples_tokens,
            len(full_prompt), token_usage.total_tokens,
        )
        return full_prompt, rag_field_ids, token_usage

    async def _execute_query(
        self,
        jql: str,
        max_results: int = MAX_RESULTS,
        extra_field_ids: list[str] | None = None,
        jira_token: str | None = None,
        jira_url: str | None = None,
    ) -> dict:
        """Execute a JQL query against the active Jira instance with automatic pagination.

        Args:
            jql: Valid JQL string produced by generate_jql().
            max_results: Maximum number of issues to return (pages automatically if > 1000).
            extra_field_ids: Additional Jira field IDs to request beyond the base set.
            jira_token: Per-request PAT from X-Jira-Token header. Takes precedence
                over the profile-configured token.
            jira_url: Per-request Jira base URL from X-Jira-Url header. Takes precedence
                over the profile-configured jira_url.

        Returns:
            dict with keys: jql, raw_issues, total, shown.
        """
        profile = load_active_jira_profile()
        if jira_url:
            parsed = urlparse(jira_url)
            if parsed.scheme in ("http", "https") and parsed.netloc:
                base_url = jira_url.rstrip("/")
            else:
                logger.warning("X-Jira-Url %r is not a valid URL — falling back to profile URL", jira_url)
                base_url = profile.jira_url
        else:
            base_url = profile.jira_url
        logger.info("Jira base URL: %s (source: %s)", base_url, "header" if jira_url and base_url == jira_url.rstrip("/") else "profile")
        credential = profile.resolve_auth(token_override=jira_token)
        auth, auth_headers = credential.auth, credential.headers

        all_fields = self.field_resolver.build_fields_param(
            self.standard_field_ids, extra_field_ids
        ) if self.field_resolver else ",".join(self.standard_field_ids)

        client = JiraSearchClient()
        jql_error = await client.validate_jql(jql, base_url, auth, auth_headers)
        if jql_error:
            logger.warning("JQL validation failed: %s | JQL: %s", jql_error, jql)
            raise ValueError(f"Jira rejected the JQL: {jql_error}")

        logger.info("Executing JQL against %s: %s", base_url, jql)
        result = await client.search(
            JiraSearchRequest(
                jql=jql,
                fields=all_fields,
                max_results=max_results,
                base_url=base_url,
                auth=auth,
                auth_headers=auth_headers,
            )
        )
        return {"jql": jql, "raw_issues": result.issues, "total": result.total, "shown": result.fetched}

    async def _handle_raw_query(self, route: RouteResult, jira_token: str | None = None, jira_url: str | None = None) -> tuple[JqlResponse, dict]:
        """Execute a user-supplied JQL string directly, bypassing RAG and LLM generation.

        Only the LIMIT clause is stripped (unsupported by the Jira REST API).
        All other sanitization and validation steps are skipped — the user owns this JQL.

        If a chart_hint is present (text right of /raw), a focused LLM call produces
        the chart_spec; otherwise chart_spec is null.

        Args:
            route: RouteResult with type="raw", raw_jql, and optional chart_hint.

        Returns:
            tuple of (JqlResponse, jira_result dict).
        """
        jql = _JQL_LIMIT_RE.sub("", route.raw_jql).strip()
        logger.info("*** Raw JQL: %s", jql)

        chart_spec_dict: dict | None = None
        if route.chart_hint:
            spec = await self.chart_spec_generator.generate(route.chart_hint)
            if spec:
                chart_spec_dict = spec.model_dump()

        jira_result = await self._execute_query(
            jql=jql,
            max_results=_parse_limit(route.chart_hint or ""),
            jira_token=jira_token,
            jira_url=jira_url,
        )
        jira_result["resolved_intent_fields"] = ResolvedIntentFields()

        return JqlResponse(jql=jql, chart_spec=chart_spec_dict), jira_result

    async def generate_jql(self, query: str, jira_token: str | None = None, jira_url: str | None = None) -> tuple[JqlResponse, dict | None]:
        """Generate a JQL query (or general answer) from a natural language request.

        Stage 1: fast route via QueryRouter — classifies as JQL, general, or raw.
        Stage 2a (raw path): user JQL sent verbatim; optional chart_spec via LLM.
        Stage 2b (JQL path): full RAG pipeline, LLM call, intent field resolution,
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
        logger.info("*** User query: %s", query)
        route = await self.router.route(query)

        if route.is_raw:
            return await self._handle_raw_query(route, jira_token=jira_token, jira_url=jira_url)

        if not route.is_jql:
            logger.info("*** AI answer: %s", route.answer)
            return JqlResponse(jql=None, chart_spec=None, answer=route.answer), None

        prompt, rag_field_ids, token_usage = await self._build_prompt(query)
        raw = await self.llm_client.generate_jql(prompt)
        logger.debug("LLM raw response: %s", raw)

        try:
            data = json.loads(_extract_json_object(raw), strict=False)
        except json.JSONDecodeError:
            try:
                data = json.loads(_extract_json_object(_repair_jql_quotes(raw)), strict=False)
                logger.warning("LLM response had unescaped quotes in jql — repaired.")
            except json.JSONDecodeError as exc:
                _err = ValueError(f"LLM response is not valid JSON: {raw!r}")
                _err.token_usage = token_usage  # type: ignore[attr-defined]
                raise _err from exc

        llm_result = JqlResponse(**data)
        logger.info("*** AI JQL: %s", llm_result.jql)
        logger.info("*** AI answer: %s", llm_result.answer)

        if not llm_result.jql and not llm_result.answer:
            logger.warning("LLM returned null jql and null answer — returning fallback")
            return JqlResponse(
                answer="The model could not generate a response for this query. Try rephrasing or use the /raw flag with explicit JQL."
            ), None

        jira_result = None
        if llm_result.jql:
            sanitized = self._sanitize_jql(llm_result.jql)
            current_jql = sanitized.jql
            pending_hints = sanitized.hints

            resolved = (
                self.field_resolver.resolve(llm_result.intent_fields)
                if self.field_resolver
                else ResolvedIntentFields()
            )

            # Include both intent fields (LLM-proposed display columns) and the
            # field IDs surfaced by the vector search — these are the fields the
            # LLM had available when writing the JQL, so they must be fetched back
            # from Jira even if the LLM didn't explicitly list them in intent_fields.
            combined_extra: list[str] = list(
                dict.fromkeys((resolved.field_ids or []) + rag_field_ids)
            )
            max_results = _parse_limit(query)
            accumulated_prompt = prompt  # grows with each retry block

            for attempt in range(1, JQL_MAX_ATTEMPTS + 1):
                try:
                    jira_result = await self._execute_query(
                        jql=current_jql,
                        max_results=max_results,
                        extra_field_ids=combined_extra or None,
                        jira_token=jira_token,
                        jira_url=jira_url,
                    )
                    break
                except ValueError as exc:
                    if attempt == JQL_MAX_ATTEMPTS:
                        exc.token_usage = token_usage  # type: ignore[attr-defined]
                        raise
                    retry_num = attempt
                    logger.warning("JQL attempt %d failed — retry[%d] with error context", attempt, retry_num)
                    logger.warning("  retry[%d] Bad JQL : %s", retry_num, current_jql)
                    logger.warning("  retry[%d] Error   : %s", retry_num, exc)

                    exc_str = str(exc)

                    # Network errors cannot be fixed by the LLM — fail immediately.
                    if "Jira connection failed" in exc_str:
                        raise

                    # Invalid field values: strip offending conditions deterministically,
                    # no LLM call needed — the LLM has no knowledge of valid allowed values.
                    value_errors = _extract_value_errors(exc_str)
                    if value_errors:
                        for vfield, vval in value_errors:
                            logger.warning("  retry[%d] Invalid value %r for field %r — stripping condition", retry_num, vval, vfield)
                        current_jql = _strip_field_conditions(current_jql, [f for f, _ in value_errors])
                        logger.info("  retry[%d] JQL after value strip: %s", retry_num, current_jql)
                        continue

                    # Unsupported operator on field (e.g. comment IS NOT EMPTY, comment ~ ''):
                    # JQL limitation — the LLM cannot fix this; rewrite or strip deterministically.
                    unsupported_fields = _extract_unsupported_op_fields(exc_str)
                    if unsupported_fields:
                        for f in unsupported_fields:
                            fp = re.escape(f)
                            before = current_jql
                            # IS NOT EMPTY → text-search existence check (comment ~ '.')
                            current_jql = re.sub(
                                rf"\b{fp}\s+IS\s+NOT\s+EMPTY\b",
                                f"{f} ~ '.'",
                                current_jql,
                                flags=re.IGNORECASE,
                            )
                            # empty-string text search (comment ~ '') → same rewrite
                            current_jql = re.sub(
                                rf"\b{fp}\s*~\s*(?:''|\"\")",
                                f"{f} ~ '.'",
                                current_jql,
                                flags=re.IGNORECASE,
                            )
                            if current_jql != before:
                                logger.warning("  retry[%d] Field %r: rewrote EMPTY search to %r ~ '.'", retry_num, f, f)
                            else:
                                # IS EMPTY — no JQL workaround; strip the condition
                                current_jql = _strip_field_conditions(current_jql, [f])
                                logger.warning("  retry[%d] Field %r: IS EMPTY has no JQL workaround — stripped condition", retry_num, f)
                        logger.info("  retry[%d] JQL after unsupported-op rewrite: %s", retry_num, current_jql)
                        continue

                    # Field supports IN/NOT IN but not IS/IS NOT (e.g. issueLinkType IS NOT EMPTY):
                    # strip the IS [NOT] EMPTY condition inline — LLM cannot fix this.
                    unsupported_op_fields = _extract_unsupported_operator_fields(exc_str)
                    if unsupported_op_fields:
                        for f in unsupported_op_fields:
                            fp = re.escape(f)
                            current_jql = re.sub(
                                rf"\s+AND\s+{fp}\s+IS\s+(?:NOT\s+)?(?:EMPTY|NULL)",
                                "",
                                current_jql,
                                flags=re.IGNORECASE,
                            )
                            current_jql = re.sub(
                                rf"^{fp}\s+IS\s+(?:NOT\s+)?(?:EMPTY|NULL)\s*(?:AND\s+)?",
                                "",
                                current_jql.strip(),
                                flags=re.IGNORECASE,
                            )
                            logger.warning("  retry[%d] Field %r: operator not supported — stripped IS [NOT] EMPTY condition", retry_num, f)
                        logger.info("  retry[%d] JQL after unsupported-operator strip: %s", retry_num, current_jql)
                        continue

                    error_fields = _extract_error_fields(exc_str)
                    _accumulated_before = len(accumulated_prompt)

                    # For each unknown field, look up the closest valid Jira field
                    # via the already-seeded field embeddings and log the suggestion.
                    field_suggestions: dict[str, tuple[str, str]] = {}
                    for ef in error_fields:
                        suggestion = self.jira_field_embeddings.find_similar_field_name(
                            ef, self.document_processor._model
                        )
                        if suggestion:
                            field_id, canonical_name = suggestion
                            logger.info(
                                "  retry[%d] Closest field to '%s': field_id=%s, name='%s'",
                                retry_num, ef, field_id, canonical_name,
                            )
                            field_suggestions[ef] = (field_id, canonical_name)
                        else:
                            logger.info(
                                "  retry[%d] No close field found for '%s' in field embeddings",
                                retry_num, ef,
                            )

                    if len(error_fields) == 1:
                        bad_field = error_fields[0]
                        logger.warning("  retry[%d] Bad field: %s — using targeted retry prompt", retry_num, bad_field)
                        retry_prompt = accumulated_prompt + JQL_RETRY_FIELD_TEMPLATE.format(
                            field=bad_field,
                            bad_jql=current_jql,
                        )
                        if bad_field in field_suggestions:
                            fid, fname = field_suggestions[bad_field]
                            retry_prompt += (
                                f"\nClosest valid field to '{bad_field}': field_id='{fid}', display name='{fname}'."
                                f" Use it only if it matches the query intent."
                            )
                    elif error_fields:
                        fields_str = ", ".join(f"'{f}'" for f in error_fields)
                        logger.warning("  retry[%d] Bad fields: %s — using multi-field retry prompt", retry_num, fields_str)
                        retry_prompt = accumulated_prompt + JQL_RETRY_FIELDS_TEMPLATE.format(
                            fields=fields_str,
                            bad_jql=current_jql,
                        )
                        if field_suggestions:
                            hints = "; ".join(
                                f"'{ef}' → field_id='{fid}', display name='{fname}'"
                                for ef, (fid, fname) in field_suggestions.items()
                            )
                            retry_prompt += f"\nClosest valid fields: {hints}. Use only if they match the query intent."
                    else:
                        retry_prompt = accumulated_prompt + JQL_RETRY_TEMPLATE.format(
                            bad_jql=current_jql,
                            error=exc,
                        )

                    # Append sanitizer value hints when present — these give the LLM
                    # the closest valid candidates for any stripped invalid values,
                    # bounded to top-N so token cost stays small.
                    if pending_hints:
                        hint_text = "\n".join(h.to_prompt_text() for h in pending_hints)
                        retry_prompt += JQL_RETRY_VALUE_HINTS_TEMPLATE.format(hints=hint_text)
                        logger.info("  retry[%d] Injecting %d value hint(s) into retry prompt.", retry_num, len(pending_hints))

                    accumulated_prompt = retry_prompt

                    _ext_tokens = (len(retry_prompt) - _accumulated_before) // 4
                    token_usage.retry_tokens += _ext_tokens
                    logger.info(
                        "  retry[%d] Prompt sizes — base: %d tokens, extension: %d tokens, total: %d tokens, cumulative_retry: %d tokens",
                        retry_num,
                        token_usage.total_tokens,
                        _ext_tokens,
                        len(retry_prompt) // 4,
                        token_usage.retry_tokens,
                    )

                    retry_raw = await self.llm_client.generate_jql(retry_prompt)
                    logger.debug("retry[%d] LLM raw response: %s", retry_num, retry_raw)

                    try:
                        retry_data = json.loads(_extract_json_object(retry_raw), strict=False)
                    except json.JSONDecodeError:
                        try:
                            retry_data = json.loads(_extract_json_object(_repair_jql_quotes(retry_raw)), strict=False)
                            logger.warning("retry[%d] LLM response had unescaped quotes in jql — repaired.", retry_num)
                        except json.JSONDecodeError as parse_exc:
                            _err = ValueError(f"LLM retry[{retry_num}] response is not valid JSON: {retry_raw!r}")
                            _err.token_usage = token_usage  # type: ignore[attr-defined]
                            raise _err from parse_exc

                    llm_result = JqlResponse(**retry_data)
                    sanitized = self._sanitize_jql(llm_result.jql or "")
                    current_jql = sanitized.jql
                    pending_hints = sanitized.hints
                    logger.info("*** AI JQL retry[%d]: %s", retry_num, current_jql)

            jira_result["resolved_intent_fields"] = resolved
            jira_result["token_usage"] = token_usage

        return llm_result, jira_result

    def _sanitize_jql(self, jql: str) -> SanitizeResult:
        """Delegate all JQL cleanup passes to JqlSanitizer.

        Returns a SanitizeResult with the cleaned JQL string, any auto-corrections
        made (logged), and any ValueHints to inject into the next retry prompt.
        Falls back to a no-op result when called before run() completes.
        """
        if self.jql_sanitizer is None:
            return SanitizeResult(jql=jql)
        return self.jql_sanitizer.sanitize(jql)
