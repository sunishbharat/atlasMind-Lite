"""
models.py — Pydantic request/response models for the aMind API.
"""

import re
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator

_JIRA_KEY_RE = re.compile(r"^[A-Z][A-Z0-9]+-\d+$")


class RouteResult(BaseModel):
    """Result of query routing — determines which pipeline handles the request."""
    type: Literal["jql", "general", "raw"]
    answer: str = ""
    raw_jql: str = ""              # type="raw": literal JQL text left of /raw
    chart_hint: str = ""           # type="raw": chart instruction right of /raw
    jira_type_override: str | None = None  # /cloud or /server flag: overrides profile jira_type

    @property
    def is_jql(self) -> bool:
        return self.type == "jql"

    @property
    def is_raw(self) -> bool:
        return self.type == "raw"


class LlmClause(BaseModel):
    """One WHERE condition as reported by the LLM in its clauses[] output."""
    field:    str
    operator: str
    value:    str | list[str] | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _coerce_value(cls, v: Any) -> str | list[str] | None:
        # LLM sometimes returns a bare int/float for numeric comparisons instead of null.
        # Treat them as null — numeric values are not correctable by the semantic validator.
        if isinstance(v, (int, float)):
            return None
        if isinstance(v, list):
            return [str(item) for item in v]
        return v


class JqlResponse(BaseModel):
    """Structured output from the LLM for a JQL query."""
    jql:           str | None            = None
    clauses:       list[LlmClause]       = []
    chart_spec:    dict[str, Any] | None = None
    answer:        str | None            = None
    intent_fields: list[str] | None      = None
    where_fields:  list[str] | None      = None  # display names of fields used in WHERE clause
    limit:         int | None            = None  # user-specified result count extracted from query

    @field_validator("limit", mode="before")
    @classmethod
    def _coerce_limit(cls, v: Any) -> int | None:
        # LLM may return 0 or a negative integer — treat anything < 1 as null.
        if isinstance(v, int) and v >= 1:
            return v
        return None


class ChartSpec(BaseModel):
    type: Literal["bar", "stacked_bar", "pie", "line", "scatter"]
    x_field: str
    y_field: str
    title: str = ""
    color_field: Optional[str] = None

    @field_validator("type", mode="before")
    @classmethod
    def _normalise_type(cls, v: str) -> str:
        """Normalise LLM chart type aliases before validation."""
        _ALIASES = {"multi-line": "line", "multiline": "line", "area": "line"}
        return _ALIASES.get(str(v).lower(), v)


class TokenUsage(BaseModel):
    """Estimated prompt token counts for a single JQL query."""
    system_tokens:   int = 0
    fields_tokens:   int = 0
    examples_tokens: int = 0
    total_tokens:    int = 0
    retry_tokens:    int = 0  # accumulated tokens added across all retry extensions


class ServerMeta(BaseModel):
    """Metadata about the server configuration sent to the frontend with every response.

    Add new fields here to expose more server-side context to the UI.
    All fields are optional so older clients are not broken when new fields are added.
    """
    model_name:  Optional[str] = None
    llm_backend: Optional[str] = None
    llm_timeout: Optional[int] = None


class QueryRequest(BaseModel):
    query:      str
    profile:    Optional[str] = None
    limit:      Optional[int] = Field(default=None, ge=1)
    request_id: Optional[str] = None  # client-generated UUID; used by POST /event to cancel


class QueryResponse(BaseModel):
    type:           str
    profile:        str
    jira_base_url:  str
    jira_type:      Optional[str]              = None
    answer:         Optional[str]              = None
    jql:            Optional[str]              = None
    total:          int                        = 0
    shown:          int                        = 0
    examined:       int                        = 0
    post_filters:   list                       = []
    display_fields: list[str]                  = []
    issues:         list[dict]                 = []
    chart_spec:     Optional[ChartSpec]        = None
    filters:        Optional[dict[str, list[str]]] = None
    meta:           Optional[ServerMeta]       = None
    token_usage:    Optional[TokenUsage]       = None


class ApiResponse(BaseModel):
    output: Optional[QueryResponse] = None
    error:  Optional[str]           = None


# -- POST /issue_details models ----------------------------------------

class Comment(BaseModel):
    id:      str
    author:  str
    body:    str
    created: str
    updated: str


class IssueLink(BaseModel):
    type:                str
    direction:           Literal["outward", "inward"]
    linked_issue_key:    str
    linked_issue_summary: Optional[str] = None


class ChangelogEntry(BaseModel):
    field:      str
    from_value: Optional[str] = None
    to_value:   str
    author:     str
    timestamp:  str


class IssueDetail(BaseModel):
    key:          str
    priority:     Optional[str] = None
    assignee:     Optional[str] = None
    due_date:     Optional[str] = None
    fix_versions: list[str]     = []
    flagged:      bool          = False
    comments:     list[Comment]        = []
    links:        list[IssueLink]      = []
    changelog:    list[ChangelogEntry] = []


class IssueDetailsRequest(BaseModel):
    issue_keys:     list[str]     = Field(..., min_length=1)
    request_id:     Optional[str] = None
    comments_limit: Optional[int] = Field(default=None, ge=1)

    @field_validator("issue_keys")
    @classmethod
    def _validate_keys(cls, v: list[str]) -> list[str]:
        for key in v:
            if not _JIRA_KEY_RE.match(key):
                raise ValueError(f"Invalid Jira issue key: {key!r}")
        return v


class IssueDetailsResponse(BaseModel):
    issues:    list[IssueDetail] = []
    not_found: list[str]         = []
    error:     Optional[str]     = None
