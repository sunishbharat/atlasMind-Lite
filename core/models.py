"""
models.py — Pydantic request/response models for the aMind API.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, field_validator


class RouteResult(BaseModel):
    """Result of query routing — determines which pipeline handles the request."""
    type: Literal["jql", "general", "raw"]
    answer: str = ""
    raw_jql: str = ""     # type="raw": literal JQL text left of /raw
    chart_hint: str = ""  # type="raw": chart instruction right of /raw

    @property
    def is_jql(self) -> bool:
        return self.type == "jql"

    @property
    def is_raw(self) -> bool:
        return self.type == "raw"


class JqlResponse(BaseModel):
    """Structured output from the LLM for a JQL query."""
    jql:           str | None            = None
    chart_spec:    dict[str, Any] | None = None
    answer:        str | None            = None
    intent_fields: list[str] | None      = None


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
    limit:      Optional[int] = None
    request_id: Optional[str] = None  # client-generated UUID; used by POST /event to cancel


class QueryResponse(BaseModel):
    type:           str
    profile:        str
    jira_base_url:  str
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


class ApiResponse(BaseModel):
    output: Optional[QueryResponse] = None
    error:  Optional[str]           = None
