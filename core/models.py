"""
models.py — Pydantic request/response models for the aMind API.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, field_validator


class JqlResponse(BaseModel):
    """Structured output from the LLM for a JQL query."""
    jql:           str | None            = None
    chart_spec:    dict[str, Any] | None = None
    answer:        str | None            = None
    intent_fields: list[str] | None      = None


class ChartSpec(BaseModel):
    type: Literal["bar", "pie", "line", "scatter"]
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


class QueryRequest(BaseModel):
    query:   str
    profile: Optional[str] = None
    limit:   Optional[int] = None


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


class ApiResponse(BaseModel):
    output: Optional[QueryResponse] = None
    error:  Optional[str]           = None
