"""
Paginated Jira search client for /rest/api/2/search.

Jira's REST API hard-caps a single response at 1000 issues. JiraSearchClient
loops over pages using startAt until max_results issues are fetched or the
result set is exhausted.
"""

import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

_JIRA_PAGE_CAP = 1000


class JiraSearchRequest(BaseModel):
    jql: str
    fields: str
    max_results: int = Field(default=10, ge=1)
    base_url: str
    auth: tuple[str, str] | None = None
    auth_headers: dict[str, str] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _strip_trailing_slash(self) -> "JiraSearchRequest":
        self.base_url = self.base_url.rstrip("/")
        return self


class JiraPage(BaseModel):
    issues: list[dict[str, Any]]
    total: int
    start_at: int
    max_results: int


class JiraSearchResult(BaseModel):
    jql: str
    issues: list[dict[str, Any]]
    total: int
    fetched: int


class JiraSearchClient:
    """Fetches Jira issues with automatic pagination across the 1000-issue-per-page cap."""

    async def search(self, request: JiraSearchRequest) -> JiraSearchResult:
        url = f"{request.base_url}/rest/api/2/search"
        issues: list[dict[str, Any]] = []
        total = 0
        start_at = 0

        while len(issues) < request.max_results:
            page_size = min(_JIRA_PAGE_CAP, request.max_results - len(issues))
            page = await self._fetch_page(
                url=url,
                jql=request.jql,
                fields=request.fields,
                start_at=start_at,
                page_size=page_size,
                auth=request.auth,
                auth_headers=request.auth_headers,
            )
            total = page.total
            issues.extend(page.issues)
            logger.info(
                "Jira page: startAt=%d pageSize=%d got=%d accumulated=%d total=%d",
                start_at, page_size, len(page.issues), len(issues), total,
            )

            if not page.issues or len(issues) >= total:
                break
            start_at += len(page.issues)

        logger.info("Jira search done: fetched=%d total=%d", len(issues), total)
        return JiraSearchResult(
            jql=request.jql,
            issues=issues,
            total=total,
            fetched=len(issues),
        )

    async def _fetch_page(
        self,
        url: str,
        jql: str,
        fields: str,
        start_at: int,
        page_size: int,
        auth: tuple[str, str] | None,
        auth_headers: dict[str, str],
    ) -> JiraPage:
        params = {
            "jql":        jql,
            "startAt":    start_at,
            "maxResults": page_size,
            "fields":     fields,
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    url,
                    params=params,
                    auth=auth if auth and any(auth) else None,
                    headers={"Accept": "application/json", **auth_headers},
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
        return JiraPage(
            issues=payload.get("issues", []),
            total=payload.get("total", 0),
            start_at=payload.get("startAt", start_at),
            max_results=payload.get("maxResults", page_size),
        )
