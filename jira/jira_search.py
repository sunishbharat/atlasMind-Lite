"""
Paginated Jira search client.

Server: GET /rest/api/2/search — offset pagination via startAt.
Cloud:  POST /rest/api/3/search/jql — cursor pagination via nextPageToken.

The active path is resolved from the profile's optional search_path override,
falling back to the jira_type default from config/jira_config.py.
"""

import logging
from typing import Any
from urllib.parse import unquote

import httpx
from pydantic import BaseModel, Field, model_validator

from cloud.tls import tls
from config.jira_config import get_search_path

logger = logging.getLogger(__name__)

_JIRA_PAGE_CAP = 1000


class JiraSearchRequest(BaseModel):
    jql: str
    fields: str
    max_results: int = Field(default=10, ge=1)
    base_url: str
    jira_type: str = "server"
    search_path: str | None = None
    auth: tuple[str, str] | None = None
    auth_headers: dict[str, str] = Field(default_factory=dict)
    # Cloud-only: CMDB/Assets field IDs to expand inline (label + objectKey).
    # Ignored by the server path.
    cmdb_field_ids: set[str] = Field(default_factory=set)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _strip_trailing_slash(self) -> "JiraSearchRequest":
        self.base_url = self.base_url.rstrip("/")
        return self

    def resolved_search_path(self) -> str:
        return self.search_path or get_search_path(self.jira_type)


class JiraPage(BaseModel):
    issues: list[dict[str, Any]]
    total: int
    start_at: int
    max_results: int
    next_page_token: str | None = None


class JiraSearchResult(BaseModel):
    jql: str
    issues: list[dict[str, Any]]
    total: int
    fetched: int


class JiraSearchClient:
    """Fetches Jira issues with automatic pagination across the 1000-issue-per-page cap."""

    async def validate_jql(
        self,
        jql: str,
        base_url: str,
        auth: tuple[str, str] | None,
        auth_headers: dict[str, str],
        jira_type: str = "server",
        search_path: str | None = None,
    ) -> str | None:
        """Validate JQL without fetching issues.

        Returns the Jira error message string if invalid, None if valid.
        """
        path = search_path or get_search_path(jira_type)
        url = f"{base_url.rstrip('/')}{path}"
        base_headers = {"Accept": "application/json", **auth_headers}
        resolved_auth = auth if auth and any(auth) else None

        try:
            async with tls.httpx_client(timeout=15) as client:
                if jira_type == "cloud":
                    response = await client.post(
                        url,
                        json={"jql": jql, "maxResults": 1},
                        auth=resolved_auth,
                        headers={**base_headers, "Content-Type": "application/json"},
                    )
                else:
                    response = await client.get(
                        url,
                        params={"jql": jql, "maxResults": 0},
                        auth=resolved_auth,
                        headers=base_headers,
                    )
                response.raise_for_status()
            payload = response.json()
            if payload.get("warningMessages") or (
                jira_type == "cloud"
                and payload.get("total", -1) == 0
                and not (auth_headers or resolved_auth)
            ):
                logger.warning(
                    "JQL validation returned 0 results with no credentials — possible silent auth failure"
                )
            return None
        except httpx.HTTPStatusError as exc:
            if exc.response.is_redirect:
                location = unquote(exc.response.headers.get("location", ""))
                return f"Jira redirect ({exc.response.status_code}) to {location} — wrong endpoint or authentication required"
            try:
                body = exc.response.json()
                messages = body.get("errorMessages", [])
                errors = body.get("errors", {})
                return "; ".join(messages + list(errors.values())) or str(exc)
            except Exception:
                return str(exc)
        except httpx.HTTPError as exc:
            return str(exc)

    async def search(self, request: JiraSearchRequest) -> JiraSearchResult:
        if request.jira_type == "cloud":
            return await self._search_cloud(request)
        return await self._search_server(request)

    async def _search_cloud(self, request: JiraSearchRequest) -> JiraSearchResult:
        url = f"{request.base_url}{request.resolved_search_path()}"
        logger.info("Jira Cloud REST API URL: POST %s", url)
        fields_list = [f.strip() for f in request.fields.split(",") if f.strip()]
        issues: list[dict[str, Any]] = []
        total = 0
        next_page_token: str | None = None

        while len(issues) < request.max_results:
            page_size = min(_JIRA_PAGE_CAP, request.max_results - len(issues))
            body: dict[str, Any] = {"jql": request.jql, "fields": fields_list, "maxResults": page_size}
            if next_page_token:
                body["nextPageToken"] = next_page_token
            if request.cmdb_field_ids:
                # Atlassian Assets inline expand: "<customfield_id>.cmdb.label" is a
                # Jira Cloud search API extension that returns the Assets object display
                # label inside each issue field entry. Not in the official REST spec but
                # confirmed via Atlassian support and community references. If a tenant
                # does not support it, the field is returned without the label key and
                # the CMDB diagnostic in atlasmind.py will log a warning.
                body["expand"] = ",".join(f"{fid}.cmdb.label" for fid in sorted(request.cmdb_field_ids))

            page = await self._fetch_page_cloud(url, body, request.auth, request.auth_headers)
            total = page.total
            issues.extend(page.issues)
            logger.info(
                "Jira Cloud page: pageSize=%d got=%d accumulated=%d total=%d nextToken=%s",
                page_size, len(page.issues), len(issues), total, bool(page.next_page_token),
            )
            if total == 0 and page.issues:
                # Jira Cloud v3 cursor pagination does not always populate `total`; rely on nextPageToken
                logger.debug("Jira Cloud v3 returned total=0 with %d issues ", len(page.issues))

            if not page.issues or not page.next_page_token or len(issues) >= request.max_results:
                break
            if total > 0 and len(issues) >= total:
                break
            next_page_token = page.next_page_token

        reported_total = total if total > 0 else len(issues)
        if reported_total == 0:
            logger.warning(
                "Jira Cloud returned 0 results for JQL %r - if results are expected, verify credentials "
                "(Jira Cloud returns HTTP 200 with total=0 on auth failure rather than 401)",
                request.jql,
            )
        logger.info("Jira Cloud search done: fetched=%d total=%d", len(issues), reported_total)
        return JiraSearchResult(jql=request.jql, issues=issues, total=reported_total, fetched=len(issues))

    async def _search_server(self, request: JiraSearchRequest) -> JiraSearchResult:
        url = f"{request.base_url}{request.resolved_search_path()}"
        issues: list[dict[str, Any]] = []
        total = 0
        start_at = 0

        while len(issues) < request.max_results:
            page_size = min(_JIRA_PAGE_CAP, request.max_results - len(issues))
            page = await self._fetch_page_server(
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
                "Jira Server page: startAt=%d pageSize=%d got=%d accumulated=%d total=%d",
                start_at, page_size, len(page.issues), len(issues), total,
            )

            if not page.issues or len(issues) >= total:
                break
            start_at += len(page.issues)

        logger.info("Jira Server search done: fetched=%d total=%d", len(issues), total)
        return JiraSearchResult(jql=request.jql, issues=issues, total=total, fetched=len(issues))

    async def _fetch_page_cloud(
        self,
        url: str,
        body: dict[str, Any],
        auth: tuple[str, str] | None,
        auth_headers: dict[str, str],
    ) -> JiraPage:
        logger.debug("Jira Cloud request body: %s", body)
        try:
            async with tls.httpx_client(timeout=30) as client:
                response = await client.post(
                    url,
                    json=body,
                    auth=auth if auth and any(auth) else None,
                    headers={"Accept": "application/json", "Content-Type": "application/json", **auth_headers},
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("Jira Cloud %d error - request body: %s | response: %s",
                         exc.response.status_code, body, exc.response.text)
            raise ValueError(f"Jira rejected the JQL: {_parse_jira_error(exc)}") from exc
        except httpx.HTTPError as exc:
            logger.warning("Jira REST API call failed: %s", exc)
            raise ValueError(f"Jira connection failed: {exc}") from exc

        payload = response.json()
        return JiraPage(
            issues=payload.get("issues", []),
            total=payload.get("total", 0),
            start_at=0,
            max_results=payload.get("maxResults", body.get("maxResults", 0)),
            next_page_token=payload.get("nextPageToken"),
        )

    async def _fetch_page_server(
        self,
        url: str,
        jql: str,
        fields: str,
        start_at: int,
        page_size: int,
        auth: tuple[str, str] | None,
        auth_headers: dict[str, str],
    ) -> JiraPage:
        params = {"jql": jql, "startAt": start_at, "maxResults": page_size, "fields": fields}
        try:
            async with tls.httpx_client(timeout=30) as client:
                response = await client.get(
                    url,
                    params=params,
                    auth=auth if auth and any(auth) else None,
                    headers={"Accept": "application/json", **auth_headers},
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ValueError(f"Jira rejected the JQL: {_parse_jira_error(exc)}") from exc
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


def _parse_jira_error(exc: httpx.HTTPStatusError) -> str:
    if exc.response.is_redirect:
        location = exc.response.headers.get("location", "")
        return f"Jira redirect ({exc.response.status_code}) to {location} — wrong endpoint or authentication required"
    try:
        body = exc.response.json()
        messages = body.get("errorMessages", [])
        errors = body.get("errors", {})
        msg = "; ".join(messages + list(errors.values()))
        if msg:
            return msg
    except Exception:
        pass
    logger.warning("Jira API error (HTTP %s): %s", exc.response.status_code, exc)
    return str(exc)
