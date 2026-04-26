"""
Jira REST API helpers for retrieving field metadata.

Provides:
- fetch_field_allowed_values(): retrieve allowed values for a single field.
- fetch_and_save_allowed_values(): one-time utility that reads jira_fields.json,
  fetches allowed values for all eligible fields, and writes the result to
  jira_fields_allowed_values.json for use during embedding.

Run the one-time fetch:
    uv run python -c "
    import asyncio
    from jira.jira_field_api import fetch_and_save_allowed_values
    asyncio.run(fetch_and_save_allowed_values())
    "
"""

import asyncio
import json
import logging
from pathlib import Path
import httpx
import requests
from config.jira_config import _SYSTEM_FIELD_ENDPOINTS, load_active_profile, get_data_dir, build_jira_auth
from settings import JIRA_FIELDS_FILENAME, JIRA_ALLOWED_VALUES_FILENAME

logger = logging.getLogger(__name__)


def fetch_and_save_fields(output_path: Path = None) -> Path:
    """Fetch all Jira fields from the active profile and save to a JSON file.

    Calls /rest/api/2/field on the Jira instance defined in config/profiles.json
    (active profile). The response list is converted to a dict keyed by field ID
    to match the format expected by _parse_jira_fields_json().

    If output_path is not provided it defaults to the domain-scoped directory:
        data/{domain_slug}/jira_fields.json

    Args:
        output_path: Destination path for the fields JSON file. Derived from
            the active profile's jira_url when omitted.

    Returns:
        Path: The path the file was written to.

    Raises:
        requests.HTTPError: If the Jira API returns a non-2xx response.
    """
    profile = load_active_profile()
    base_url = profile["jira_url"].rstrip("/")
    auth, auth_headers = build_jira_auth(profile)

    if output_path is None:
        output_path = get_data_dir(profile["jira_url"]) / JIRA_FIELDS_FILENAME

    url = f"{base_url}/rest/api/2/field"
    logger.info("Fetching Jira fields from %s", url)

    response = requests.get(url, auth=auth, headers={"Accept": "application/json", **auth_headers}, timeout=30)
    response.raise_for_status()

    fields_list: list[dict] = response.json()
    fields_dict = {f["id"]: f for f in fields_list}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fields_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved %d Jira fields to %s", len(fields_dict), output_path)
    return output_path


async def fetch_field_allowed_values(
    base_url: str,
    field_id: str,
    field_type: str,
    auth: tuple[str, str],
    extra_headers: dict | None = None,
) -> list[str]:
    """Retrieve the allowed values for a Jira field via the REST API.

    Routes to the appropriate endpoint based on field_type:
    - System fields (status, priority, resolution, issuetype) use dedicated endpoints.
    - Custom option/array fields use /rest/api/2/field/{fieldId}/option with pagination.
    - All other field types (string, number, date, user, version, etc.) return empty list.

    Args:
        base_url: Jira instance base URL, e.g. "https://your-org.atlassian.net".
        field_id: The field ID, e.g. "status" or "customfield_10023".
        field_type: The schema type from the fields JSON, e.g. "option", "status".
        auth: (username_or_email, api_token_or_password) for Basic auth. Pass None
            when using Bearer auth via extra_headers instead.
        extra_headers: Additional headers to merge (e.g. {"Authorization": "Bearer ..."}).

    Returns:
        list[str]: Sorted list of allowed value names. Empty list when the field
        type does not have discrete options.
    """
    endpoint = _resolve_endpoint(field_id, field_type)
    if endpoint is None:
        logger.debug("field_id=%s type=%s has no discrete allowed values", field_id, field_type)
        return []

    url = f"{base_url.rstrip('/')}{endpoint}"
    logger.debug("Fetching allowed values for field_id=%s from %s", field_id, url)

    headers = {"Accept": "application/json", **(extra_headers or {})}
    is_paginated = endpoint.endswith("/option")
    all_values: list[str] = []
    start_at = 0

    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            params = {"startAt": start_at, "maxResults": 100} if is_paginated else {}
            response = await client.get(
                url,
                params=params,
                auth=auth if auth and any(auth) else None,
                headers=headers,
            )

            if response.status_code == 401:
                logger.warning("field_id=%s  skipped — endpoint requires authentication (401)", field_id)
                return []
            if response.status_code == 403:
                logger.warning("field_id=%s  skipped — access denied (403)", field_id)
                return []
            if response.status_code == 404:
                logger.warning("field_id=%s  skipped — endpoint not found (404)", field_id)
                return []

            response.raise_for_status()
            data = response.json()
            all_values.extend(_extract_names(data))

            if is_paginated and isinstance(data, dict) and not data.get("isLast", True):
                start_at += data.get("maxResults", 100)
            else:
                break

    result = sorted(set(all_values))
    logger.debug("field_id=%s  allowed_values count=%d", field_id, len(result))
    return result


def _resolve_endpoint(field_id: str, field_type: str) -> str | None:
    """Return the REST path for the given field, or None if not applicable."""
    if field_type in _SYSTEM_FIELD_ENDPOINTS:
        return _SYSTEM_FIELD_ENDPOINTS[field_type]

    # Only option and array (multi-select) custom fields support the /option endpoint.
    # version-type custom fields are handled separately via project versions.
    if field_id.startswith("customfield_") and field_type in ("option", "array"):
        return f"/rest/api/2/field/{field_id}/option"

    return None


async def _fetch_all_version_names(
    base_url: str,
    auth: tuple[str, str] | None,
    auth_headers: dict,
) -> list[str]:
    """Aggregate all version names across every project in the Jira instance.

    Used to populate allowed values for version-type custom fields, which have no
    field-level /option endpoint. The returned list is the union of all project versions.
    """
    headers = {"Accept": "application/json", **auth_headers}
    base = base_url.rstrip("/")
    all_versions: set[str] = set()

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(
            f"{base}/rest/api/2/project",
            auth=auth if auth and any(auth) else None,
            headers=headers,
        )
        if resp.status_code != 200:
            logger.warning("Could not fetch projects for version aggregation (HTTP %s)", resp.status_code)
            return []

        projects = resp.json()
        logger.info("Fetching versions from %d projects for version-type custom fields", len(projects))

        async def _project_versions(key: str) -> list[str]:
            try:
                r = await client.get(
                    f"{base}/rest/api/2/project/{key}/versions",
                    auth=auth if auth and any(auth) else None,
                    headers=headers,
                )
                if r.status_code == 200:
                    return [v["name"] for v in r.json() if "name" in v]
            except Exception:
                pass
            return []

        results = await asyncio.gather(*[_project_versions(p["key"]) for p in projects])
        for names in results:
            all_versions.update(names)

    return sorted(all_versions)


async def fetch_and_save_allowed_values(
    fields_json: Path = None,
    output_json: Path = None,
    base_url: str = None,
    auth: tuple[str, str] = None,
) -> None:
    """Fetch allowed values for all eligible fields and write them to a JSON file.

    Reads jira_fields.json to get field IDs and types, calls the Jira REST API
    for each eligible field, and saves the results so _parse_jira_fields_json()
    can load them without making live API calls during embedding.

    Handles three categories:
    - System fields (status, priority, resolution, issuetype): dedicated endpoints.
    - option/array custom fields: /field/{id}/option with pagination.
    - version custom fields: aggregated from all project versions.

    Fields that return an empty list are omitted from the output.

    Args:
        fields_json: Path to the Jira fields JSON from /rest/api/2/field.
        output_json: Destination path for the allowed values JSON.
        base_url: Jira instance base URL.
        auth: (username, api_token). Leave empty to derive from the active profile.
    """
    profile = load_active_profile()
    data_dir = get_data_dir(profile["jira_url"])

    if fields_json is None:
        fields_json = data_dir / JIRA_FIELDS_FILENAME
    if output_json is None:
        output_json = data_dir / JIRA_ALLOWED_VALUES_FILENAME
    if base_url is None:
        base_url = profile["jira_url"]
    if auth is None:
        auth, auth_headers = build_jira_auth(profile)
    else:
        auth_headers = {}

    raw: dict = json.loads(fields_json.read_text(encoding="utf-8"))
    has_auth = bool(auth) or bool(auth_headers)

    eligible: list[tuple[str, str]] = []
    version_field_ids: list[str] = []

    for field_id, field in raw.items():
        schema = field.get("schema") or {}
        field_type = schema.get("type", "unknown")
        items_type = schema.get("items", "")
        is_custom = field.get("custom", False)

        # version-type custom fields have no /option endpoint; collect separately
        if is_custom and (field_type == "version" or (field_type == "array" and items_type == "version")):
            version_field_ids.append(field_id)
            continue

        if _resolve_endpoint(field_id, field_type) is None:
            continue
        eligible.append((field_id, field_type))

    logger.info(
        "Fields to fetch: %d system/option, %d version-type custom, %d custom option/array (auth=%s)",
        sum(1 for fid, _ in eligible if not fid.startswith("customfield_")),
        len(version_field_ids),
        sum(1 for fid, _ in eligible if fid.startswith("customfield_")),
        has_auth,
    )

    async def _fetch(field_id: str, field_type: str) -> tuple[str, list[str]]:
        values = await fetch_field_allowed_values(
            base_url=base_url,
            field_id=field_id,
            field_type=field_type,
            auth=auth,
            extra_headers=auth_headers,
        )
        return field_id, values

    results = await asyncio.gather(*(_fetch(fid, ftype) for fid, ftype in eligible))

    allowed: dict[str, list[str]] = {
        field_id: values for field_id, values in results if values
    }

    if version_field_ids:
        version_names = await _fetch_all_version_names(base_url, auth, auth_headers)
        if version_names:
            logger.info(
                "Storing %d version names for %d version-type custom fields",
                len(version_names), len(version_field_ids),
            )
            for fid in version_field_ids:
                allowed[fid] = version_names
        else:
            logger.warning("No version names fetched — version-type custom fields will have no allowed values")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(allowed, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved allowed values for %d fields to %s", len(allowed), output_json)


def _extract_names(data: list | dict) -> list[str]:
    """Extract the 'name' or 'value' from a Jira API response payload.

    System field endpoints return a list of objects with a 'name' key.
    Custom field option endpoints return a paginated dict with a 'values' list
    where each entry has a 'value' key.
    """
    if isinstance(data, list):
        return sorted(item["name"] for item in data if "name" in item)

    if isinstance(data, dict) and "values" in data:
        return sorted(item["value"] for item in data["values"] if "value" in item)

    return []
