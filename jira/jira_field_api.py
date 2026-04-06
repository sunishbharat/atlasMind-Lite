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
    from jira_field_api import fetch_and_save_allowed_values
    asyncio.run(fetch_and_save_allowed_values())
    "
"""

import asyncio
import json
import logging
from pathlib import Path
import httpx
import requests
from config.jira_config import _SYSTEM_FIELD_ENDPOINTS, load_active_profile, get_data_dir
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
    email = profile.get("email", "")
    token = profile.get("token", "")
    auth = (email, token) if email and token else None

    if output_path is None:
        output_path = get_data_dir(profile["jira_url"]) / JIRA_FIELDS_FILENAME

    url = f"{base_url}/rest/api/2/field"
    logger.info("Fetching Jira fields from %s", url)

    response = requests.get(url, auth=auth, headers={"Accept": "application/json"}, timeout=30)
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
) -> list[str]:
    """Retrieve the allowed values for a Jira field via the REST API.

    Routes to the appropriate endpoint based on field_type:
    - System fields (status, priority, resolution, issuetype) use dedicated endpoints.
    - Custom option fields use /rest/api/2/field/{fieldId}/option.
    - All other field types (string, number, date, user, etc.) have no discrete
      allowed values and return an empty list.

    Args:
        base_url: Jira instance base URL, e.g. "https://your-org.atlassian.net".
        field_id: The field ID, e.g. "status" or "customfield_10023".
        field_type: The schema type from the fields JSON, e.g. "option", "status".
        auth: (username_or_email, api_token_or_password) for Basic auth.

    Returns:
        list[str]: Sorted list of allowed value names. Empty list when the field
        type does not have discrete options.

    Raises:
        httpx.HTTPStatusError: If the Jira API returns a non-2xx response.
    """
    endpoint = _resolve_endpoint(field_id, field_type)
    if endpoint is None:
        logger.debug("field_id=%s type=%s has no discrete allowed values", field_id, field_type)
        return []

    url = f"{base_url.rstrip('/')}{endpoint}"
    logger.info("Fetching allowed values for field_id=%s from %s", field_id, url)

    headers = {"Accept": "application/json"}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, auth=auth if any(auth) else None, headers=headers)

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
    values = _extract_names(response.json())
    logger.info("field_id=%s  allowed_values=%s", field_id, values)
    return values


def _resolve_endpoint(field_id: str, field_type: str) -> str | None:
    """Return the REST path for the given field, or None if not applicable."""
    if field_type in _SYSTEM_FIELD_ENDPOINTS:
        return _SYSTEM_FIELD_ENDPOINTS[field_type]

    if field_type == "option" or field_id.startswith("customfield_"):
        return f"/rest/api/2/field/{field_id}/option"

    return None


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

    Skips fields whose type has no discrete options (string, number, date, user).
    Fields that return an empty list are omitted from the output.

    Args:
        fields_json: Path to the Jira fields JSON from /rest/api/2/field.
        output_json: Destination path for the allowed values JSON.
        base_url: Jira instance base URL.
        auth: (username, api_token). Leave empty for public Jira instances.
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
        email = profile.get("email", "")
        token = profile.get("token", "")
        auth = (email, token)

    raw: dict = json.loads(fields_json.read_text(encoding="utf-8"))
    has_auth = any(auth)

    # Custom field option endpoints (/field/{id}/option) require authentication
    # on most Jira instances. Skip them when no credentials are provided to avoid
    # flooding the log with 401 warnings and wasting time on requests that will fail.
    eligible: list[tuple[str, str]] = []
    for field_id, field in raw.items():
        schema = field.get("schema") or {}
        field_type = schema.get("type", "unknown")

        is_custom = field.get("custom", False)

        if is_custom and not has_auth:
            continue
        if _resolve_endpoint(field_id, field_type) is None:
            continue
        eligible.append((field_id, field_type))

    logger.info("Fetching allowed values for %d eligible fields (auth=%s)", len(eligible), has_auth)

    async def _fetch(field_id: str, field_type: str) -> tuple[str, list[str]]:
        values = await fetch_field_allowed_values(
            base_url=base_url,
            field_id=field_id,
            field_type=field_type,
            auth=auth,
        )
        return field_id, values

    results = await asyncio.gather(*(_fetch(fid, ftype) for fid, ftype in eligible))

    allowed: dict[str, list[str]] = {
        field_id: values for field_id, values in results if values
    }

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
