"""
Jira Assets REST API helpers.

Auto-detects Assets-type custom fields from jira_fields.json and fetches their
object labels via the Jira Assets AQL API. Labels are stored in jira_assets.json,
which seeds the jira_asset_values pgvector table for cosine-similarity value
lookup during JQL generation.

Assets fields are identified by schema.custom starting with:
    com.atlassian.jira.plugins.cmdb:

On-demand refresh (when asset objects change in Jira):
    uv run python -c "
    import asyncio
    from jira.jira_assets_api import refresh_asset_values
    asyncio.run(refresh_asset_values())
    "
"""

import json
import logging
from pathlib import Path

import httpx

from cloud.tls import tls
from config.jira_config import build_jira_auth, get_data_dir, load_active_profile
from settings import JIRA_ASSETS_CONFIG_FILE, JIRA_ASSETS_FILENAME, JIRA_FIELDS_FILENAME

logger = logging.getLogger(__name__)

_DEFAULT_ASSET_KEYWORDS = [".insight", ".cmdb"]


def _load_asset_keywords_from_config() -> list[str]:
    """Read asset_field_keywords from config/jira_assets_fields.json at runtime."""
    try:
        config_file = Path(JIRA_ASSETS_CONFIG_FILE)
        if config_file.exists():
            config_data = json.loads(config_file.read_text(encoding="utf-8"))
            keywords = config_data.get("asset_field_keywords")
            if isinstance(keywords, list) and keywords:
                logger.info(
                    "detect_asset_fields: using %d asset keyword(s) from config: %s",
                    len(keywords), keywords,
                )
                return keywords
    except Exception as exc:
        logger.warning(
            "Failed to load asset keywords from %s - using defaults %s. Error: %s",
            JIRA_ASSETS_CONFIG_FILE, _DEFAULT_ASSET_KEYWORDS, exc,
        )
    return _DEFAULT_ASSET_KEYWORDS


def detect_asset_fields(fields_json: Path, asset_keywords: list[str] | None = None) -> dict[str, dict]:
    """Read jira_fields.json and return all Assets-type fields.

    Asset keywords (e.g. ".insight", ".cmdb") are used to detect Assets fields
    from schema.custom. When asset_keywords is None, reads from
    config/jira_assets_fields.json at runtime — no rebuild needed to change keywords.
    Fields whose schema.custom contains any configured keyword are treated as Assets
    fields. Uses the field display name as the AQL object type name. Override via
    config/jira_assets_fields.json when it differs.

    Args:
        fields_json: Path to data/<hostname>/jira_fields.json.
        asset_keywords: Optional list of keywords to detect Assets fields. If None,
            reads from config at runtime. Defaults to [".insight", ".cmdb"].

    Returns:
        {field_id: {"display_name": str, "object_type": str}}, or {} if the file
        does not exist or contains no Assets fields.
    """
    if not fields_json.exists():
        logger.info("detect_asset_fields: %s not found — skipping.", fields_json)
        return {}

    if asset_keywords is None:
        asset_keywords = _load_asset_keywords_from_config()

    raw: dict = json.loads(fields_json.read_text(encoding="utf-8"))
    result: dict[str, dict] = {}
    for field_id, field in raw.items():
        schema_custom: str = (field.get("schema") or {}).get("custom", "")
        if any(kw in schema_custom for kw in asset_keywords):
            name = field.get("name", field_id)
            result[field_id] = {"display_name": name, "object_type": name}
            logger.info("detect_asset_fields: found %s (%r)", field_id, name)

    if result:
        logger.info(
            "Detected %d Assets field(s): %s",
            len(result),
            ", ".join(sorted(result)),
        )
    else:
        logger.info("No Assets fields detected in %s", fields_json.name)
    return result


async def fetch_asset_object_labels(
    base_url: str,
    object_type_name: str,
    auth: tuple | None,
    auth_headers: dict,
) -> list[str]:
    """Fetch all object labels for a given Assets object type via AQL.

    Uses paginated GET /rest/assets/1.0/object/aql. Each page returns up to
    1000 objects. Continues until isLast=true.

    Args:
        base_url: Jira instance base URL, e.g. "https://your-org.atlassian.net".
        object_type_name: The Assets object type to query, e.g. "Domain".
        auth: Basic auth tuple, or None when using Bearer via auth_headers.
        auth_headers: Extra headers, e.g. {"Authorization": "Bearer ..."}.

    Returns:
        Sorted list of object label strings, e.g. ["Sample Domain (ABCD-1234)", ...].
    """
    headers = {"Accept": "application/json", **auth_headers}
    base = base_url.rstrip("/")
    ql_query = f'objectType = "{object_type_name}"'
    labels: list[str] = []
    page = 1

    async with tls.httpx_client(timeout=60, follow_redirects=False) as client:
        while True:
            resp = await client.get(
                f"{base}/rest/assets/1.0/object/aql",
                params={
                    "qlQuery": ql_query,
                    "includeAttributes": "false",
                    "maxResults": 1000,
                    "page": page,
                },
                auth=auth if auth and any(auth) else None,
                headers=headers,
            )
            if 300 <= resp.status_code < 400:
                logger.warning(
                    "Assets AQL endpoint returned redirect (HTTP %s) — "
                    "Assets module may not be available on this instance.",
                    resp.status_code,
                )
                return []
            if resp.status_code == 404:
                logger.warning(
                    "Assets AQL endpoint not found (404) — ensure Assets module is enabled "
                    "and the Jira instance is Cloud."
                )
                return []
            if resp.status_code in (401, 403):
                logger.warning(
                    "Assets API access denied (HTTP %s) for object type %r",
                    resp.status_code, object_type_name,
                )
                return []
            resp.raise_for_status()
            data = resp.json()
            entries = data.get("objectEntries", [])
            labels.extend(e["label"] for e in entries if "label" in e)
            if data.get("isLast", True):
                break
            page += 1

    labels = sorted(set(labels))
    logger.info("Fetched %d labels for object type %r", len(labels), object_type_name)
    return labels


async def fetch_labels_for_config(
    asset_config: dict[str, dict],
    output_json: Path,
    base_url: str,
    auth: tuple | None,
    auth_headers: dict,
) -> None:
    """Fetch and write asset object labels for each field in asset_config.

    Core fetch + write routine. Called by refresh_asset_values() (CLI) and
    AtlasMind.run() (startup). Writes jira_assets.json with the results.

    Output format::

        {
            "customfield_10200": {
                "display_name": "Domain",
                "object_type": "Domain",
                "labels": ["Sample Domain (ABCD-1234)", ...]
            }
        }

    Args:
        asset_config: {field_id: {"display_name": ..., "object_type": ...}}.
        output_json: Destination for jira_assets.json.
        base_url: Jira instance base URL.
        auth: Basic auth tuple, or None when using Bearer via auth_headers.
        auth_headers: Extra auth headers.
    """
    result: dict[str, dict] = {}
    for field_id, cfg in asset_config.items():
        object_type = cfg.get("object_type", "")
        display_name = cfg.get("display_name", field_id)
        if not object_type:
            logger.warning("No object_type for field %s — skipping", field_id)
            continue
        try:
            labels = await fetch_asset_object_labels(base_url, object_type, auth, auth_headers)
        except Exception as exc:
            logger.error(
                "Failed to read asset field %s (object_type=%r) — skipping. Error: %s",
                field_id, object_type, exc,
            )
            continue
        result[field_id] = {
            "display_name": display_name,
            "object_type": object_type,
            "labels": labels,
        }
        if not labels:
            logger.warning(
                "0 labels returned for field %s (object_type=%r) — verify the object type name "
                "matches exactly; add an override in config/jira_assets_fields.json if it differs.",
                field_id, object_type,
            )
        else:
            logger.info("field_id=%s  object_type=%r  labels=%d", field_id, object_type, len(labels))

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved asset values for %d field(s) to %s", len(result), output_json)


async def refresh_asset_values(
    output_json: Path | None = None,
    base_url: str | None = None,
    auth: tuple | None = None,
) -> None:
    """Force-refresh asset object labels from Jira, bypassing the startup hash gate.

    Auto-detects Assets fields from jira_fields.json, merges any overrides from
    config/jira_assets_fields.json, then re-fetches all labels from the Jira Assets
    AQL API and writes jira_assets.json.

    Run after bulk changes to asset objects in Jira:
        uv run python -c "
        import asyncio
        from jira.jira_assets_api import refresh_asset_values
        asyncio.run(refresh_asset_values())
        "

    Args:
        output_json: Destination for jira_assets.json. Defaults to
            data/<hostname>/jira_assets.json.
        base_url: Jira instance base URL. Derived from active profile when None.
        auth: Basic auth tuple. Derived from active profile when None.
    """
    profile = load_active_profile()
    data_dir = get_data_dir(profile["jira_url"])

    fields_json = data_dir / JIRA_FIELDS_FILENAME
    asset_config = detect_asset_fields(fields_json)

    override_file = Path(JIRA_ASSETS_CONFIG_FILE)
    if override_file.exists():
        overrides: dict = json.loads(override_file.read_text(encoding="utf-8"))
        # Per-field overrides (e.g. {"customfield_xxx": {"display_name": ..., "object_type": ...}}).
        # asset_field_keywords is read internally by detect_asset_fields — not an override.
        field_overrides = {
            k: v for k, v in overrides.items()
            if isinstance(v, dict)
        }
        skipped = {k: type(v).__name__ for k, v in overrides.items() if not isinstance(v, dict)}
        if skipped:
            logger.info(
                "Skipped %d non-dict config entry(ies) from %s: %s",
                len(skipped), override_file.name, skipped,
            )
        if field_overrides:
            # Full-entry replace by design: override supplies complete {display_name, object_type}
            # for any field whose auto-detected values are wrong (e.g. display name differs from AQL type).
            asset_config.update(field_overrides)
            logger.info("Applied %d override(s) from %s", len(field_overrides), override_file.name)

    if not asset_config:
        logger.info("No Assets fields detected or configured — nothing to fetch.")
        return

    if output_json is None:
        output_json = data_dir / JIRA_ASSETS_FILENAME
    if base_url is None:
        base_url = profile["jira_url"]
    if auth is None:
        auth, auth_headers = build_jira_auth(profile)
    else:
        auth_headers = {}

    await fetch_labels_for_config(asset_config, output_json, base_url, auth, auth_headers)


def list_asset_fields(fields_json: Path | None = None) -> None:
    """Print all Assets-type custom fields found in the cached jira_fields.json.

    Uses the configured asset_field_keywords from config/jira_assets_fields.json
    (defaulting to [".insight", ".cmdb"]) to detect Assets fields via schema.custom.

    Run::

        uv run python -c "
        from jira.jira_assets_api import list_asset_fields
        list_asset_fields()
        "
    """
    from config.jira_config import get_data_dir, load_active_profile
    from settings import JIRA_FIELDS_FILENAME

    if fields_json is None:
        profile = load_active_profile()
        fields_json = get_data_dir(profile["jira_url"]) / JIRA_FIELDS_FILENAME

    if not fields_json.exists():
        print(f"jira_fields.json not found at {fields_json}.")
        print("Run: uv run python -c \"from jira.jira_field_api import fetch_and_save_fields; fetch_and_save_fields()\"")
        return

    asset_keywords = _load_asset_keywords_from_config()
    raw: dict = json.loads(fields_json.read_text(encoding="utf-8"))
    found = [
        (fid, f.get("name", fid), (f.get("schema") or {}).get("custom", ""))
        for fid, f in raw.items()
        if any(kw in (f.get("schema") or {}).get("custom", "") for kw in asset_keywords)
    ]

    if not found:
        print("No Assets-type fields found in jira_fields.json.")
        print("If your Jira instance uses Assets, ensure jira_fields.json is up to date.")
        return

    print(f"Assets fields found in {fields_json.name}:")
    for fid, name, custom in sorted(found, key=lambda x: x[0]):
        print(f"  {fid:<24}  {name:<30}  {custom}")
    print(
        f"\nThese are auto-detected at startup. Add entries to config/jira_assets_fields.json"
        f" only to override the AQL object type name when it differs from the field display name."
    )


_ASSETS_RESOLVE_BATCH = 200


async def resolve_asset_object_refs(
    raw_issues: list[dict],
    asset_field_ids: set[str],
    base_url: str,
    auth: tuple | None,
    auth_headers: dict,
) -> None:
    """Resolve CMDB object references in raw Cloud Jira issues to human-readable labels.

    Cloud Jira Assets fields return object references instead of labels:
        [{"workspaceId": "...", "id": "...:objectId", "objectId": "1502194"}]

    Patches a "name" key into each ref dict so _extract_field_value returns the
    label string instead of None. Mutates raw_issues in place. Cloud-only.

    Groups objectIds by workspaceId and resolves each group in batches of
    _ASSETS_RESOLVE_BATCH via POST /jsm/assets/workspace/{workspaceId}/v1/object/aql.

    Args:
        raw_issues: Raw issue dicts from Jira search response.
        asset_field_ids: Field IDs whose values are Assets object references.
        base_url: Jira Cloud instance base URL.
        auth: Basic auth tuple or None.
        auth_headers: Bearer or other extra auth headers.
    """
    # workspace_id -> {object_id -> [ref dicts to patch]}
    workspace_map: dict[str, dict[str, list[dict]]] = {}

    for issue in raw_issues:
        fields = issue.get("fields", {})
        for field_id in asset_field_ids:
            value = fields.get(field_id)
            if not isinstance(value, list):
                continue
            for ref in value:
                if not isinstance(ref, dict):
                    continue
                workspace_id = ref.get("workspaceId")
                object_id = str(ref.get("objectId", "")).strip()
                if not workspace_id or not object_id:
                    continue
                ws = workspace_map.setdefault(workspace_id, {})
                ws.setdefault(object_id, []).append(ref)

    if not workspace_map:
        return

    base = base_url.rstrip("/")
    headers = {"Accept": "application/json", "Content-Type": "application/json", **auth_headers}
    resolved_total = 0
    auth_failed = False

    # Hoist client outside both loops to reuse the connection pool across all
    # workspaces and batches (avoids a new TLS handshake per batch).
    async with tls.httpx_client(timeout=30) as client:
        for workspace_id, obj_map in workspace_map.items():
            if auth_failed:
                break
            object_ids = list(obj_map.keys())
            url = f"{base}/jsm/assets/workspace/{workspace_id}/v1/object/aql"

            # Resolve in batches to keep AQL query length bounded.
            for i in range(0, len(object_ids), _ASSETS_RESOLVE_BATCH):
                batch = object_ids[i : i + _ASSETS_RESOLVE_BATCH]
                # Quote each ID - objectIds may be non-numeric (UUIDs, compound strings).
                aql_query = "objectId in ({})".format(", ".join(f'"{oid}"' for oid in batch))
                try:
                    resp = await client.post(
                        url,
                        json={"aqlQuery": aql_query},
                        auth=auth if auth and any(auth) else None,
                        headers=headers,
                    )
                    # Auth/not-found failures are permanent - stop all workspace loops.
                    if resp.status_code in (401, 403, 404):
                        logger.warning(
                            "Assets resolve: HTTP %s for workspace %s - object labels will be null",
                            resp.status_code, workspace_id,
                        )
                        auth_failed = True
                        break
                    # Rate-limit and server errors - abort this workspace's batches.
                    if resp.status_code == 429 or resp.status_code >= 500:
                        logger.warning(
                            "Assets resolve: HTTP %s for workspace %s - aborting remaining batches",
                            resp.status_code, workspace_id,
                        )
                        break
                    resp.raise_for_status()
                    data = resp.json()
                    # JSM v1 returns {"values": [...]}, older endpoint {"objectEntries": [...]}.
                    # Use key presence check, not `or`, so an empty page doesn't fall through.
                    if "values" in data:
                        entries = data["values"]
                    else:
                        entries = data.get("objectEntries", [])
                    for entry in entries:
                        label = entry.get("label", "")
                        if not label:
                            continue
                        # entry "id" may be plain objectId or compound "workspaceId:objectId"
                        raw_id = str(entry.get("id", ""))
                        entry_obj_id = raw_id.split(":")[-1] if ":" in raw_id else raw_id
                        if entry_obj_id in obj_map:
                            for ref in obj_map[entry_obj_id]:
                                ref["name"] = label
                            resolved_total += 1
                except Exception as exc:
                    logger.warning(
                        "Assets resolve: failed for workspace %s batch %d-%d - %s",
                        workspace_id, i, i + len(batch), exc,
                    )

    if resolved_total:
        logger.info(
            "Assets resolve: patched labels for %d object(s) across %d workspace(s)",
            resolved_total, len(workspace_map),
        )
    elif workspace_map:
        logger.warning(
            "Assets resolve: 0 labels patched for %d object(s) - "
            "check Assets API access or workspace IDs",
            sum(len(v) for v in workspace_map.values()),
        )


def load_asset_data(assets_file: Path) -> tuple[set[str], dict[str, list[str]]]:
    """Read jira_assets.json and return (field_ids, allowed_values_dict).

    Used at startup to:
    - Populate asset_field_ids so _build_prompt routes asset fields to the
      dedicated jira_asset_values table instead of jira_field_values.
    - Merge asset labels into allowed_values so JqlSanitizer can validate
      asset values in generated JQL (Pass 7).

    Args:
        assets_file: Path to data/<hostname>/jira_assets.json.

    Returns:
        Tuple of:
            - set of field IDs that are asset-type fields.
            - dict mapping field_id → sorted list of label strings.
    """
    if not assets_file.exists():
        return set(), {}
    data: dict = json.loads(assets_file.read_text(encoding="utf-8"))
    field_ids = set(data.keys())
    allowed = {
        fid: sorted(cfg.get("labels", []))
        for fid, cfg in data.items()
        if cfg.get("labels")
    }
    return field_ids, allowed
