"""
field_resolver.py — Jira field name resolution for intent field handling.

Translates LLM-proposed display names to Jira field IDs, validates them
against jira_fields.json, and builds display name lists for the frontend.

Classes:
    ExtraField           — a single field to extract generically from a raw issue
    ResolvedIntentFields — result of resolving LLM-proposed names to field IDs
    FieldResolver        — stateful resolver built from jira_fields.json at startup
"""

import json
import logging
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ExtraField(BaseModel):
    """A single Jira field resolved from an LLM-proposed display name."""
    field_id: str
    display_name: str


class ResolvedIntentFields(BaseModel):
    """Parallel lists of resolved field IDs and their canonical display names."""
    field_ids: list[str] = []
    display_names: list[str] = []

    def as_extra_fields(self) -> list[ExtraField]:
        """Return field_ids and display_names zipped as ExtraField objects."""
        return [
            ExtraField(field_id=fid, display_name=name)
            for fid, name in zip(self.field_ids, self.display_names)
        ]

    def is_empty(self) -> bool:
        return len(self.field_ids) == 0


class FieldResolver:
    """
    Resolves LLM-proposed field display names to Jira field IDs.

    Built once from jira_fields.json at startup. Handles:
    - Case-insensitive, whitespace-stripped name lookup
    - Duplicate name detection with warnings
    - Field count cap (MAX_INTENT_FIELDS)
    - Hallucinated name detection with warnings
    - Standard field display name lookup for the frontend

    Usage:
        resolver = FieldResolver.from_file(path, max_intent_fields=5)
        resolved = resolver.resolve(["Story Points", "sprint"])
        # resolved.field_ids    -> ["customfield_10016", "customfield_10020"]
        # resolved.display_names -> ["Story Points", "Sprint"]
    """

    def __init__(self, jira_fields: dict[str, dict], max_intent_fields: int = 5):
        self._max = max_intent_fields
        self._name_to_id: dict[str, str] = {}       # normalized name → field_id
        self._id_to_name: dict[str, str] = {}       # field_id → canonical display name
        self._build(jira_fields)

    @classmethod
    def from_file(cls, path: Path, max_intent_fields: int = 5) -> "FieldResolver":
        """Construct a FieldResolver from a jira_fields.json path (used in tests)."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(raw, max_intent_fields)

    @classmethod
    def from_db_mappings(
        cls,
        name_to_id: dict[str, str],
        id_to_name: dict[str, str],
        max_intent_fields: int = 5,
    ) -> "FieldResolver":
        """Construct a FieldResolver from mappings fetched directly from the vector DB.

        This is the preferred runtime constructor — the DB is the authoritative
        source for which fields are available, covering both intent field name
        resolution and field ID validation.

        Args:
            name_to_id: {field_name.strip().lower(): field_id} from the DB.
            id_to_name: {field_id: canonical_field_name} from the DB.
            max_intent_fields: Maximum number of intent fields the LLM may propose.
        """
        instance = cls.__new__(cls)
        instance._max = max_intent_fields
        instance._name_to_id = name_to_id
        instance._id_to_name = id_to_name
        logger.info(
            "FieldResolver: %d field names loaded from vector DB", len(name_to_id)
        )
        return instance

    def _build(self, jira_fields: dict[str, dict]) -> None:
        """Build lookup maps with duplicate detection."""
        seen: dict[str, str] = {}
        id_to_name: dict[str, str] = {}

        for field_id, field in jira_fields.items():
            name: str = field.get("name", field_id)
            key = name.strip().lower()

            if key in seen:
                logger.warning(
                    "Duplicate field name %r in jira_fields.json: %s vs %s — keeping %s",
                    name, seen[key], field_id, seen[key],
                )
            else:
                seen[key] = field_id

            # Always store the canonical display name keyed by field_id,
            # even for duplicates, so id_to_name lookup is always complete.
            id_to_name[field_id] = name

        self._name_to_id = seen
        self._id_to_name = id_to_name
        logger.info("FieldResolver: %d unique field names loaded from jira_fields.json", len(seen))

    def resolve(self, intent_names: list[str] | None) -> ResolvedIntentFields:
        """
        Translate LLM-proposed display names to field IDs.

        Applies the field count cap, validates names case-insensitively against
        the loaded field set, and logs any names that were discarded.

        Args:
            intent_names: Raw list from the LLM response. May be None or empty.

        Returns:
            ResolvedIntentFields with parallel field_ids and display_names.
            Unknown or excess names are silently dropped after being logged.
        """
        if not intent_names:
            return ResolvedIntentFields()

        if len(intent_names) > self._max:
            logger.warning(
                "LLM proposed %d intent_fields (cap=%d) — excess discarded: %s",
                len(intent_names), self._max, intent_names[self._max:],
            )

        capped = intent_names[:self._max]
        field_ids: list[str] = []
        display_names: list[str] = []
        invalid: list[str] = []

        for name in capped:
            key = name.strip()
            fid = self._name_to_id.get(key.lower())
            if not fid and key in self._id_to_name:
                # LLM used the field ID directly instead of the display name — accept it.
                logger.debug(
                    "intent_field %r matched by field ID (expected display name %r)",
                    key, self._id_to_name[key],
                )
                fid = key
            if fid:
                field_ids.append(fid)
                display_names.append(self._id_to_name[fid])
            else:
                invalid.append(name)

        if invalid:
            logger.warning(
                "LLM proposed unknown intent_fields (discarded): %s", invalid
            )

        return ResolvedIntentFields(field_ids=field_ids, display_names=display_names)

    def validate_field_ids(self, field_ids: list[str], known_ids: set[str]) -> list[str]:
        """
        Filter a list of field IDs to those present in the vector DB.

        Accepts both raw field IDs (e.g. "customfield_1234") and display names
        (e.g. "Domain"). Display names are resolved to their field ID via
        name_to_id before validation, so STANDARD_FIELD_IDS may use either form.

        The caller supplies known_ids derived from the id_to_name mapping
        returned by Jira_Field_Embeddings.fetch_field_mappings(), making the
        vector DB the authoritative source rather than the local JSON file.

        Missing IDs are logged as warnings and dropped. Called at startup to
        validate STANDARD_FIELD_IDS before any queries.

        Args:
            field_ids:  Desired field IDs or display names (e.g. from settings.py).
            known_ids:  Set of field_ids currently indexed in the vector DB.

        Returns:
            Subset of field_ids (resolved to raw IDs) that exist in the DB,
            in original order.
        """
        valid: list[str] = []
        missing: list[str] = []
        for fid in field_ids:
            if fid in known_ids:
                valid.append(fid)
            else:
                resolved = self._name_to_id.get(fid.strip().lower())
                if resolved and resolved in known_ids:
                    logger.info(
                        "STANDARD_FIELD_IDS: %r resolved to field ID %r", fid, resolved
                    )
                    valid.append(resolved)
                else:
                    missing.append(fid)
        if missing:
            logger.warning(
                "Field IDs not found in vector DB (excluded from use): %s", missing
            )
        return valid

    def build_fields_param(
        self,
        base_field_ids: list[str],
        extra_field_ids: list[str] | None = None,
    ) -> str:
        """
        Build the comma-separated fields string for the Jira REST API.

        Combines the validated base set with any per-query intent field IDs,
        deduplicating so the same field is not requested twice.

        Args:
            base_field_ids: Standard field IDs (from AtlasMind.standard_field_ids).
            extra_field_ids: Intent field IDs resolved for this specific query.

        Returns:
            Comma-separated string ready for the Jira /rest/api/2/search fields param.
        """
        seen: set[str] = set()
        parts: list[str] = []
        for fid in (base_field_ids + (extra_field_ids or [])):
            if fid not in seen:
                seen.add(fid)
                parts.append(fid)
        return ",".join(parts)

    def display_names_for_ids(self, field_ids: list[str]) -> list[str]:
        """
        Return canonical display names for a list of field IDs.
        Falls back to the raw field_id string when the ID is not in the map.
        """
        return [self._id_to_name.get(fid, fid) for fid in field_ids]

    def filter_to_known_ids(self, candidates: list[str]) -> list[str]:
        """Return only candidates that are valid field IDs in the resolver's map."""
        return [fid for fid in candidates if fid in self._id_to_name]
