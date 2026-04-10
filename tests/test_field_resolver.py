"""
tests/test_field_resolver.py

Tests for the LLM-driven intent field design:
  - FieldResolver  (construction, resolve, validate, build_fields_param, display_names)
  - ResolvedIntentFields model helpers
  - normalize_issue() extra_fields handling (core/atlasmind.py)
  - server._build_display_fields()
"""

import json
import logging
import pytest
from unittest.mock import MagicMock, patch

import server
from core.atlasmind import normalize_issue
from core.field_resolver import ExtraField, FieldResolver, ResolvedIntentFields


# ===========================================================================
# FieldResolver — construction
# ===========================================================================

class TestFieldResolverConstruction:

    def test_from_db_mappings_loads_maps(self, sample_name_to_id, sample_id_to_name):
        r = FieldResolver.from_db_mappings(sample_name_to_id, sample_id_to_name)
        assert r._name_to_id == sample_name_to_id
        assert r._id_to_name == sample_id_to_name

    def test_from_db_mappings_respects_max(self, sample_name_to_id, sample_id_to_name):
        r = FieldResolver.from_db_mappings(sample_name_to_id, sample_id_to_name, max_intent_fields=2)
        assert r._max == 2

    def test_from_file_reads_json(self, jira_fields_json, sample_name_to_id, sample_id_to_name):
        r = FieldResolver.from_file(jira_fields_json)
        for key, fid in sample_name_to_id.items():
            assert r._name_to_id.get(key) == fid
        for fid, name in sample_id_to_name.items():
            assert r._id_to_name.get(fid) == name

    def test_from_file_duplicate_name_first_seen_wins(self, tmp_path):
        fields = {
            "customfield_001": {"name": "Story Points"},
            "customfield_002": {"name": "Story Points"},
        }
        p = tmp_path / "fields.json"
        p.write_text(json.dumps(fields), encoding="utf-8")
        r = FieldResolver.from_file(p)
        assert r._name_to_id["story points"] == "customfield_001"

    def test_from_file_duplicate_name_both_in_id_to_name(self, tmp_path):
        fields = {
            "customfield_001": {"name": "Story Points"},
            "customfield_002": {"name": "Story Points"},
        }
        p = tmp_path / "fields.json"
        p.write_text(json.dumps(fields), encoding="utf-8")
        r = FieldResolver.from_file(p)
        assert r._id_to_name["customfield_001"] == "Story Points"
        assert r._id_to_name["customfield_002"] == "Story Points"

    def test_from_file_duplicate_name_warns(self, tmp_path, caplog):
        fields = {
            "customfield_001": {"name": "Story Points"},
            "customfield_002": {"name": "Story Points"},
        }
        p = tmp_path / "fields.json"
        p.write_text(json.dumps(fields), encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="core.field_resolver"):
            FieldResolver.from_file(p)
        assert any("Duplicate" in m or "duplicate" in m for m in caplog.messages)


# ===========================================================================
# FieldResolver — resolve()
# ===========================================================================

class TestFieldResolverResolve:

    def test_resolve_happy_path(self, resolver):
        result = resolver.resolve(["Story Points", "Sprint"])
        assert result.field_ids == ["customfield_10016", "customfield_10020"]
        assert result.display_names == ["Story Points", "Sprint"]

    def test_resolve_case_insensitive(self, resolver):
        result = resolver.resolve(["STORY POINTS"])
        assert result.field_ids == ["customfield_10016"]
        assert result.display_names == ["Story Points"]

    def test_resolve_whitespace_stripped(self, resolver):
        result = resolver.resolve(["  Epic Link  "])
        assert result.field_ids == ["customfield_10014"]
        assert result.display_names == ["Epic Link"]

    def test_resolve_unknown_names_dropped(self, resolver):
        result = resolver.resolve(["Story Points", "NonExistentField"])
        assert result.field_ids == ["customfield_10016"]
        assert "NonExistentField" not in result.display_names

    def test_resolve_field_id_fallback(self, resolver):
        # LLM used the field ID directly ("priority") instead of display name ("Priority")
        result = resolver.resolve(["priority"])
        assert result.field_ids == ["priority"]
        assert result.display_names == ["Priority"]

    def test_resolve_field_id_fallback_custom_field(self, resolver):
        # LLM used a custom field ID directly
        result = resolver.resolve(["customfield_10016"])
        assert result.field_ids == ["customfield_10016"]
        assert result.display_names == ["Story Points"]

    def test_resolve_field_id_fallback_returns_canonical_name(self, resolver):
        # Display name returned should be canonical, not the raw ID
        result = resolver.resolve(["customfield_10020"])
        assert result.display_names == ["Sprint"]

    def test_resolve_all_unknown_returns_empty(self, resolver):
        result = resolver.resolve(["FakeA", "FakeB"])
        assert result.is_empty()

    def test_resolve_none_input_returns_empty(self, resolver):
        result = resolver.resolve(None)
        assert result.is_empty()

    def test_resolve_empty_list_returns_empty(self, resolver):
        result = resolver.resolve([])
        assert result.is_empty()

    def test_resolve_over_cap_discards_excess(self, resolver):
        # resolver has max_intent_fields=3; supply 5 valid names
        names = ["Story Points", "Sprint", "Epic Link", "Priority", "Assignee"]
        result = resolver.resolve(names)
        assert len(result.field_ids) == 3
        assert result.field_ids == [
            "customfield_10016", "customfield_10020", "customfield_10014"
        ]

    def test_resolve_over_cap_warns(self, resolver, caplog):
        names = ["Story Points", "Sprint", "Epic Link", "Priority", "Assignee"]
        with caplog.at_level(logging.WARNING, logger="core.field_resolver"):
            resolver.resolve(names)
        assert any("cap" in m.lower() or "excess" in m.lower() for m in caplog.messages)

    def test_resolve_unknown_names_warn(self, resolver, caplog):
        with caplog.at_level(logging.WARNING, logger="core.field_resolver"):
            resolver.resolve(["FakeField"])
        assert any("unknown" in m.lower() or "discarded" in m.lower() for m in caplog.messages)

    def test_resolve_returns_canonical_display_names(self, resolver):
        # Input is lowercase; canonical name should be title-case
        result = resolver.resolve(["story points", "sprint"])
        assert result.display_names == ["Story Points", "Sprint"]


# ===========================================================================
# FieldResolver — validate_field_ids()
# ===========================================================================

class TestFieldResolverValidateFieldIds:

    def test_validate_all_present(self, resolver):
        known = {"customfield_10016", "customfield_10020"}
        result = resolver.validate_field_ids(["customfield_10016", "customfield_10020"], known)
        assert result == ["customfield_10016", "customfield_10020"]

    def test_validate_missing_filtered(self, resolver):
        known = {"customfield_10016"}
        result = resolver.validate_field_ids(
            ["customfield_10016", "customfield_99999"], known
        )
        assert result == ["customfield_10016"]
        assert "customfield_99999" not in result

    def test_validate_preserves_input_order(self, resolver):
        known = {"a", "b", "c"}
        result = resolver.validate_field_ids(["c", "a", "b"], known)
        assert result == ["c", "a", "b"]

    def test_validate_empty_input(self, resolver):
        result = resolver.validate_field_ids([], {"customfield_10016"})
        assert result == []

    def test_validate_all_missing(self, resolver):
        result = resolver.validate_field_ids(["bad_1", "bad_2"], {"known_id"})
        assert result == []

    def test_validate_missing_ids_warns(self, resolver, caplog):
        with caplog.at_level(logging.WARNING, logger="core.field_resolver"):
            resolver.validate_field_ids(["customfield_10016", "bad_id"], {"customfield_10016"})
        assert any("bad_id" in m for m in caplog.messages)


# ===========================================================================
# FieldResolver — build_fields_param()
# ===========================================================================

class TestFieldResolverBuildFieldsParam:

    def test_build_no_extra(self, resolver):
        result = resolver.build_fields_param(["summary", "status"])
        assert result == "summary,status"

    def test_build_with_extra(self, resolver):
        result = resolver.build_fields_param(["summary"], ["customfield_10016"])
        assert result == "summary,customfield_10016"

    def test_build_deduplicates_overlap(self, resolver):
        result = resolver.build_fields_param(
            ["summary", "customfield_10016"],
            ["customfield_10016", "customfield_10020"],
        )
        parts = result.split(",")
        assert parts.count("customfield_10016") == 1
        assert "summary" in parts
        assert "customfield_10020" in parts

    def test_build_extra_none(self, resolver):
        result = resolver.build_fields_param(["summary", "status"], None)
        assert result == "summary,status"

    def test_build_empty_base(self, resolver):
        result = resolver.build_fields_param([], ["customfield_10016"])
        assert result == "customfield_10016"

    def test_build_empty_both(self, resolver):
        result = resolver.build_fields_param([], [])
        assert result == ""


# ===========================================================================
# FieldResolver — display_names_for_ids()
# ===========================================================================

class TestFieldResolverDisplayNames:

    def test_all_known(self, resolver):
        result = resolver.display_names_for_ids(["customfield_10016", "customfield_10020"])
        assert result == ["Story Points", "Sprint"]

    def test_fallback_to_raw_id(self, resolver):
        result = resolver.display_names_for_ids(["unknown_field_99"])
        assert result == ["unknown_field_99"]

    def test_mixed_known_and_unknown(self, resolver):
        result = resolver.display_names_for_ids(["customfield_10016", "unknown_field_99"])
        assert result == ["Story Points", "unknown_field_99"]

    def test_empty_input(self, resolver):
        result = resolver.display_names_for_ids([])
        assert result == []


# ===========================================================================
# ResolvedIntentFields — model helpers
# ===========================================================================

class TestResolvedIntentFields:

    def test_as_extra_fields_zips_correctly(self):
        r = ResolvedIntentFields(
            field_ids=["customfield_10016", "customfield_10020"],
            display_names=["Story Points", "Sprint"],
        )
        extras = r.as_extra_fields()
        assert len(extras) == 2
        assert extras[0] == ExtraField(field_id="customfield_10016", display_name="Story Points")
        assert extras[1] == ExtraField(field_id="customfield_10020", display_name="Sprint")

    def test_as_extra_fields_empty(self):
        assert ResolvedIntentFields().as_extra_fields() == []

    def test_is_empty_true_when_no_fields(self):
        assert ResolvedIntentFields().is_empty() is True

    def test_is_empty_false_when_fields_present(self):
        r = ResolvedIntentFields(
            field_ids=["customfield_10016"],
            display_names=["Story Points"],
        )
        assert r.is_empty() is False


# ===========================================================================
# normalize_issue() — extra_fields handling
# ===========================================================================

class TestNormalizeIssueExtraFields:

    def test_standard_fields_extracted(self, raw_jira_issue):
        result = normalize_issue(raw_jira_issue)
        assert result["key"] == "TEST-1"
        assert result["summary"] == "Fix login bug"
        assert result["status"] == "In Progress"
        assert result["assignee"] == "Alice"
        assert result["labels"] == ["sso", "login"]

    def test_extra_field_dict_name_extraction(self, raw_jira_issue):
        raw_jira_issue["fields"]["customfield_10100"] = {"name": "High Risk"}
        ef = ExtraField(field_id="customfield_10100", display_name="Risk Level")
        result = normalize_issue(raw_jira_issue, extra_fields=[ef])
        assert result["Risk Level"] == "High Risk"

    def test_extra_field_dict_value_fallback(self, raw_jira_issue):
        # No "name" key — should fall back to "value"
        raw_jira_issue["fields"]["customfield_10200"] = {"value": "External"}
        ef = ExtraField(field_id="customfield_10200", display_name="Scope")
        result = normalize_issue(raw_jira_issue, extra_fields=[ef])
        assert result["Scope"] == "External"

    def test_extra_field_list_of_dicts_flattened(self, raw_jira_issue):
        raw_jira_issue["fields"]["customfield_10300"] = [
            {"name": "Team A"},
            {"name": "Team B"},
        ]
        ef = ExtraField(field_id="customfield_10300", display_name="Teams")
        result = normalize_issue(raw_jira_issue, extra_fields=[ef])
        assert result["Teams"] == ["Team A", "Team B"]

    def test_extra_field_list_mixed_types(self, raw_jira_issue):
        raw_jira_issue["fields"]["customfield_10400"] = [
            {"name": "Label X"},
            "raw-string",
        ]
        ef = ExtraField(field_id="customfield_10400", display_name="MixedList")
        result = normalize_issue(raw_jira_issue, extra_fields=[ef])
        assert result["MixedList"] == ["Label X", "raw-string"]

    def test_extra_field_primitive_int(self, raw_jira_issue):
        raw_jira_issue["fields"]["customfield_10500"] = 13
        ef = ExtraField(field_id="customfield_10500", display_name="Story Points")
        result = normalize_issue(raw_jira_issue, extra_fields=[ef])
        assert result["Story Points"] == 13

    def test_extra_field_none_when_field_id_absent(self, raw_jira_issue):
        # field_id not present in fields — stored as None
        ef = ExtraField(field_id="customfield_99999", display_name="Missing Field")
        result = normalize_issue(raw_jira_issue, extra_fields=[ef])
        assert result["Missing Field"] is None

    def test_extra_field_collision_with_standard_key_skipped(self, raw_jira_issue):
        # "summary" already exists as a standard key — extra field must not overwrite it
        raw_jira_issue["fields"]["customfield_99999"] = "SHOULD NOT APPEAR"
        ef = ExtraField(field_id="customfield_99999", display_name="summary")
        result = normalize_issue(raw_jira_issue, extra_fields=[ef])
        assert result["summary"] == "Fix login bug"

    def test_multiple_extra_fields(self, raw_jira_issue):
        raw_jira_issue["fields"]["customfield_10100"] = {"name": "Done"}
        raw_jira_issue["fields"]["customfield_10200"] = 5
        extras = [
            ExtraField(field_id="customfield_10100", display_name="Outcome"),
            ExtraField(field_id="customfield_10200", display_name="Votes"),
        ]
        result = normalize_issue(raw_jira_issue, extra_fields=extras)
        assert result["Outcome"] == "Done"
        assert result["Votes"] == 5


# ===========================================================================
# server._build_display_fields()
# ===========================================================================

class TestBuildDisplayFields:

    def _mock_atlasmind(self, sample_name_to_id, sample_id_to_name, standard_ids):
        mock_am = MagicMock()
        mock_am.field_resolver = FieldResolver.from_db_mappings(
            sample_name_to_id, sample_id_to_name
        )
        mock_am.standard_field_ids = standard_ids
        return mock_am

    def test_standard_and_intent_names_combined(self, sample_name_to_id, sample_id_to_name):
        mock_am = self._mock_atlasmind(
            sample_name_to_id, sample_id_to_name,
            standard_ids=["assignee", "priority"],
        )
        resolved = ResolvedIntentFields(
            field_ids=["customfield_10016"],
            display_names=["Story Points"],
        )
        with patch.object(server, "_atlasmind", mock_am):
            result = server._build_display_fields(resolved)
        assert result == ["Assignee", "Priority", "Story Points"]

    def test_only_standard_when_no_intent(self, sample_name_to_id, sample_id_to_name):
        mock_am = self._mock_atlasmind(
            sample_name_to_id, sample_id_to_name,
            standard_ids=["assignee"],
        )
        with patch.object(server, "_atlasmind", mock_am):
            result = server._build_display_fields(ResolvedIntentFields())
        assert result == ["Assignee"]

    def test_fallback_to_raw_ids_when_no_resolver(self):
        mock_am = MagicMock()
        mock_am.field_resolver = None
        mock_am.standard_field_ids = ["key", "summary"]
        with patch.object(server, "_atlasmind", mock_am):
            result = server._build_display_fields(ResolvedIntentFields())
        assert result == ["key", "summary"]

    def test_unknown_standard_ids_fall_back_to_raw(self, sample_name_to_id, sample_id_to_name):
        # "key" is not in id_to_name — should fall back to raw ID string
        mock_am = self._mock_atlasmind(
            sample_name_to_id, sample_id_to_name,
            standard_ids=["key", "assignee"],
        )
        with patch.object(server, "_atlasmind", mock_am):
            result = server._build_display_fields(ResolvedIntentFields())
        assert result == ["key", "Assignee"]

    def test_atlasmind_none_returns_only_intent(self):
        resolved = ResolvedIntentFields(
            field_ids=["customfield_10016"],
            display_names=["Story Points"],
        )
        with patch.object(server, "_atlasmind", None):
            result = server._build_display_fields(resolved)
        assert result == ["Story Points"]
