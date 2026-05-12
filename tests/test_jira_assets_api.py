"""
tests/test_jira_assets_api.py

Tests for Jira Assets field detection and label fetching:
  - detect_asset_fields() reads keywords from config at runtime
  - keyword-based detection (not hardcoded prefix)
  - non-dict config entries are not treated as field overrides
  - refresh_asset_values() skips non-dict entries gracefully
"""

import json
import logging
import pytest
from pathlib import Path
from unittest.mock import patch

from jira.jira_assets_api import detect_asset_fields


# ---------------------------------------------------------------------------
# Sample field data — system field + Assets fields using various plugins
# ---------------------------------------------------------------------------

SAMPLE_FIELDS = {
    "customfield_1556": {
        "id": "customfield_1836",
        "name": "Domain",
        "custom": True,
        "schema": {
            "type": "any",
            "custom": "com.abcd.jira.plugins.insight:rlabs-customfield-default-object",
            "customId": 1556,
        },
    },
    "customfield_20001": {
        "name": "Application",
        "custom": True,
        "schema": {
            "type": "any",
            "custom": "com.atlassian.jira.plugins.cmdb:objectfield",
            "customId": 20001,
        },
    },
    "customfield_30001": {
        "name": "Related CI",
        "custom": True,
        "schema": {
            "type": "any",
            "custom": "com.atlaslabs.jira.plugins.insight:object-type-field",
            "customId": 30001,
        },
    },
    "customfield_40001": {
        "name": "Team",
        "custom": True,
        "schema": {
            "type": "any",
            "custom": "com.atlaslabs.jira.plugins.insight:related-object-field",
            "customId": 40001,
        },
    },
    "customfield_90001": {
        "name": "Related Tickets",
        "custom": True,
        "schema": {
            "type": "array",
            "custom": "com.example.ticket-ref",
            "customId": 90001,
        },
    },
    "priority": {
        "name": "Priority",
        "custom": False,
        "schema": {"type": "option"},
    },
}


# ---------------------------------------------------------------------------
# detect_asset_fields — keyword detection
# ---------------------------------------------------------------------------

class TestDetectAssetFields:

    def test_detects_insight_prefix_field(self, tmp_path):
        """Field with com.abcd.jira.plugins.insight in schema.custom is detected."""
        fields_file = tmp_path / "jira_fields.json"
        fields_file.write_text(json.dumps({"customfield_1556": SAMPLE_FIELDS["customfield_1556"]}))
        result = detect_asset_fields(fields_file)
        assert "customfield_1556" in result
        assert result["customfield_1556"]["display_name"] == "Domain"
        assert result["customfield_1556"]["object_type"] == "Domain"

    def test_detects_cmdb_prefix_field(self, tmp_path):
        """Field with com.atlassian.jira.plugins.cmdb in schema.custom is detected."""
        fields_file = tmp_path / "jira_fields.json"
        fields_file.write_text(json.dumps({"customfield_20001": SAMPLE_FIELDS["customfield_20001"]}))
        result = detect_asset_fields(fields_file)
        assert "customfield_20001" in result

    def test_detects_atlaslabs_insight_prefix_field(self, tmp_path):
        """Field with com.atlaslabs.jira.plugins.insight in schema.custom is detected."""
        fields_file = tmp_path / "jira_fields.json"
        fields_file.write_text(json.dumps({"customfield_30001": SAMPLE_FIELDS["customfield_30001"]}))
        result = detect_asset_fields(fields_file)
        assert "customfield_30001" in result

    def test_ignores_non_asset_custom_field(self, tmp_path):
        """Field with unrelated custom schema is not detected as Assets field."""
        fields_file = tmp_path / "jira_fields.json"
        fields_file.write_text(json.dumps({"customfield_90001": SAMPLE_FIELDS["customfield_90001"]}))
        result = detect_asset_fields(fields_file)
        assert "customfield_90001" not in result

    def test_ignores_system_fields(self, tmp_path):
        """Non-custom system fields are not detected as Assets fields."""
        fields_file = tmp_path / "jira_fields.json"
        fields_file.write_text(json.dumps({"priority": SAMPLE_FIELDS["priority"]}))
        result = detect_asset_fields(fields_file)
        assert "priority" not in result

    def test_returns_empty_when_no_fields(self, tmp_path):
        """Returns empty dict when no fields exist."""
        fields_file = tmp_path / "jira_fields.json"
        fields_file.write_text(json.dumps({}))
        result = detect_asset_fields(fields_file)
        assert result == {}

    def test_returns_empty_when_file_missing(self, tmp_path):
        """Returns empty dict when jira_fields.json does not exist."""
        result = detect_asset_fields(tmp_path / "nonexistent.json")
        assert result == {}

    def test_uses_custom_keywords_from_config(self, tmp_path):
        """Custom asset_field_keywords from config override defaults."""
        fields_file = tmp_path / "jira_fields.json"
        fields_file.write_text(json.dumps({"customfield_40001": SAMPLE_FIELDS["customfield_40001"]}))

        # Pass custom keywords directly — "atlaslabs" matches com.atlaslabs.jira.plugins.insight
        result = detect_asset_fields(fields_file, asset_keywords=["atlaslabs"])
        assert "customfield_40001" in result

    def test_custom_keywords_not_matching_defaults(self, tmp_path):
        """Custom keywords that don't match any field return empty result."""
        fields_file = tmp_path / "jira_fields.json"
        fields_file.write_text(json.dumps({"customfield_1556": SAMPLE_FIELDS["customfield_1556"]}))

        # .xyz does not match com.abcd.jira.plugins.insight:...
        result = detect_asset_fields(fields_file, asset_keywords=[".xyz"])
        assert result == {}

    def test_uses_default_keywords_when_config_missing(self, tmp_path, monkeypatch):
        """Defaults to [".insight", ".cmdb"] when config or asset_field_keywords absent."""
        fields_file = tmp_path / "jira_fields.json"
        fields_file.write_text(json.dumps({"customfield_1556": SAMPLE_FIELDS["customfield_1556"]}))

        config_file = tmp_path / "jira_assets_fields.json"
        config_file.write_text(json.dumps({"some_other_key": "value"}))

        monkeypatch.setattr(
            "jira.jira_assets_api.JIRA_ASSETS_CONFIG_FILE",
            config_file,
        )
        result = detect_asset_fields(fields_file)
        # .insight is in defaults, should match
        assert "customfield_1556" in result


# ---------------------------------------------------------------------------
# refresh_asset_values — non-dict config entry handling
# ---------------------------------------------------------------------------

class TestRefreshAssetValuesNonDictHandling:
    """asset_field_keywords list must not crash refresh_asset_values."""

    def test_skips_non_dict_config_entries(self, tmp_path, monkeypatch):
        """refresh_asset_values skips entries whose value is not a dict."""
        from jira.jira_assets_api import refresh_asset_values

        fields_file = tmp_path / "jira_fields.json"
        fields_file.write_text(json.dumps({}))

        config_file = tmp_path / "jira_assets_fields.json"
        config_file.write_text(json.dumps({
            "asset_field_keywords": [".insight", ".cmdb"],
        }))

        assets_file = tmp_path / "jira_assets.json"

        monkeypatch.setattr(
            "jira.jira_assets_api.JIRA_FIELDS_FILENAME",
            fields_file.name,
        )
        monkeypatch.setattr(
            "jira.jira_assets_api.JIRA_ASSETS_CONFIG_FILE",
            config_file,
        )
        monkeypatch.setattr(
            "jira.jira_assets_api.get_data_dir",
            lambda _: tmp_path,
        )
        monkeypatch.setattr(
            "jira.jira_assets_api.load_active_profile",
            lambda: {"jira_url": "https://example.atlassian.net"},
        )

        # Should not raise — the list entry must be skipped gracefully
        import asyncio
        try:
            asyncio.run(refresh_asset_values(output_json=assets_file))
        except AttributeError:
            pytest.fail(
                "refresh_asset_values crashed on non-dict config entry. "
                "asset_field_keywords list must be skipped."
            )
