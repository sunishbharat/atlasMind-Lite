"""
tests/test_cf_config_externalization.py

Verifies the CF config externalization changes across three scenarios:
  - Default / plain Docker: no new env vars set → all config reads from disk
  - CF deployment: env vars set via cf set-env / manifest → config read from env
  - Edge cases: bad JSON, missing file, fetch failure, partial env, whitespace

Sections:
  1. cloud/config_fetcher.py  — fetch_to_file HTTP fetch + Basic auth
  2. settings.py              — JIRA_FIELD_IGNORE_IDS; JQL_ANNOTATION_URL fetch
  3. config/jira_config.py    — _load_profiles_data; load_active_profile;
                                load_active_jira_profile
"""

import base64
import importlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest


# ---------------------------------------------------------------------------
# Fixture: restore settings module to a clean state after each reload test
# ---------------------------------------------------------------------------

@pytest.fixture
def settings_restored():
    """Reload settings to a known-clean state after any test that modifies it."""
    yield
    # Clear every env var that could trigger non-default behaviour on reload
    for key in (
        "JIRA_FIELD_IGNORE_IDS",
        "JQL_ANNOTATION_URL",
        "JQL_ANNOTATION_FILE",
        "CONFIG_REGISTRY_USER",
        "CONFIG_REGISTRY_TOKEN",
    ):
        os.environ.pop(key, None)
    with patch("cloud.oci_vault.resolve_secret", return_value=""):
        import settings
        importlib.reload(settings)


# ===========================================================================
# 1. cloud/config_fetcher.py — fetch_to_file
# ===========================================================================

class TestFetchToFile:
    """fetch_to_file — stdlib HTTPS download with optional Basic auth."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_resp(body: bytes = b""):
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.read.return_value = body
        return resp

    @staticmethod
    def _capture_req(body: bytes = b""):
        """Return (side_effect_fn, captured_dict). captured["req"] set on call."""
        captured: dict = {}

        def fake_urlopen(req, timeout):
            captured["req"] = req
            resp = MagicMock()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            resp.read.return_value = body
            return resp

        return fake_urlopen, captured

    # ------------------------------------------------------------------
    # happy path
    # ------------------------------------------------------------------

    def test_writes_response_bytes_to_dest(self, tmp_path):
        from cloud.config_fetcher import fetch_to_file

        dest = tmp_path / "out.md"
        with patch("cloud.config_fetcher.urlopen", return_value=self._mock_resp(b"# content")):
            fetch_to_file("https://registry.example.com/file.md", dest)

        assert dest.read_bytes() == b"# content"

    # ------------------------------------------------------------------
    # auth header
    # ------------------------------------------------------------------

    def test_no_auth_omits_authorization_header(self, tmp_path):
        from cloud.config_fetcher import fetch_to_file

        dest = tmp_path / "out.md"
        fake, captured = self._capture_req()
        with patch("cloud.config_fetcher.urlopen", side_effect=fake):
            fetch_to_file("https://registry.example.com/file.md", dest)

        assert captured["req"].get_header("Authorization") is None

    def test_username_and_token_sets_basic_auth_header(self, tmp_path):
        from cloud.config_fetcher import fetch_to_file

        dest = tmp_path / "out.md"
        fake, captured = self._capture_req()
        with patch("cloud.config_fetcher.urlopen", side_effect=fake):
            fetch_to_file("https://registry.example.com/file.md", dest,
                          username="deployer", token="registry-secret")

        expected = "Basic " + base64.b64encode(b"deployer:registry-secret").decode()
        assert captured["req"].get_header("Authorization") == expected

    def test_token_only_encodes_as_colon_token(self, tmp_path):
        from cloud.config_fetcher import fetch_to_file

        dest = tmp_path / "out.md"
        fake, captured = self._capture_req()
        with patch("cloud.config_fetcher.urlopen", side_effect=fake):
            fetch_to_file("https://registry.example.com/file.md", dest, token="registry-secret")

        expected = "Basic " + base64.b64encode(b":registry-secret").decode()
        assert captured["req"].get_header("Authorization") == expected

    # ------------------------------------------------------------------
    # failure modes
    # ------------------------------------------------------------------

    def test_http_404_raises_runtime_error_with_status_code(self, tmp_path):
        from cloud.config_fetcher import fetch_to_file

        dest = tmp_path / "out.md"
        err = HTTPError("https://registry.example.com/file.md", 404, "Not Found", {}, None)
        with patch("cloud.config_fetcher.urlopen", side_effect=err):
            with pytest.raises(RuntimeError, match=r"Config fetch failed \[404\]"):
                fetch_to_file("https://registry.example.com/file.md", dest)

    def test_http_401_raises_runtime_error_with_status_code(self, tmp_path):
        from cloud.config_fetcher import fetch_to_file

        dest = tmp_path / "out.md"
        err = HTTPError("https://registry.example.com/file.md", 401, "Unauthorized", {}, None)
        with patch("cloud.config_fetcher.urlopen", side_effect=err):
            with pytest.raises(RuntimeError, match=r"Config fetch failed \[401\]"):
                fetch_to_file("https://registry.example.com/file.md", dest)

    def test_url_error_raises_runtime_error(self, tmp_path):
        from cloud.config_fetcher import fetch_to_file

        dest = tmp_path / "out.md"
        with patch("cloud.config_fetcher.urlopen",
                   side_effect=URLError("Name or service not known")):
            with pytest.raises(RuntimeError, match="Config fetch unreachable"):
                fetch_to_file("https://registry.example.com/file.md", dest)

    def test_url_error_does_not_create_dest_file(self, tmp_path):
        from cloud.config_fetcher import fetch_to_file

        dest = tmp_path / "out.md"
        with patch("cloud.config_fetcher.urlopen", side_effect=URLError("unreachable")):
            with pytest.raises(RuntimeError):
                fetch_to_file("https://registry.example.com/file.md", dest)

        assert not dest.exists()


# ===========================================================================
# 2. settings.py — JIRA_FIELD_IGNORE_IDS
# ===========================================================================

class TestJiraFieldIgnoreIds:
    """JIRA_FIELD_IGNORE_IDS env var drives the exclusion set; default is empty."""

    def _load(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("JIRA_FIELD_IGNORE_IDS", raising=False)
        else:
            monkeypatch.setenv("JIRA_FIELD_IGNORE_IDS", value)
        with patch("cloud.oci_vault.resolve_secret", return_value=""):
            import settings
            importlib.reload(settings)
            return settings.JIRA_FIELD_IGNORE_IDS

    # default / plain Docker
    def test_unset_returns_empty_set(self, monkeypatch, settings_restored):
        assert self._load(monkeypatch, None) == set()

    def test_empty_string_returns_empty_set(self, monkeypatch, settings_restored):
        assert self._load(monkeypatch, "") == set()

    # CF deployment cases
    def test_single_field_id(self, monkeypatch, settings_restored):
        assert self._load(monkeypatch, "customfield_10200") == {"customfield_10200"}

    def test_multiple_field_ids(self, monkeypatch, settings_restored):
        result = self._load(monkeypatch, "customfield_10200,customfield_10300")
        assert result == {"customfield_10200", "customfield_10300"}

    def test_whitespace_around_ids_is_stripped(self, monkeypatch, settings_restored):
        result = self._load(monkeypatch, " customfield_10200 , customfield_10300 ")
        assert result == {"customfield_10200", "customfield_10300"}

    def test_trailing_comma_is_ignored(self, monkeypatch, settings_restored):
        assert self._load(monkeypatch, "customfield_10200,") == {"customfield_10200"}

    def test_result_type_is_set(self, monkeypatch, settings_restored):
        assert isinstance(self._load(monkeypatch, "customfield_10200"), set)


# ===========================================================================
# 3. settings.py — DEFAULT_ANNOTATION_FILE / JQL_ANNOTATION_URL
# ===========================================================================

class TestAnnotationFileResolution:
    """DEFAULT_ANNOTATION_FILE: disk fallback without CF; /tmp/ fetch with CF."""

    # ------------------------------------------------------------------
    # default / plain Docker — no JQL_ANNOTATION_URL
    # ------------------------------------------------------------------

    def test_default_uses_data_dir_file(self, monkeypatch, settings_restored):
        monkeypatch.delenv("JQL_ANNOTATION_URL", raising=False)
        monkeypatch.delenv("JQL_ANNOTATION_FILE", raising=False)
        with patch("cloud.oci_vault.resolve_secret", return_value=""):
            import settings
            importlib.reload(settings)

        expected_suffix = os.path.join("data", "jira_jql_annotated_queries.md")
        assert settings.DEFAULT_ANNOTATION_FILE.endswith(expected_suffix)

    def test_default_does_not_point_to_tmp(self, monkeypatch, settings_restored):
        monkeypatch.delenv("JQL_ANNOTATION_URL", raising=False)
        monkeypatch.delenv("JQL_ANNOTATION_FILE", raising=False)
        with patch("cloud.oci_vault.resolve_secret", return_value=""):
            import settings
            importlib.reload(settings)

        assert not settings.DEFAULT_ANNOTATION_FILE.startswith(tempfile.gettempdir())

    def test_explicit_jql_annotation_file_env_var_respected(self, monkeypatch, settings_restored, tmp_path):
        override = str(tmp_path / "custom_annotations.md")
        monkeypatch.setenv("JQL_ANNOTATION_FILE", override)
        monkeypatch.delenv("JQL_ANNOTATION_URL", raising=False)
        with patch("cloud.oci_vault.resolve_secret", return_value=""):
            import settings
            importlib.reload(settings)

        assert settings.DEFAULT_ANNOTATION_FILE == override

    # ------------------------------------------------------------------
    # CF deployment — JQL_ANNOTATION_URL set
    # ------------------------------------------------------------------

    def test_cf_url_set_calls_fetch_to_file_once(self, monkeypatch, settings_restored):
        mock_fetch = MagicMock()
        monkeypatch.setenv("JQL_ANNOTATION_URL", "https://registry.example.com/annotations.md")
        monkeypatch.delenv("CONFIG_REGISTRY_USER", raising=False)
        monkeypatch.delenv("CONFIG_REGISTRY_TOKEN", raising=False)
        with patch("cloud.oci_vault.resolve_secret", return_value=""), \
             patch("cloud.config_fetcher.fetch_to_file", mock_fetch):
            import settings
            importlib.reload(settings)

        mock_fetch.assert_called_once()

    def test_cf_url_passes_correct_url_to_fetcher(self, monkeypatch, settings_restored):
        mock_fetch = MagicMock()
        monkeypatch.setenv("JQL_ANNOTATION_URL", "https://registry.example.com/annotations.md")
        with patch("cloud.oci_vault.resolve_secret", return_value=""), \
             patch("cloud.config_fetcher.fetch_to_file", mock_fetch):
            import settings
            importlib.reload(settings)

        url_arg = mock_fetch.call_args[0][0]
        assert url_arg == "https://registry.example.com/annotations.md"

    def test_cf_url_passes_dest_filename_jira_jql_annotated_queries(self, monkeypatch, settings_restored):
        mock_fetch = MagicMock()
        monkeypatch.setenv("JQL_ANNOTATION_URL", "https://registry.example.com/annotations.md")
        with patch("cloud.oci_vault.resolve_secret", return_value=""), \
             patch("cloud.config_fetcher.fetch_to_file", mock_fetch):
            import settings
            importlib.reload(settings)

        dest_arg: Path = mock_fetch.call_args[0][1]
        assert dest_arg.name == "jira_jql_annotated_queries.md"

    def test_cf_url_passes_registry_credentials(self, monkeypatch, settings_restored):
        mock_fetch = MagicMock()
        monkeypatch.setenv("JQL_ANNOTATION_URL", "https://registry.example.com/annotations.md")
        monkeypatch.setenv("CONFIG_REGISTRY_USER", "deployer")
        monkeypatch.setenv("CONFIG_REGISTRY_TOKEN", "registry-secret")
        with patch("cloud.oci_vault.resolve_secret", return_value=""), \
             patch("cloud.config_fetcher.fetch_to_file", mock_fetch):
            import settings
            importlib.reload(settings)

        _, kwargs = mock_fetch.call_args
        assert kwargs["username"] == "deployer"
        assert kwargs["token"] == "registry-secret"

    def test_cf_url_no_credentials_passes_empty_strings(self, monkeypatch, settings_restored):
        mock_fetch = MagicMock()
        monkeypatch.setenv("JQL_ANNOTATION_URL", "https://registry.example.com/annotations.md")
        monkeypatch.delenv("CONFIG_REGISTRY_USER", raising=False)
        monkeypatch.delenv("CONFIG_REGISTRY_TOKEN", raising=False)
        with patch("cloud.oci_vault.resolve_secret", return_value=""), \
             patch("cloud.config_fetcher.fetch_to_file", mock_fetch):
            import settings
            importlib.reload(settings)

        _, kwargs = mock_fetch.call_args
        assert kwargs["username"] == ""
        assert kwargs["token"] == ""

    def test_cf_url_redirects_default_annotation_file_to_tmp(self, monkeypatch, settings_restored):
        mock_fetch = MagicMock()
        monkeypatch.setenv("JQL_ANNOTATION_URL", "https://registry.example.com/annotations.md")
        with patch("cloud.oci_vault.resolve_secret", return_value=""), \
             patch("cloud.config_fetcher.fetch_to_file", mock_fetch):
            import settings
            importlib.reload(settings)

        expected = str(Path(tempfile.gettempdir()) / "jira_jql_annotated_queries.md")
        assert settings.DEFAULT_ANNOTATION_FILE == expected

    def test_cf_url_does_not_use_data_dir_path(self, monkeypatch, settings_restored):
        mock_fetch = MagicMock()
        monkeypatch.setenv("JQL_ANNOTATION_URL", "https://registry.example.com/annotations.md")
        with patch("cloud.oci_vault.resolve_secret", return_value=""), \
             patch("cloud.config_fetcher.fetch_to_file", mock_fetch):
            import settings
            importlib.reload(settings)

        data_suffix = os.path.join("data", "jira_jql_annotated_queries.md")
        assert not settings.DEFAULT_ANNOTATION_FILE.endswith(data_suffix)

    # ------------------------------------------------------------------
    # failure: bad URL → startup raises immediately
    # ------------------------------------------------------------------

    def test_fetch_failure_raises_runtime_error_at_startup(self, monkeypatch, settings_restored):
        monkeypatch.setenv("JQL_ANNOTATION_URL", "https://registry.example.com/bad.md")
        with patch("cloud.oci_vault.resolve_secret", return_value=""), \
             patch("cloud.config_fetcher.fetch_to_file",
                   side_effect=RuntimeError("Config fetch failed [404]: ...")):
            import settings
            with pytest.raises(RuntimeError, match="Config fetch failed"):
                importlib.reload(settings)


# ===========================================================================
# 4. config/jira_config.py — _load_profiles_data + call sites
# ===========================================================================

SAMPLE_PROFILES = {
    "default": "work",
    "profiles": {
        "work": {
            "jira_url":    "https://issues.example.com/jira",
            "email":       "",
            "token":       "",
            "jira_type":   "server",
            "search_path": "",
        }
    },
}

CF_PROFILES = {
    "default": "cf",
    "profiles": {
        "cf": {
            "jira_url":    "https://cf-jira.example.com",
            "email":       "",
            "token":       "",
            "jira_type":   "server",
            "search_path": "",
        }
    },
}


class TestLoadProfilesData:
    """_load_profiles_data: env var takes precedence; falls back to file."""

    def test_default_docker_reads_from_file(self, tmp_path, monkeypatch):
        from config import jira_config

        f = tmp_path / "profiles.json"
        f.write_text(json.dumps(SAMPLE_PROFILES), encoding="utf-8")
        monkeypatch.delenv("JIRA_PROFILES_JSON", raising=False)
        monkeypatch.setattr(jira_config, "_PROFILES_FILE", f)

        assert jira_config._load_profiles_data() == SAMPLE_PROFILES

    def test_cf_env_var_overrides_file(self, tmp_path, monkeypatch):
        from config import jira_config

        f = tmp_path / "profiles.json"
        f.write_text(json.dumps(SAMPLE_PROFILES), encoding="utf-8")
        monkeypatch.setenv("JIRA_PROFILES_JSON", json.dumps(CF_PROFILES))
        monkeypatch.setattr(jira_config, "_PROFILES_FILE", f)

        result = jira_config._load_profiles_data()
        assert result["default"] == "cf"
        assert "cf" in result["profiles"]

    def test_cf_env_var_works_without_file(self, tmp_path, monkeypatch):
        from config import jira_config

        monkeypatch.setenv("JIRA_PROFILES_JSON", json.dumps(SAMPLE_PROFILES))
        monkeypatch.setattr(jira_config, "_PROFILES_FILE", tmp_path / "missing.json")

        result = jira_config._load_profiles_data()
        assert result["default"] == "work"

    def test_invalid_json_in_env_var_raises(self, monkeypatch):
        from config import jira_config

        monkeypatch.setenv("JIRA_PROFILES_JSON", "not-valid-json{{")
        with pytest.raises(json.JSONDecodeError):
            jira_config._load_profiles_data()


class TestLoadActiveProfile:
    """load_active_profile: dict with name + profile keys; env var or file source."""

    def test_default_docker_reads_correct_profile(self, tmp_path, monkeypatch):
        from config import jira_config

        f = tmp_path / "profiles.json"
        f.write_text(json.dumps(SAMPLE_PROFILES), encoding="utf-8")
        monkeypatch.delenv("JIRA_PROFILES_JSON", raising=False)
        monkeypatch.setattr(jira_config, "_PROFILES_FILE", f)

        profile = jira_config.load_active_profile()
        assert profile["jira_url"] == "https://issues.example.com/jira"
        assert profile["name"] == "work"

    def test_cf_env_var_returns_cf_profile(self, tmp_path, monkeypatch):
        from config import jira_config

        monkeypatch.setenv("JIRA_PROFILES_JSON", json.dumps(CF_PROFILES))
        monkeypatch.setattr(jira_config, "_PROFILES_FILE", tmp_path / "missing.json")

        profile = jira_config.load_active_profile()
        assert profile["jira_url"] == "https://cf-jira.example.com"
        assert profile["name"] == "cf"

    def test_missing_default_key_raises_key_error(self, tmp_path, monkeypatch):
        from config import jira_config

        bad = {"profiles": SAMPLE_PROFILES["profiles"]}
        f = tmp_path / "profiles.json"
        f.write_text(json.dumps(bad), encoding="utf-8")
        monkeypatch.delenv("JIRA_PROFILES_JSON", raising=False)
        monkeypatch.setattr(jira_config, "_PROFILES_FILE", f)

        with pytest.raises(KeyError):
            jira_config.load_active_profile()

    def test_unknown_default_profile_name_raises_key_error(self, tmp_path, monkeypatch):
        from config import jira_config

        bad = {"default": "nonexistent", "profiles": SAMPLE_PROFILES["profiles"]}
        f = tmp_path / "profiles.json"
        f.write_text(json.dumps(bad), encoding="utf-8")
        monkeypatch.delenv("JIRA_PROFILES_JSON", raising=False)
        monkeypatch.setattr(jira_config, "_PROFILES_FILE", f)

        with pytest.raises(KeyError):
            jira_config.load_active_profile()


class TestLoadActiveJiraProfile:
    """load_active_jira_profile: returns JiraProfile model; env var or file source."""

    def test_default_docker_returns_jira_profile_model(self, tmp_path, monkeypatch):
        from config import jira_config
        from core.jira_auth import JiraProfile

        f = tmp_path / "profiles.json"
        f.write_text(json.dumps(SAMPLE_PROFILES), encoding="utf-8")
        monkeypatch.delenv("JIRA_PROFILES_JSON", raising=False)
        monkeypatch.setattr(jira_config, "_PROFILES_FILE", f)

        profile = jira_config.load_active_jira_profile()
        assert isinstance(profile, JiraProfile)
        assert profile.jira_url == "https://issues.example.com/jira"
        assert profile.name == "work"

    def test_cf_env_var_returns_correct_jira_profile(self, tmp_path, monkeypatch):
        from config import jira_config
        from core.jira_auth import JiraProfile

        monkeypatch.setenv("JIRA_PROFILES_JSON", json.dumps(CF_PROFILES))
        monkeypatch.setattr(jira_config, "_PROFILES_FILE", tmp_path / "missing.json")

        profile = jira_config.load_active_jira_profile()
        assert isinstance(profile, JiraProfile)
        assert profile.jira_url == "https://cf-jira.example.com"
        assert profile.name == "cf"

    def test_cf_env_var_jira_type_is_preserved(self, tmp_path, monkeypatch):
        from config import jira_config

        monkeypatch.setenv("JIRA_PROFILES_JSON", json.dumps(CF_PROFILES))
        monkeypatch.setattr(jira_config, "_PROFILES_FILE", tmp_path / "missing.json")

        profile = jira_config.load_active_jira_profile()
        assert profile.jira_type == "server"
