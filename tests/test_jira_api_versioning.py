"""
tests/test_jira_api_versioning.py

Verifies that the correct Jira REST API version is selected based on jira_type:
  - Cloud  → /rest/api/3/
  - Server → /rest/api/2/
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from config.jira_config import get_field_api_base
from jira.jira_field_api import _resolve_endpoint


class TestGetFieldApiBase:
    def test_cloud_returns_v3(self):
        assert get_field_api_base("cloud") == "/rest/api/3"

    def test_server_returns_v2(self):
        assert get_field_api_base("server") == "/rest/api/2"

    def test_unknown_type_defaults_to_v2(self):
        assert get_field_api_base("unknown") == "/rest/api/2"


class TestResolveEndpoint:
    def test_custom_option_field_cloud_uses_v3(self):
        result = _resolve_endpoint("customfield_10001", "option", "cloud")
        assert result == "/rest/api/3/field/customfield_10001/option"

    def test_custom_option_field_server_uses_v2(self):
        result = _resolve_endpoint("customfield_10001", "option", "server")
        assert result == "/rest/api/2/field/customfield_10001/option"

    def test_system_field_cloud_uses_v3(self):
        assert _resolve_endpoint("status", "status", "cloud") == "/rest/api/3/status"

    def test_system_field_server_uses_v2(self):
        assert _resolve_endpoint("status", "status", "server") == "/rest/api/2/status"

    def test_unsupported_field_type_returns_none(self):
        assert _resolve_endpoint("customfield_10001", "string", "cloud") is None


class TestFetchAndSaveFieldsApiVersion:
    """fetch_and_save_fields must call the correct /field endpoint for each jira_type."""

    def _run(self, jira_type: str, tmp_path):
        profile = {
            "name": "test",
            "jira_url": "https://sample-domain.atlassian.net",
            "email": "user@example.com",
            "token": "token123",
            "jira_type": jira_type,
        }
        mock_response = MagicMock()
        mock_response.json.return_value = [{"id": "status", "name": "Status"}]
        mock_response.raise_for_status = MagicMock()

        output = tmp_path / "jira_fields.json"
        with patch("jira.jira_field_api.load_active_profile", return_value=profile), \
             patch("jira.jira_field_api.build_jira_auth", return_value=(None, {})), \
             patch("jira.jira_field_api.get_data_dir", return_value=tmp_path), \
             patch("jira.jira_field_api.requests.get", return_value=mock_response) as mock_get:
            from jira.jira_field_api import fetch_and_save_fields
            fetch_and_save_fields(output_path=output)
            return mock_get.call_args[0][0]  # first positional arg = URL

    def test_cloud_calls_v3_field_endpoint(self, tmp_path):
        url = self._run("cloud", tmp_path)
        assert "/rest/api/3/field" in url
        assert "/rest/api/2/" not in url

    def test_server_calls_v2_field_endpoint(self, tmp_path):
        url = self._run("server", tmp_path)
        assert "/rest/api/2/field" in url
        assert "/rest/api/3/" not in url


class TestFetchAllVersionNamesApiVersion:
    """_fetch_all_version_names must use the correct /project endpoint per jira_type."""

    @pytest.mark.asyncio
    async def test_cloud_calls_v3_project_endpoint(self):
        import httpx
        from jira.jira_field_api import _fetch_all_version_names
        captured = []

        async def mock_get(url, **kwargs):
            captured.append(url)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = []
            return mock_resp

        with patch("cloud.tls.httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = mock_get
            mock_client_cls.return_value = mock_client

            await _fetch_all_version_names(
                "https://sample-domain.atlassian.net", None, {}, jira_type="cloud"
            )

        assert any("/rest/api/3/project" in u for u in captured)
        assert not any("/rest/api/2/" in u for u in captured)

    @pytest.mark.asyncio
    async def test_server_calls_v2_project_endpoint(self):
        from jira.jira_field_api import _fetch_all_version_names
        captured = []

        async def mock_get(url, **kwargs):
            captured.append(url)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = []
            return mock_resp

        with patch("cloud.tls.httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = mock_get
            mock_client_cls.return_value = mock_client

            await _fetch_all_version_names(
                "https://issues.apache.org/jira", None, {}, jira_type="server"
            )

        assert any("/rest/api/2/project" in u for u in captured)
        assert not any("/rest/api/3/" in u for u in captured)
