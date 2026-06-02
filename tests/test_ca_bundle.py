"""
tests/test_ca_bundle.py

Verifies CA_BUNDLE_B64 certificate injection in settings.py and
the SSL context wiring in cloud/config_fetcher.py.
"""

import base64
import importlib
import os
import ssl
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixture: clean up env vars and reload settings after each test
# ---------------------------------------------------------------------------

@pytest.fixture
def settings_restored():
    yield
    for key in ("CA_BUNDLE_B64", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "AWS_CA_BUNDLE"):
        os.environ.pop(key, None)
    with patch("cloud.oci_vault.resolve_secret", return_value=""):
        import settings
        importlib.reload(settings)


# ===========================================================================
# 1. settings.py — CA_BUNDLE_B64 handling
# ===========================================================================

class TestCaBundleSettings:

    def test_no_env_var_sets_no_paths(self, settings_restored):
        """When CA_BUNDLE_B64 is absent, REQUESTS_CA_BUNDLE must not be set by settings."""
        os.environ.pop("CA_BUNDLE_B64", None)
        # Temporarily clear any ambient value (e.g. set by miniconda/conda)
        saved = {k: os.environ.pop(k, None) for k in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")}
        try:
            with patch("cloud.oci_vault.resolve_secret", return_value=""):
                import settings
                importlib.reload(settings)
            assert os.getenv("REQUESTS_CA_BUNDLE") is None
            assert os.getenv("SSL_CERT_FILE") is None
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_valid_b64_writes_pem_and_sets_env(self, settings_restored, tmp_path):
        """Valid CA_BUNDLE_B64 must decode and set REQUESTS_CA_BUNDLE + SSL_CERT_FILE."""
        encoded = base64.b64encode(b"placeholder-cert-content").decode()
        os.environ["CA_BUNDLE_B64"] = encoded
        os.environ.pop("REQUESTS_CA_BUNDLE", None)
        os.environ.pop("SSL_CERT_FILE", None)

        with patch("cloud.oci_vault.resolve_secret", return_value=""), \
             patch("builtins.open", MagicMock()), \
             patch("settings.Path") as MockPath:
            mock_file = MagicMock()
            MockPath.return_value = mock_file
            MockPath.side_effect = lambda *a, **k: Path(*a, **k)
            import settings
            importlib.reload(settings)

        assert os.getenv("REQUESTS_CA_BUNDLE") is not None
        assert os.getenv("SSL_CERT_FILE") is not None

    def test_setdefault_does_not_override_existing(self, settings_restored, tmp_path):
        """Pre-set REQUESTS_CA_BUNDLE must not be overwritten by settings reload."""
        encoded = base64.b64encode(b"placeholder-cert-content").decode()
        os.environ["CA_BUNDLE_B64"] = encoded
        os.environ["REQUESTS_CA_BUNDLE"] = "/custom/path/bundle.pem"

        with patch("cloud.oci_vault.resolve_secret", return_value=""):
            import settings
            importlib.reload(settings)

        assert os.environ["REQUESTS_CA_BUNDLE"] == "/custom/path/bundle.pem"


# ===========================================================================
# 2. cloud/config_fetcher.py — SSL context wiring
# ===========================================================================

class TestFetchToFileSSL:

    @staticmethod
    def _mock_resp(body: bytes = b"data"):
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.read.return_value = body
        return resp

    def test_uses_ssl_context(self, tmp_path):
        """fetch_to_file must pass an SSLContext to urlopen."""
        from cloud.config_fetcher import fetch_to_file

        dest = tmp_path / "out.txt"
        captured: dict = {}

        def fake_urlopen(req, timeout, context=None):
            captured["context"] = context
            return self._mock_resp()

        with patch("cloud.config_fetcher.urlopen", side_effect=fake_urlopen):
            fetch_to_file("https://registry.example.com/file.txt", dest)

        assert isinstance(captured.get("context"), ssl.SSLContext)

    def test_loads_ca_bundle_when_env_set(self, tmp_path):
        """When REQUESTS_CA_BUNDLE is set, load_verify_locations must be called with its path."""
        from cloud.config_fetcher import fetch_to_file

        ca_file = tmp_path / "ca-bundle.pem"
        ca_file.write_bytes(b"placeholder")  # content irrelevant — SSL context is mocked
        dest = tmp_path / "out.txt"

        os.environ["REQUESTS_CA_BUNDLE"] = str(ca_file)
        mock_ctx = MagicMock(spec=ssl.SSLContext)
        captured: dict = {}

        def fake_urlopen(req, timeout, context=None):
            captured["context"] = context
            return self._mock_resp()

        try:
            with patch("cloud.tls.ssl.create_default_context", return_value=mock_ctx), \
                 patch("cloud.config_fetcher.urlopen", side_effect=fake_urlopen):
                fetch_to_file("https://registry.example.com/file.txt", dest)
        finally:
            os.environ.pop("REQUESTS_CA_BUNDLE", None)

        mock_ctx.load_verify_locations.assert_called_once_with(str(ca_file))
        assert captured["context"] is mock_ctx

    def test_no_ca_bundle_env_still_uses_default_context(self, tmp_path):
        """Without REQUESTS_CA_BUNDLE, fetch_to_file must still pass a default SSLContext."""
        from cloud.config_fetcher import fetch_to_file

        os.environ.pop("REQUESTS_CA_BUNDLE", None)
        dest = tmp_path / "out.txt"
        captured: dict = {}

        def fake_urlopen(req, timeout, context=None):
            captured["context"] = context
            return self._mock_resp()

        with patch("cloud.config_fetcher.urlopen", side_effect=fake_urlopen):
            fetch_to_file("https://registry.example.com/file.txt", dest)

        assert isinstance(captured.get("context"), ssl.SSLContext)
