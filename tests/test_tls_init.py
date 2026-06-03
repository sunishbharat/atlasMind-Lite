"""
tests/test_tls_init.py

Tests for cloud/tls._init_ca_bundle() — four CA bundle injection scenarios:
  1. No cert  — neither CA_BUNDLE_B64 nor VCAP_SERVICES set
  2. Small cert — single PEM via CA_BUNDLE_B64 env var
  3. Split cert = 8  — VCAP_SERVICES with exactly 8 CredHub services
  4. Split cert > 10 — VCAP_SERVICES with more than 10 CredHub services
"""

import base64
import json
import os
from pathlib import Path

import pytest

import cloud.tls
from cloud.tls import _init_ca_bundle, TLSConfig


_ENV_KEYS = (
    "CA_BUNDLE_B64",
    "VCAP_SERVICES",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "AWS_CA_BUNDLE",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vcap(n: int, *, shuffle: bool = False) -> str:
    """Build a VCAP_SERVICES JSON string with n CredHub services.

    Each service carries a `ca-certificates` key with raw PEM text (the
    industry-standard CF CredHub format — no base64 encoding needed for PEM).
    When shuffle=True the services list is in reversed order to verify that
    the code sorts by name before concatenating.
    """
    indices = list(range(1, n + 1))
    if shuffle:
        indices = list(reversed(indices))
    services = [
        {
            "name": f"ca-bundle-{i:02d}",
            "credentials": {
                "ca-certificates": f"chunk-{i:02d}-content",
            },
        }
        for i in indices
    ]
    return json.dumps({"credhub": services})


def _expected(n: int) -> bytes:
    """Expected reassembled bytes when n sequential chunks are concatenated."""
    return b"".join(f"chunk-{i:02d}-content".encode() for i in range(1, n + 1))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def tls_env(tmp_path, monkeypatch):
    """Per-test setup: clear all TLS env vars and redirect the hardcoded
    /tmp/ca-bundle.pem to a writable temp directory.

    Returns the test-local Path so assertions can inspect whether the file
    was written and what its contents are.
    """
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    test_ca = tmp_path / "ca-bundle.pem"
    _real_path = Path

    def _intercepted(s):
        return test_ca if str(s) == "/tmp/ca-bundle.pem" else _real_path(s)

    monkeypatch.setattr(cloud.tls, "Path", _intercepted)
    return test_ca


# ===========================================================================
# Scenario 1 — No cert
# ===========================================================================

class TestNoCert:
    """Neither CA_BUNDLE_B64 nor VCAP_SERVICES — system CAs used; nothing written."""

    def test_no_env_vars_set(self):
        _init_ca_bundle()

        assert os.getenv("REQUESTS_CA_BUNDLE") is None
        assert os.getenv("SSL_CERT_FILE") is None
        assert os.getenv("AWS_CA_BUNDLE") is None

    def test_no_cert_file_written(self, tls_env):
        _init_ca_bundle()

        assert not tls_env.exists()

    def test_tls_verify_returns_true(self):
        _init_ca_bundle()

        assert TLSConfig().verify is True

    def test_tls_ca_bundle_path_is_none(self):
        _init_ca_bundle()

        assert TLSConfig().ca_bundle_path is None


# ===========================================================================
# Scenario 2 — Small cert via CA_BUNDLE_B64
# ===========================================================================

class TestSmallCert:
    """Single PEM supplied as a base64 env var."""

    _PEM = b"-----BEGIN CERTIFICATE-----\nMIIBsmall\n-----END CERTIFICATE-----\n"

    def test_decoded_pem_written_to_disk(self, tls_env, monkeypatch):
        monkeypatch.setenv("CA_BUNDLE_B64", base64.b64encode(self._PEM).decode())

        _init_ca_bundle()

        assert tls_env.read_bytes() == self._PEM

    def test_all_three_env_vars_set_to_bundle_path(self, tls_env, monkeypatch):
        monkeypatch.setenv("CA_BUNDLE_B64", base64.b64encode(b"cert-data").decode())

        _init_ca_bundle()

        expected = str(tls_env)
        assert os.environ["REQUESTS_CA_BUNDLE"] == expected
        assert os.environ["SSL_CERT_FILE"]       == expected
        assert os.environ["AWS_CA_BUNDLE"]        == expected

    def test_tls_verify_returns_path_string(self, tls_env, monkeypatch):
        monkeypatch.setenv("CA_BUNDLE_B64", base64.b64encode(b"cert-data").decode())

        _init_ca_bundle()

        assert TLSConfig().verify == str(tls_env)

    def test_takes_priority_over_vcap_services(self, tls_env, monkeypatch):
        """When both are set, CA_BUNDLE_B64 wins; VCAP chunks must not appear."""
        monkeypatch.setenv("CA_BUNDLE_B64", base64.b64encode(self._PEM).decode())
        monkeypatch.setenv("VCAP_SERVICES", _make_vcap(3))

        _init_ca_bundle()

        assert tls_env.read_bytes() == self._PEM

    def test_setdefault_does_not_overwrite_existing_env(self, tls_env, monkeypatch):
        """A pre-set REQUESTS_CA_BUNDLE must survive the init call unchanged."""
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/pre-existing/bundle.pem")
        monkeypatch.setenv("CA_BUNDLE_B64", base64.b64encode(b"cert-data").decode())

        _init_ca_bundle()

        assert os.environ["REQUESTS_CA_BUNDLE"] == "/pre-existing/bundle.pem"


# ===========================================================================
# Scenario 3 — Split cert: exactly 8 CredHub services
# ===========================================================================

class TestVcap8Services:
    """CA bundle split across exactly 8 CredHub services in VCAP_SERVICES."""

    def test_all_8_chunks_concatenated(self, tls_env, monkeypatch):
        monkeypatch.setenv("VCAP_SERVICES", _make_vcap(8))

        _init_ca_bundle()

        assert tls_env.read_bytes() == _expected(8)

    def test_all_three_env_vars_set(self, tls_env, monkeypatch):
        monkeypatch.setenv("VCAP_SERVICES", _make_vcap(8))

        _init_ca_bundle()

        expected = str(tls_env)
        assert os.environ["REQUESTS_CA_BUNDLE"] == expected
        assert os.environ["SSL_CERT_FILE"]       == expected
        assert os.environ["AWS_CA_BUNDLE"]        == expected

    def test_chunks_sorted_by_service_name_regardless_of_input_order(self, tls_env, monkeypatch):
        """Services shuffled in the JSON must still be assembled in name order."""
        monkeypatch.setenv("VCAP_SERVICES", _make_vcap(8, shuffle=True))

        _init_ca_bundle()

        assert tls_env.read_bytes() == _expected(8)

    def test_services_without_ca_bundle_b64_credential_excluded(self, tls_env, monkeypatch):
        """CredHub services with no ca_bundle_b64 credential must not contribute bytes.
        Exclusion is by credential key, not by service name."""
        extra = [
            {"name": "atlasmind-secrets", "credentials": {"GROQ_API_KEY": "secret"}},
            {"name": "db-service",        "credentials": {"url": "postgresql://..."}},
        ]
        vcap = json.loads(_make_vcap(8))
        vcap["credhub"].extend(extra)
        monkeypatch.setenv("VCAP_SERVICES", json.dumps(vcap))

        _init_ca_bundle()

        assert tls_env.read_bytes() == _expected(8)

    def test_any_service_name_with_credential_is_included(self, tls_env, monkeypatch):
        """A service NOT named ca-bundle-* but carrying ca-certificates must be included."""
        services = [
            {
                "name": f"corp-cert-{i:02d}",
                "credentials": {"ca-certificates": f"chunk-{i:02d}-content"},
            }
            for i in range(1, 9)
        ]
        monkeypatch.setenv("VCAP_SERVICES", json.dumps({"credhub": services}))

        _init_ca_bundle()

        assert tls_env.read_bytes() == _expected(8)

    def test_service_missing_credential_key_is_skipped(self, tls_env, monkeypatch):
        """A service bound without ca-certificates must be silently skipped."""
        services = [
            {"name": "ca-bundle-01", "credentials": {}},  # missing key
            {"name": "ca-bundle-02", "credentials": {"ca-certificates": "chunk-02-content"}},
        ]
        monkeypatch.setenv("VCAP_SERVICES", json.dumps({"credhub": services}))

        _init_ca_bundle()

        assert tls_env.read_bytes() == b"chunk-02-content"

    def test_tls_verify_returns_path_string(self, tls_env, monkeypatch):
        monkeypatch.setenv("VCAP_SERVICES", _make_vcap(8))

        _init_ca_bundle()

        assert TLSConfig().verify == str(tls_env)


# ===========================================================================
# Scenario 4 — Split cert: more than 10 CredHub services
# ===========================================================================

class TestVcapMoreThan10Services:
    """No hardcoded service limit — any number of ca-bundle-* services reassembled."""

    def test_12_services_all_chunks_included(self, tls_env, monkeypatch):
        monkeypatch.setenv("VCAP_SERVICES", _make_vcap(12))

        _init_ca_bundle()

        assert tls_env.read_bytes() == _expected(12)

    def test_12_services_all_env_vars_set(self, tls_env, monkeypatch):
        monkeypatch.setenv("VCAP_SERVICES", _make_vcap(12))

        _init_ca_bundle()

        expected = str(tls_env)
        assert os.environ["REQUESTS_CA_BUNDLE"] == expected
        assert os.environ["SSL_CERT_FILE"]       == expected
        assert os.environ["AWS_CA_BUNDLE"]        == expected

    def test_12_services_zero_padded_names_sort_correctly(self, tls_env, monkeypatch):
        """ca-bundle-09 must sort before ca-bundle-10 (string sort on zero-padded names)."""
        monkeypatch.setenv("VCAP_SERVICES", _make_vcap(12, shuffle=True))

        _init_ca_bundle()

        assert tls_env.read_bytes() == _expected(12)

    def test_15_services_no_hardcoded_limit(self, tls_env, monkeypatch):
        """Verify behaviour scales beyond 8 — 15 services must all be reassembled."""
        monkeypatch.setenv("VCAP_SERVICES", _make_vcap(15))

        _init_ca_bundle()

        assert tls_env.read_bytes() == _expected(15)

    def test_tls_verify_returns_path_string(self, tls_env, monkeypatch):
        monkeypatch.setenv("VCAP_SERVICES", _make_vcap(12))

        _init_ca_bundle()

        assert TLSConfig().verify == str(tls_env)


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_malformed_vcap_json_does_not_raise(self, monkeypatch):
        """Corrupt VCAP_SERVICES JSON must be silently swallowed."""
        monkeypatch.setenv("VCAP_SERVICES", "not-valid-json{")

        _init_ca_bundle()  # must not raise

        assert os.getenv("REQUESTS_CA_BUNDLE") is None

    def test_vcap_with_no_ca_bundle_services_writes_nothing(self, tls_env, monkeypatch):
        """VCAP_SERVICES present but with no ca-bundle-* names must write no file."""
        monkeypatch.setenv("VCAP_SERVICES", json.dumps({"credhub": [
            {"name": "other-service", "credentials": {"key": "value"}},
        ]}))

        _init_ca_bundle()

        assert not tls_env.exists()
        assert os.getenv("REQUESTS_CA_BUNDLE") is None

    def test_empty_credhub_list_writes_nothing(self, tls_env, monkeypatch):
        monkeypatch.setenv("VCAP_SERVICES", json.dumps({"credhub": []}))

        _init_ca_bundle()

        assert not tls_env.exists()

    def test_vcap_missing_credhub_key_writes_nothing(self, tls_env, monkeypatch):
        """VCAP_SERVICES with only non-credhub service types must write no file."""
        monkeypatch.setenv("VCAP_SERVICES", json.dumps({"user-provided": []}))

        _init_ca_bundle()

        assert not tls_env.exists()
