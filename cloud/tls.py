"""Central TLS/certificate configuration for all outbound HTTPS clients.

CA bundle injection (runs once at import time via _init_ca_bundle):
  1. CA_BUNDLE_B64 env var — single base64-encoded PEM (local override, small certs)
  2. VCAP_SERVICES CredHub bindings — chunked CA bundle for CF deployments where the
     bundle is too large for a single env var; any CredHub service whose credentials
     contain a `ca-certificates` key is treated as a chunk; services are sorted by name
     before concatenation, so names must sort in the intended chunk order
  3. Neither set → system CAs (local dev, AWS, OCI)

All outbound HTTPS clients (httpx, requests, boto3) pick up the cert automatically
via REQUESTS_CA_BUNDLE / SSL_CERT_FILE / AWS_CA_BUNDLE once this module is imported.
"""

import logging
import os
import ssl
from pathlib import Path

import httpx

_log = logging.getLogger(__name__)

# Paths where decoded client cert/key are written at startup
_CLIENT_CERT_PATH = Path("/tmp/client-cert.pem")
_CLIENT_KEY_PATH  = Path("/tmp/client-key.pem")


def _set_ca_env(path: str) -> None:
    os.environ.setdefault("REQUESTS_CA_BUNDLE", path)
    os.environ.setdefault("SSL_CERT_FILE",       path)
    os.environ.setdefault("AWS_CA_BUNDLE",        path)


def _init_client_cert() -> None:
    """Decode CLIENT_CERT_B64 + CLIENT_KEY_B64 env vars to /tmp for mTLS."""
    import base64
    cert_b64 = os.getenv("CLIENT_CERT_B64", "")
    key_b64  = os.getenv("CLIENT_KEY_B64",  "")
    if not cert_b64 or not key_b64:
        _log.debug("TLS: no CLIENT_CERT_B64/CLIENT_KEY_B64 — mTLS client cert not configured")
        return
    try:
        _CLIENT_CERT_PATH.write_bytes(base64.b64decode(cert_b64))
        _CLIENT_KEY_PATH.write_bytes(base64.b64decode(key_b64))
        _log.info("TLS: client cert written to %s", _CLIENT_CERT_PATH)
    except Exception:
        _log.exception("TLS: failed to decode client cert/key")


def _init_ca_bundle() -> None:
    import base64

    # 0. REQUESTS_CA_BUNDLE already set externally (volume mount, docker-compose, K8s)
    if os.environ.get("REQUESTS_CA_BUNDLE"):
        _log.info("TLS: using pre-set REQUESTS_CA_BUNDLE=%s", os.environ["REQUESTS_CA_BUNDLE"])
        return

    # 1. Single cert via CA_BUNDLE_B64 env var
    b64 = os.getenv("CA_BUNDLE_B64", "")
    if b64:
        ca_path = Path("/tmp/ca-bundle.pem")
        ca_path.write_bytes(base64.b64decode(b64))
        _set_ca_env(str(ca_path))
        _log.info("TLS: CA bundle loaded from CA_BUNDLE_B64 (%d bytes)", len(ca_path.read_bytes()))
        return

    # 2. Chunked cert via VCAP_SERVICES CredHub bindings (CF deployments)
    vcap_raw = os.getenv("VCAP_SERVICES", "")
    if not vcap_raw:
        _log.debug("TLS: no CA_BUNDLE_B64 and no VCAP_SERVICES — using system CAs")
        return
    try:
        import json
        vcap = json.loads(vcap_raw)
        all_credhub = vcap.get("credhub", [])
        svcs = sorted(
            [s for s in all_credhub
             if s.get("credentials", {}).get("ca-certificates")],
            key=lambda s: s["name"],
        )
        _log.info(
            "TLS: VCAP_SERVICES has %d CredHub service(s), %d have ca-certificates: %s",
            len(all_credhub),
            len(svcs),
            [s["name"] for s in svcs],
        )
        chunks = [
            s["credentials"]["ca-certificates"].encode()
            for s in svcs
        ]
        if chunks:
            ca_path = Path("/tmp/ca-bundle.pem")
            bundle = b"".join(chunks)
            ca_path.write_bytes(bundle)
            _set_ca_env(str(ca_path))
            _log.info("TLS: CA bundle written to %s (%d bytes from %d chunk(s))",
                      ca_path, len(bundle), len(chunks))
        else:
            _log.warning("TLS: VCAP_SERVICES present but no ca-certificates found — using system CAs")
    except Exception:
        _log.exception("TLS: failed to assemble CA bundle from VCAP_SERVICES")


_init_ca_bundle()
_init_client_cert()


class TLSConfig:
    """Provides pre-wired HTTP clients using the CA bundle set at import time.

    Works with any cert injection method:
      - CA_BUNDLE_B64 / VCAP_SERVICES decoded by _init_ca_bundle() above
      - REQUESTS_CA_BUNDLE set externally (K8s volume mount, CI, local corp proxy)
      - SSL_CERT_FILE fallback (Python ssl module default)
      - Neither set → True (system CAs, local dev / AWS / OCI)

    mTLS client cert:
      - CLIENT_CERT_B64 + CLIENT_KEY_B64 decoded by _init_client_cert() above
      - client_cert returns (cert_path, key_path) tuple, or None if not configured
    """

    @property
    def ca_bundle_path(self) -> "str | None":
        """Path to the active CA bundle file, or None when using system CAs."""
        return os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")

    @property
    def verify(self) -> "str | bool":
        """CA bundle path for httpx/requests verify=, or True for system defaults."""
        return self.ca_bundle_path or True

    @property
    def client_cert(self) -> "tuple[str, str] | None":
        """(cert_path, key_path) for mTLS, or None if not configured."""
        if _CLIENT_CERT_PATH.exists() and _CLIENT_KEY_PATH.exists():
            return (str(_CLIENT_CERT_PATH), str(_CLIENT_KEY_PATH))
        return None

    def ssl_context(self) -> ssl.SSLContext:
        """ssl.SSLContext for urllib/stdlib use (config_fetcher)."""
        ctx = ssl.create_default_context()
        ca = os.environ.get("REQUESTS_CA_BUNDLE")
        if ca:
            ctx.load_verify_locations(ca)
        return ctx

    def httpx_client(self, **kwargs) -> httpx.AsyncClient:
        """Return httpx.AsyncClient with CA bundle and mTLS cert pre-configured."""
        kwargs.setdefault("verify", self.verify)
        if self.client_cert:
            kwargs.setdefault("cert", self.client_cert)
        return httpx.AsyncClient(**kwargs)


tls = TLSConfig()
