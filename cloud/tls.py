"""Central TLS/certificate configuration for all outbound HTTPS clients.

CA bundle injection (runs once at import time via _init_ca_bundle):
  1. CA_BUNDLE_B64 env var — single base64-encoded PEM (local override, small certs)
  2. VCAP_SERVICES CredHub bindings — chunked CA bundle for CF deployments where the
     bundle is too large for a single env var; services named ca-bundle-01..N are
     reassembled in sorted order
  3. Neither set → system CAs (local dev, AWS, OCI)

All outbound HTTPS clients (httpx, requests, boto3) pick up the cert automatically
via REQUESTS_CA_BUNDLE / SSL_CERT_FILE / AWS_CA_BUNDLE once this module is imported.
"""

import os
import ssl
from pathlib import Path

import httpx


def _set_ca_env(path: str) -> None:
    os.environ.setdefault("REQUESTS_CA_BUNDLE", path)
    os.environ.setdefault("SSL_CERT_FILE",       path)
    os.environ.setdefault("AWS_CA_BUNDLE",        path)


def _init_ca_bundle() -> None:
    import base64

    # 1. Single cert via CA_BUNDLE_B64 env var
    b64 = os.getenv("CA_BUNDLE_B64", "")
    if b64:
        ca_path = Path("/tmp/ca-bundle.pem")
        ca_path.write_bytes(base64.b64decode(b64))
        _set_ca_env(str(ca_path))
        return

    # 2. Chunked cert via VCAP_SERVICES CredHub bindings (CF deployments)
    vcap_raw = os.getenv("VCAP_SERVICES", "")
    if not vcap_raw:
        return
    try:
        import json
        vcap = json.loads(vcap_raw)
        svcs = sorted(
            [s for s in vcap.get("credhub", [])
             if s.get("name", "").startswith("ca-bundle-")],
            key=lambda s: s["name"],
        )
        chunks = [
            base64.b64decode(s["credentials"]["ca_bundle_b64"])
            for s in svcs
            if s.get("credentials", {}).get("ca_bundle_b64")
        ]
        if chunks:
            ca_path = Path("/tmp/ca-bundle.pem")
            ca_path.write_bytes(b"".join(chunks))
            _set_ca_env(str(ca_path))
    except Exception:
        pass


_init_ca_bundle()


class TLSConfig:
    """Provides pre-wired HTTP clients using the CA bundle set at import time.

    Works with any cert injection method:
      - CA_BUNDLE_B64 / VCAP_SERVICES decoded by _init_ca_bundle() above
      - REQUESTS_CA_BUNDLE set externally (K8s volume mount, CI, local corp proxy)
      - SSL_CERT_FILE fallback (Python ssl module default)
      - Neither set → True (system CAs, local dev / AWS / OCI)
    """

    @property
    def ca_bundle_path(self) -> "str | None":
        """Path to the active CA bundle file, or None when using system CAs."""
        return os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")

    @property
    def verify(self) -> "str | bool":
        """CA bundle path for httpx/requests verify=, or True for system defaults."""
        return self.ca_bundle_path or True

    def ssl_context(self) -> ssl.SSLContext:
        """ssl.SSLContext for urllib/stdlib use (config_fetcher)."""
        ctx = ssl.create_default_context()
        ca = os.environ.get("REQUESTS_CA_BUNDLE")
        if ca:
            ctx.load_verify_locations(ca)
        return ctx

    def httpx_client(self, **kwargs) -> httpx.AsyncClient:
        """Return httpx.AsyncClient with CA bundle pre-configured.

        Pass any httpx.AsyncClient kwargs (timeout, follow_redirects, etc.).
        verify= is set from the environment unless the caller overrides it.
        """
        kwargs.setdefault("verify", self.verify)
        return httpx.AsyncClient(**kwargs)


tls = TLSConfig()
