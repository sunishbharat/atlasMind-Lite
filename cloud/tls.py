"""Central TLS/certificate configuration for all outbound HTTPS clients."""

import os
import ssl

import httpx


class TLSConfig:
    """Reads cert config from environment and provides pre-wired HTTP clients.

    Works with any cert injection method:
      - REQUESTS_CA_BUNDLE set by settings.py when CA_BUNDLE_B64 is decoded (CF/PCF/Docker)
      - REQUESTS_CA_BUNDLE set externally (K8s volume mount, CI, local corp proxy)
      - SSL_CERT_FILE fallback (Python ssl module default)
      - Neither set → True (system CAs, local dev / AWS / OCI)
    """

    @property
    def verify(self) -> "str | bool":
        """CA bundle path for httpx verify=, or True for system defaults."""
        return (
            os.environ.get("REQUESTS_CA_BUNDLE")
            or os.environ.get("SSL_CERT_FILE")
            or True
        )

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
