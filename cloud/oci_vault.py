"""
OCI Vault secret retrieval.

Provides fetch_secret() and resolve_secret() for loading secrets from
Oracle Cloud Infrastructure Vault at runtime.

Authentication order:
  1. Instance Principal — used when running on an OCI compute instance.
     No credentials on disk; the instance identity is used automatically.
  2. File-based OCI config (~/.oci/config) — fallback for local development.

To switch cloud providers, replace this module and update the imports in
settings.py (resolve_secret calls) without touching any other application code.
"""

import base64
import logging
import os

_log = logging.getLogger(__name__)


def fetch_secret(ocid: str) -> str:
    """Fetch a secret value from OCI Vault by its OCID.

    Args:
        ocid: The full OCID of the secret, e.g.
              ocid1.vaultsecret.oc1.ap-sydney-1.<unique_id>

    Returns:
        The secret value as a plain string.

    Raises:
        RuntimeError: If the oci package is missing, auth fails, or the
                      secret cannot be retrieved.
    """
    try:
        import oci
    except ImportError as e:
        raise RuntimeError("oci package is not installed — run: uv add oci") from e

    client = _get_client(oci)

    try:
        bundle = client.get_secret_bundle(ocid).data
        return base64.b64decode(bundle.secret_bundle_content.content).decode().strip()
    except Exception as e:
        raise RuntimeError(f"OCI Vault: failed to fetch secret {ocid!r} — {e}") from e


def resolve_secret(ocid_env: str, value_env: str, default: str = "") -> str:
    """Resolve a secret from OCI Vault or fall back to a plain env var.

    If the env var named by ocid_env is set, the secret is fetched from
    OCI Vault using that OCID. Otherwise the env var named by value_env
    is returned (useful for local development without Vault access).

    Args:
        ocid_env:  Name of the env var holding the OCI Vault secret OCID.
                   Example: "GROQ_API_KEY_OCID"
        value_env: Name of the env var holding the plaintext fallback value.
                   Example: "GROQ_API_KEY"
        default:   Value to return if neither env var is set.

    Returns:
        The resolved secret string.
    """
    ocid = os.getenv(ocid_env)
    if ocid:
        _log.info("OCI Vault: resolving %s from Vault (OCID env: %s)", value_env, ocid_env)
        return fetch_secret(ocid)
    return os.getenv(value_env, default)


def _get_client(oci):
    """Return a SecretsClient authenticated via Instance Principal or file config."""
    try:
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        _log.info("OCI Vault: authenticated via Instance Principal")
        return oci.secrets.SecretsClient(config={}, signer=signer)
    except Exception:
        _log.info("OCI Vault: Instance Principal unavailable, falling back to ~/.oci/config")
        return oci.secrets.SecretsClient(config=oci.config.from_file())
