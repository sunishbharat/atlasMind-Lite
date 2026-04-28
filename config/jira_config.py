import json
from pathlib import Path
from urllib.parse import urlparse
from settings import DATA_DIR
from core.jira_auth import JiraProfile

_PROFILES_FILE = Path(__file__).parent / "profiles.json"

# System fields that have their own REST endpoints
_SYSTEM_FIELD_ENDPOINTS: dict[str, str] = {
    "status":      "/rest/api/2/status",
    "priority":    "/rest/api/2/priority",
    "resolution":  "/rest/api/2/resolution",
    "issuetype":   "/rest/api/2/issuetype",
}


def get_data_dir(jira_url: str) -> Path:
    """Derive a domain-scoped data directory from a Jira base URL.

    Converts the hostname to a filesystem-safe slug by replacing dots and
    hyphens with underscores. Each Jira instance stores its files in its
    own subdirectory under data/.

    Examples:
        https://issues.apache.org/jira  →  data/issues_apache_org
        https://myorg.atlassian.net     →  data/myorg_atlassian_net

    Args:
        jira_url: The Jira instance base URL from the active profile.

    Returns:
        Path: e.g. Path("data/issues_apache_org")
    """
    hostname = urlparse(jira_url).hostname or "unknown"
    slug = hostname.replace(".", "_").replace("-", "_")
    return DATA_DIR / slug


def build_jira_auth(profile: dict) -> tuple:
    """Return (httpx_auth, extra_headers) for a Jira profile.

    Jira Cloud uses Basic auth (email:api_token).
    Jira Server uses Bearer auth (PAT) — Basic auth is not accepted for PATs.

    Returns:
        (auth_tuple_or_None, headers_dict)
    """
    jira_type = profile.get("jira_type", "cloud")
    email = profile.get("email", "")
    token = profile.get("token", "")

    if jira_type == "server" and token:
        return None, {"Authorization": f"Bearer {token}"}
    if email and token:
        return (email, token), {}
    return None, {}


def load_active_profile() -> dict:
    """Load the active Jira profile from config/profiles.json.

    Returns:
        dict with keys: jira_url, email, token, jira_type (and optionally
        client_id, client_secret).

    Raises:
        FileNotFoundError: If profiles.json does not exist.
        KeyError: If the default profile name is not found in profiles.
    """
    data = json.loads(_PROFILES_FILE.read_text(encoding="utf-8"))
    default = data["default"]
    profile = data["profiles"][default]
    return {"name": default, **profile}


def load_active_jira_profile() -> JiraProfile:
    """Load and validate the active Jira profile as a JiraProfile model.

    Returns:
        JiraProfile with validated fields and resolve_auth() available.

    Raises:
        FileNotFoundError: If profiles.json does not exist.
        KeyError: If the default profile name is not found in profiles.
        ValidationError: If the profile fields fail Pydantic validation.
    """
    data = json.loads(_PROFILES_FILE.read_text(encoding="utf-8"))
    default = data["default"]
    profile = data["profiles"][default]
    return JiraProfile(name=default, **profile)
