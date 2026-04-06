import json
from pathlib import Path
from urllib.parse import urlparse
from settings import DATA_DIR

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
