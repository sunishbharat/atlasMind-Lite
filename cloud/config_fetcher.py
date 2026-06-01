import base64
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def fetch_to_file(url: str, dest: Path, username: str = "", token: str = "") -> None:
    """Fetch a URL to a local file with optional Basic auth.

    Raises RuntimeError on failure — fail fast so CF startup surfaces the
    problem immediately rather than silently using stale or missing data.
    """
    req = Request(url)
    if username or token:
        creds = base64.b64encode(f"{username}:{token}".encode()).decode()
        req.add_header("Authorization", f"Basic {creds}")
    try:
        with urlopen(req, timeout=30) as resp:
            dest.write_bytes(resp.read())
    except HTTPError as e:
        raise RuntimeError(f"Config fetch failed [{e.code}]: {url}") from e
    except URLError as e:
        raise RuntimeError(f"Config fetch unreachable: {url} — {e.reason}") from e
