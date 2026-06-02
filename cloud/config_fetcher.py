import base64
import os
import ssl
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
    ctx = ssl.create_default_context()
    ca_pem = os.getenv("REQUESTS_CA_BUNDLE")
    if ca_pem:
        ctx.load_verify_locations(ca_pem)
    try:
        with urlopen(req, timeout=30, context=ctx) as resp:
            dest.write_bytes(resp.read())
    except HTTPError as e:
        raise RuntimeError(f"Config fetch failed [{e.code}]: {url}") from e
    except URLError as e:
        raise RuntimeError(f"Config fetch unreachable: {url} — {e.reason}") from e
