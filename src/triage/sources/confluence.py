import hashlib
import json
import time
from collections.abc import Iterator
from pathlib import Path

import httpx

from triage.schemas import Page
from triage.sources.http import FetchError, get_json  # noqa: F401

BASE = "https://cwiki.apache.org/confluence"
def make_session(contact: str) -> httpx.Client:
    ua = f"capstone-support-triage/0.1 (research; contact: {contact})"
    return httpx.Client(headers={"User-Agent": ua}, timeout=30, follow_redirects=True)


def fetch_space_pages(space: str, session, limit: int = 200, delay: float = 1.0) -> Iterator[Page]:
    url = f"{BASE}/rest/api/space/{space}/content/page"
    params: dict | None = {"limit": limit, "start": 0, "expand": "body.storage,version"}
    while url:
        data = get_json(session, url, params, delay)
        base = data.get("_links", {}).get("base", BASE)
        for r in data.get("results", []):
            yield Page(
                id=str(r["id"]),
                title=r["title"],
                url=base + r.get("_links", {}).get("webui", ""),
                body_html=r.get("body", {}).get("storage", {}).get("value", ""),
                updated=r.get("version", {}).get("when", ""),
            )
        nxt = data.get("_links", {}).get("next")
        url, params = (base + nxt, None) if nxt else (None, None)
        if url:
            time.sleep(delay)


def write_snapshot(pages: list[Page], path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(p.model_dump(), sort_keys=True) + "\n" for p in sorted(pages, key=lambda p: p.id))
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def read_snapshot(path: Path) -> list[Page]:
    return [Page(**json.loads(line)) for line in Path(path).read_text().splitlines() if line]
