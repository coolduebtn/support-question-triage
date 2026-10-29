import json
from pathlib import Path

import pytest

from triage.schemas import Page
from triage.sources.confluence import (FetchError, fetch_space_pages, make_session,
                                       read_snapshot, write_snapshot)

FIX = json.loads((Path(__file__).parent / "fixtures" / "confluence_page.json").read_text())


class Resp:
    def __init__(self, status, payload=None):
        self.status_code, self._p = status, payload or {}

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, headers=None):
        self.calls.append(url)
        return self.responses.pop(0)


def page_payload(ids, nxt):
    results = [dict(FIX["results"][0], id=i, title=f"T{i}") for i in ids]
    links = {"base": "https://cwiki.apache.org/confluence"}
    if nxt:
        links["next"] = "/rest/api/space/OFBIZ/content/page?start=2"
    return {"results": results, "_links": links}


def test_fetch_paginates_until_no_next():
    s = FakeSession([Resp(200, page_payload(["1", "2"], True)), Resp(200, page_payload(["3"], False))])
    pages = list(fetch_space_pages("OFBIZ", s, delay=0))
    assert [p.id for p in pages] == ["1", "2", "3"]
    assert len(s.calls) == 2
    assert pages[0].url.startswith("https://cwiki.apache.org/confluence/display/OFBIZ")
    assert "Run the reset target" in pages[0].body_html


def test_user_agent_uses_env_contact():
    s = make_session("me@example.org")
    ua = s.headers["User-Agent"]
    assert "capstone" in ua and "me@example.org" in ua


def test_snapshot_roundtrip_and_hash_stable(tmp_path):
    pages = [Page(id="1", title="a", url="u", body_html="<p>x</p>", updated="d")]
    h1 = write_snapshot(pages, tmp_path / "a.jsonl")
    h2 = write_snapshot(pages, tmp_path / "b.jsonl")
    assert h1 == h2 and len(h1) == 64
    assert read_snapshot(tmp_path / "a.jsonl") == pages


def test_http_500_three_times_raises_fetch_error():
    s = FakeSession([Resp(500), Resp(500), Resp(500)])
    with pytest.raises(FetchError):
        list(fetch_space_pages("OFBIZ", s, delay=0))
    assert len(s.calls) == 3
