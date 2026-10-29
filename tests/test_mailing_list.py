import time

from triage.sources.mailing_list import (fetch_threads, is_noise, names_from_senders,
                                         scrub_pii)


def test_scrub_removes_emails_ips_and_header_names():
    text = "From: Jane Doe <jane.doe@example.com>\nServer 192.168.1.20 failed. Mail bob@corp.io"
    out = scrub_pii(text, {"Jane Doe"})
    assert "jane.doe@example.com" not in out and "bob@corp.io" not in out
    assert "192.168.1.20" not in out
    assert "Jane Doe" not in out


def test_scrub_removes_names_seen_in_from_fields_inside_bodies():
    names = names_from_senders(["Alice Sample <al...@example.org>"])
    out = scrub_pii("Thanks Alice, and regards from sample.", names)
    assert "Alice" not in out and "sample" not in out.lower()


def test_is_noise_flags_announce_cve_vote_release():
    for s in ["[ANNOUNCE] Apache OFBiz 18.12.09 released",
              "CVE-2023-46819: Apache OFBiz: Execution of Solr plugin queries",
              "[VOTE] Release 22.01.01", "Apache OFBiz board report"]:
        assert is_noise(s, ""), s


def test_is_noise_keeps_real_question():
    assert not is_noise("Order manager PDF not working", "The PDF is blank")
    assert not is_noise("how to upgrade ofbiz", "")


class Resp:
    def __init__(self, payload):
        self.status_code, self._p = 200, payload

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params=None, headers=None):
        self.calls.append(url)
        for key, payload in self.routes.items():
            if key in url:
                return Resp(payload)
        raise AssertionError(f"unexpected url {url}")


NOW = int(time.time())


def stats(emails, struct):
    return {"emails": emails, "thread_struct": struct}


def em(mid, subject, reply_to="", epoch=NOW - 1000, frm="Jane Doe <ja...@x.org>", body="short"):
    return {"mid": mid, "subject": subject, "in-reply-to": reply_to, "epoch": str(epoch),
            "from": frm, "body": body}


def test_fetch_threads_builds_threads_and_skips_noise_and_future():
    emails = [em("r1", "Order PDF not working"), em("a1", "Re: Order PDF not working", reply_to="r1",
                                                    body="Which version?"),
              em("n1", "[ANNOUNCE] release 1"), em("f1", "Future dated question", epoch=NOW + 10**7)]
    struct = [{"tid": "r1", "children": [{"tid": "a1", "children": []}]},
              {"tid": "n1", "children": []}, {"tid": "f1", "children": []}]
    s = FakeSession({"stats.lua": stats(emails, struct),
                     "email.lua?id=r1": {"body": "The PDF is blank. mail jane.doe@example.com", "epoch": NOW - 1000}})
    threads = fetch_threads("user", "ofbiz.apache.org", session=s, delay=0)
    assert [t.tid for t in threads] == ["r1"]
    t = threads[0]
    assert t.n_replies == 1 and t.replies_text == ["Which version?"]
    assert t.permalink == "https://lists.apache.org/thread/r1"
    assert "jane.doe@example.com" not in t.body and "blank" in t.body
    assert not any("n1" in c for c in s.calls)
