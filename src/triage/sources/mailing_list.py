import re
import time

from triage.schemas import Thread
from triage.sources.http import get_json

API = "https://lists.apache.org/api"
NOISE_SUBJECT = re.compile(
    r"\[(ANN|ANNOUNCE|VOTE|RESULT|DISCUSS|CANCELLED?)\b|\bCVE-\d{4}-\d+|\breleased?\b|board report|digest|\bDRAFT\b",
    re.I)
EMAIL = re.compile(r"[\w.+-]*\w@[\w.\-\u2026]+")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
FROM_NAME = re.compile(r"^(.*?)\s*<")
LINKEDIN = re.compile(r"https?://\S*linkedin\.com\S*|\blinkedin\.com/\S+", re.I)
PHONE = re.compile(r"(?<![\w.])\+?\d[\d ().\-]{7,}\d(?![\w])")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _phone(m: re.Match) -> str:
    text = m.group(0)
    if ISO_DATE.fullmatch(text) or sum(c.isdigit() for c in text) < 10:
        return text
    return "[phone]"


def names_from_senders(senders: list[str]) -> set[str]:
    names: set[str] = set()
    for s in senders:
        m = FROM_NAME.match(s.strip())
        full = (m.group(1) if m else s).strip().strip('"')
        if not full or "@" in full:
            continue
        names.add(full)
        names.update(t for t in re.split(r"[\s,]+", full) if len(t) >= 5)
    return names


def scrub_pii(text: str, known_names: set[str]) -> str:
    text = EMAIL.sub("[email]", text)
    text = LINKEDIN.sub("[link]", text)
    text = IPV4.sub("[ip]", text)
    text = PHONE.sub(_phone, text)
    for name in sorted(known_names, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(name)}\b", "[name]", text, flags=re.I)
    return text


def is_noise(subject: str, body: str) -> bool:
    return bool(NOISE_SUBJECT.search(subject))


def _descendants(node: dict) -> list[str]:
    out: list[str] = []
    for child in node.get("children", []):
        out.append(child["tid"])
        out.extend(_descendants(child))
    return out


def fetch_threads(list_name: str, domain: str, window: str = "3y", session=None, delay: float = 1.0) -> list[Thread]:
    stats = get_json(session, f"{API}/stats.lua", {"list": list_name, "domain": domain, "d": f"lte={window}"}, delay)
    by_mid = {e["mid"]: e for e in stats["emails"]}
    names = names_from_senders([e.get("from", "") for e in stats["emails"]])
    now = time.time()
    threads: list[Thread] = []
    for node in stats["thread_struct"]:
        root = by_mid.get(node["tid"])
        if root is None or root.get("in-reply-to"):
            continue
        if is_noise(root["subject"], root.get("body", "")) or int(root["epoch"]) > now + 86400:
            continue
        time.sleep(delay)
        full = get_json(session, f"{API}/email.lua?id={root['mid']}", None, delay)
        replies = [by_mid[t].get("body", "") for t in _descendants(node) if t in by_mid]
        threads.append(Thread(
            tid=root["mid"],
            subject=scrub_pii(root["subject"], names),
            body=scrub_pii(full.get("body", ""), names),
            n_replies=len(replies),
            epoch=int(root["epoch"]),
            replies_text=[scrub_pii(r, names) for r in replies],
            permalink=f"https://lists.apache.org/thread/{root['mid']}",
        ))
    return threads
