import re

CLOSING = re.compile(r"^(thanks|thank you|regards|best regards|kind regards|cheers|br|best|sincerely)[,.!]?\s*$", re.I)
CUT_MARKERS = [
    re.compile(r"^-{2,}\s*Original Message\s*-{2,}", re.I),
    re.compile(r"^On .{5,200} wrote:\s*$", re.I),
    re.compile(r"^Le .{5,200} a écrit\s*:\s*$", re.I),
    re.compile(r"^-- ?$"),
    re.compile(r"^Sent from my ", re.I),
]


def clean_email(text: str) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        if any(m.match(line.strip()) or m.match(line) for m in CUT_MARKERS):
            break
        if line.lstrip().startswith(">"):
            continue
        kept.append(line.rstrip())
    while kept and not kept[-1].strip():
        kept.pop()
    for i, line in enumerate(kept):
        trailing = [l for l in kept[i + 1:] if l.strip()]
        if CLOSING.match(line.strip()) and len(trailing) <= 4:
            kept = kept[:i]
            break
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return out
