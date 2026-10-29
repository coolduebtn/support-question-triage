import html
import re

CLOSING = re.compile(r"^(thanks|thank you|regards|best regards|kind regards|cheers|br|best|sincerely)[,.!]?\s*$", re.I)
STRONG_CLOSING = re.compile(r"^(thanks\s*&\s*)?(kind |best |warm |with )?(regards|sincerely|cheers)[,.!]?\s*$", re.I)
FOOTER = re.compile(r"^(原始邮件|-----|from:|发件人|sent:|de :|von:)", re.I)
CUT_MARKERS = [
    re.compile(r"^-{2,}\s*Original Message\s*-{2,}", re.I),
    re.compile(r"^On .{5,200} wrote:\s*$", re.I),
    re.compile(r"^Le .{5,200} a écrit\s*:\s*$", re.I),
    re.compile(r"^-- ?$"),
    re.compile(r"^Sent from my ", re.I),
]


def clean_email(text: str) -> str:
    kept: list[str] = []
    text = html.unescape(html.unescape(text))
    for line in text.splitlines():
        if any(m.match(line.strip()) or m.match(line) for m in CUT_MARKERS):
            break
        if line.lstrip().startswith(">"):
            continue
        kept.append(line.rstrip())
    while kept and not kept[-1].strip():
        kept.pop()
    for i, line in enumerate(kept):
        before_strong = [l for l in kept[:i] if l.strip()]
        if STRONG_CLOSING.match(line.strip()) and len(before_strong) >= 2:
            j = i + 1
            while j < len(kept) and kept[j].strip():  # the name / title block right under the closing
                j += 1
            rest = [l for l in kept[j:] if l.strip()]
            kept = kept[:i] + ([] if not rest or FOOTER.match(rest[0].strip()) else kept[j:])
            break
        trailing = [l for l in kept[i + 1:] if l.strip()]
        before = [l for l in kept[:i] if l.strip()]
        if (CLOSING.match(line.strip()) and len(before) >= 2 and len(trailing) <= 4
                and all(len(l.strip()) <= 60 for l in trailing)):
            kept = kept[:i]
            break
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return out
