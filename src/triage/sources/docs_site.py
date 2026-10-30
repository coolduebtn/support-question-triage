import html
import re
import subprocess
from pathlib import Path

from triage.schemas import Page

REPO = "https://github.com/apache/ofbiz-framework.git"
BLOB = "https://github.com/apache/ofbiz-framework/blob"
SPARSE = ["/docs/", "/applications/*/src/docs/", "/framework/*/src/docs/", "/themes/*/src/docs/"]
HEADING = re.compile(r"^(=+)\s+(.+?)\s*$")
XREF = re.compile(r"<<[^,>]+,\s*([^>]+)>>")
XREF_BARE = re.compile(r"<<([^,>]+)>>")
LINK = re.compile(r"(?:link:)?(?:https?://\S*?)?\[([^\]]+)\]")
EMPHASIS = re.compile(r"(?<![\w*])[*_]([^*_\n]+)[*_](?![\w*])")


def _inline(text: str) -> str:
    text = XREF.sub(r"\1", text)
    text = XREF_BARE.sub(r"\1", text)
    text = re.sub(r"link:\S*?\[([^\]]*)\]", r"\1", text)
    text = re.sub(r"https?://\S+?\[([^\]]*)\]", r"\1", text)
    text = EMPHASIS.sub(r"\1", text)
    return html.escape(text.replace("`", ""), quote=False)


def adoc_to_html(text: str) -> str:
    """Small AsciiDoc to HTML converter: headings, paragraphs, lists, listing blocks and table rows.
    Directives (include, image), attributes and comments are dropped. Enough for chunking, not a full renderer."""
    out: list[str] = []
    para: list[str] = []
    code: list[str] | None = None
    fence = ""
    in_table = False
    in_comment = False

    def flush():
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip() == "////":
            in_comment = not in_comment
            continue
        if in_comment:
            continue
        if code is not None:
            if line == fence:
                out.append(f"<pre>{html.escape(chr(10).join(code), quote=False)}</pre>")
                code = None
            else:
                code.append(line)
            continue
        if line in ("----", "....") or line.startswith("```"):
            flush()
            code, fence = [], line
            continue
        if line.startswith("|==="):
            flush()
            in_table = not in_table
            continue
        if line.startswith("//") or re.match(r"^:[\w-]+:", line) or re.match(r"^(include|image|ifdef|ifndef|endif)::", line):
            continue
        if re.match(r"^\[[^\]]*\]$", line) or re.match(r"^\.[A-Za-z]", line) and not line.startswith(". "):
            continue
        if in_table:
            cells = [c.strip() for c in line.lstrip("|").split("|") if c.strip()]
            if cells:
                out.append(f"<p>{_inline(' | '.join(cells))}</p>")
            continue
        m = HEADING.match(line)
        if m:
            flush()
            out.append(f"<h{min(len(m.group(1)), 6)}>{_inline(m.group(2))}</h{min(len(m.group(1)), 6)}>")
            continue
        li = re.match(r"^(\*+|-|\.+)\s+(.*)$", line)
        if li:
            flush()
            out.append(f"<li>{_inline(li.group(2))}</li>")
            continue
        if not line.strip():
            flush()
            continue
        para.append(line.strip())
    flush()
    return "\n".join(out)


def load_asciidoc_pages(root: Path, blob_base: str, commit: str) -> list[Page]:
    pages = []
    for path in sorted(Path(root).rglob("*.adoc")):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".git/"):
            continue
        body = adoc_to_html(path.read_text(errors="replace"))
        text_only = re.sub(r"<[^>]+>", " ", body)
        if len(text_only.split()) < 5:
            continue
        title = re.search(r"<h1>(.*?)</h1>", body)
        pages.append(Page(id=f"doc:{rel}", title=html.unescape(title.group(1)) if title else path.stem.replace("-", " "),
                          url=f"{blob_base}/{commit}/{rel}", body_html=body, updated=commit))
    return pages


def fetch_docs_repo(dest: Path) -> str:
    """Shallow sparse clone of the public repo (Apache-2.0), only the documentation sources. Returns the commit."""
    dest = Path(dest)
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "--no-cone", *SPARSE], check=True)
    return subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
