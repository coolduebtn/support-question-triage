import re

import tiktoken
from bs4 import BeautifulSoup

from triage.schemas import Chunk, Page

ENC = tiktoken.get_encoding("cl100k_base")
HEADINGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
BLOCKS = ["p", "li", "tr", "pre", "blockquote", "ac:plain-text-body"]
LEAFY = {"p", "li", "tr", "pre", "blockquote", "ac:plain-text-body"}


def _tokens(text: str) -> int:
    return len(ENC.encode(text))


def _block_text(el) -> str:
    if el.name == "tr":
        cells = [c.get_text(" ", strip=True) for c in el.find_all(["th", "td"])]
        return " | ".join(c for c in cells if c)
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True))


def _sections(page: Page) -> list[tuple[list[str], list[str]]]:
    soup = BeautifulSoup(page.body_html, "html.parser")
    for el in soup.find_all(["ac:parameter", "ri:page", "ri:attachment", "ri:url"]):
        el.decompose()
    path, sections, current = [], [], []  # path holds (level, title)
    root = [page.title]

    def flush():
        if current:
            sections.append((root + [t for _, t in path], list(current)))
            current.clear()

    for el in soup.find_all(HEADINGS + BLOCKS):
        if el.name in HEADINGS:
            flush()
            level = int(el.name[1])
            while path and path[-1][0] >= level:
                path.pop()
            title = _block_text(el)
            if title:
                path.append((level, title))
            continue
        if any(a.name in LEAFY for a in el.parents):
            continue
        text = _block_text(el)
        if text:
            current.append(text)
    flush()
    return sections


def _split_words(text: str, budget: int) -> list[str]:
    ids = ENC.encode(text)
    return [ENC.decode(ids[i:i + budget]) for i in range(0, len(ids), budget)]


def chunk_page(page: Page, max_tokens: int = 500) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading_path, blocks in _sections(page):
        prefix = " > ".join(heading_path) + "\n\n"
        budget = max_tokens - _tokens(prefix)
        if budget < 20:
            raise ValueError(f"breadcrumb for {page.id} leaves no room under {max_tokens} tokens")
        pieces: list[str] = []
        for b in blocks:
            pieces.extend(_split_words(b, budget) if _tokens(b) > budget else [b])
        group: list[str] = []
        for piece in pieces:
            if group and _tokens("\n".join(group + [piece])) > budget:
                chunks.append(_make(page, heading_path, prefix, group, len(chunks)))
                group = []
            group.append(piece)
        if group:
            chunks.append(_make(page, heading_path, prefix, group, len(chunks)))
    return chunks


def _make(page: Page, heading_path: list[str], prefix: str, group: list[str], n: int) -> Chunk:
    return Chunk(id=f"{page.id}-{n}", page_id=page.id, page_title=page.title,
                 heading_path=heading_path, text=prefix + "\n".join(group), url=page.url)


def page_text(page: Page) -> str:
    """Plain text of the whole page (no breadcrumbs), used to verify quoted evidence."""
    return "\n".join(block for _, blocks in _sections(page) for block in blocks)
