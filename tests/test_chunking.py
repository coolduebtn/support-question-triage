import tiktoken

from triage.ingest.chunking import chunk_page
from triage.schemas import Page

ENC = tiktoken.get_encoding("cl100k_base")


def page(html, pid="1", title="Guide"):
    return Page(id=pid, title=title, url="https://x/p", body_html=html, updated="")


HTML = "<h1>Setup</h1><p>Install it.</p><h2>Config</h2><p>Edit the file.</p><ul><li>one</li><li>two</li></ul>"


def test_splits_on_headings_and_keeps_heading_path():
    chunks = chunk_page(page(HTML))
    paths = [c.heading_path for c in chunks]
    assert ["Guide", "Setup"] in paths
    assert ["Guide", "Setup", "Config"] in paths
    cfg = next(c for c in chunks if c.heading_path[-1] == "Config")
    assert "Edit the file." in cfg.text and "one" in cfg.text and "<" not in cfg.text


def test_embedding_text_prefixed_with_breadcrumb():
    c = next(c for c in chunk_page(page(HTML)) if c.heading_path[-1] == "Config")
    assert c.text.startswith("Guide > Setup > Config")


def test_page_without_headings_yields_chunks():
    chunks = chunk_page(page("<p>Just a paragraph.</p><p>Another one.</p>"))
    assert len(chunks) == 1 and chunks[0].heading_path == ["Guide"]
    assert "Another one." in chunks[0].text


def test_long_section_split_under_max_tokens():
    paras = "".join(f"<p>{'word ' * 120}number {i}.</p>" for i in range(12))
    chunks = chunk_page(page("<h1>Long</h1>" + paras), max_tokens=200)
    assert len(chunks) > 3
    assert all(len(ENC.encode(c.text)) <= 200 for c in chunks)


def test_single_giant_paragraph_is_still_bounded():
    chunks = chunk_page(page("<p>" + "alpha " * 3000 + "</p>"), max_tokens=100)
    assert len(chunks) > 1
    assert all(len(ENC.encode(c.text)) <= 100 for c in chunks)


def test_empty_page_yields_no_chunks():
    assert chunk_page(page("")) == []
    assert chunk_page(page("<h1>Only a heading</h1>")) == []


def test_chunk_ids_unique_and_stable():
    a = chunk_page(page(HTML))
    b = chunk_page(page(HTML))
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)


def test_duplicate_titles_get_distinct_ids():
    a = chunk_page(page(HTML, pid="1", title="Same"))
    b = chunk_page(page(HTML, pid="2", title="Same"))
    assert not ({c.id for c in a} & {c.id for c in b})


def test_table_rows_flattened_and_macro_parameters_dropped():
    html = ('<p>Intro</p><table><tr><th>Key</th><th>Val</th></tr><tr><td>a</td><td>b</td></tr></table>'
            '<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">xml</ac:parameter>'
            '<ac:plain-text-body>run me</ac:plain-text-body></ac:structured-macro>')
    text = chunk_page(page(html))[0].text
    assert "Key | Val" in text and "a | b" in text
    assert "xml" not in text.split("\n", 1)[1]


def test_page_text_joins_blocks_without_breadcrumb():
    from triage.ingest.chunking import page_text
    text = page_text(page(HTML))
    assert "Install it." in text and "Edit the file." in text
    assert "Guide >" not in text and "<" not in text
