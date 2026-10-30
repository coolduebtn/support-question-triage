from triage.ingest.chunking import chunk_page
from triage.sources.docs_site import adoc_to_html, load_asciidoc_pages

ADOC = """:toc: left
// a comment
= Publishing Guide

include::_include/other.adoc[]

== Reset the config

Run the *reset* target from the command line. See <<config,the config page>> or link:https://example.org/x[the site].

* first step
* second step

[source,bash]
----
./gradlew reset
----

|===
|Name |Value
|a |b
|===
"""


def test_adoc_headings_paragraphs_lists_and_code():
    html = adoc_to_html(ADOC)
    assert "<h1>Publishing Guide</h1>" in html and "<h2>Reset the config</h2>" in html
    assert "<li>first step</li>" in html and "<li>second step</li>" in html
    assert "<pre>./gradlew reset</pre>" in html


def test_adoc_drops_directives_comments_and_attributes_and_flattens_inline_markup():
    html = adoc_to_html(ADOC)
    assert "include::" not in html and "a comment" not in html and ":toc:" not in html
    assert "Run the reset target" in html and "the config page" in html and "the site" in html
    assert "<<" not in html and "link:" not in html and "*reset*" not in html


def test_adoc_table_rows_are_kept_as_text():
    assert "Name | Value" in adoc_to_html(ADOC) and "a | b" in adoc_to_html(ADOC)


def test_adoc_html_escapes_markup_characters():
    assert "&lt;b&gt;" in adoc_to_html("= T\n\nUse <b> carefully\n")


def test_load_pages_builds_ids_urls_and_titles_and_chunks_cleanly(tmp_path):
    (tmp_path / "docs" / "asciidoc").mkdir(parents=True)
    (tmp_path / "docs" / "asciidoc" / "guide.adoc").write_text(ADOC)
    (tmp_path / "docs" / "asciidoc" / "empty.adoc").write_text(":toc:\n")
    pages = load_asciidoc_pages(tmp_path, "https://github.com/apache/ofbiz-framework/blob", "abc123")
    assert [p.id for p in pages] == ["doc:docs/asciidoc/guide.adoc"]
    p = pages[0]
    assert p.title == "Publishing Guide" and p.url.endswith("/abc123/docs/asciidoc/guide.adoc")
    chunks = chunk_page(p)
    assert chunks and chunks[0].heading_path[0] == "Publishing Guide"


def test_adoc_block_comments_such_as_licence_headers_are_dropped():
    html = adoc_to_html("////\nLicensed to the Apache Software Foundation (ASF)\n////\n= Help\n\nReal content here today.\n")
    assert "Licensed" not in html and "Real content here today." in html
