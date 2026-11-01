from triage.ingest.index import build_index
from triage.schemas import Chunk


def _ef(col) -> int:
    return col.configuration["hnsw"]["ef_search"]


def test_new_index_uses_the_raised_hnsw_search_width(tmp_path):
    from triage.ingest.index import HNSW_EF_SEARCH, open_collection

    build_index([Chunk(id="a-0", page_id="a", page_title="A", text="alpha", heading_path=["A"], url="u")],
                lambda texts: [[1.0, 0.0] for _ in texts], tmp_path)
    assert HNSW_EF_SEARCH >= 400 and _ef(open_collection(tmp_path)) == HNSW_EF_SEARCH


def test_opening_an_older_index_raises_a_low_search_width(tmp_path):
    import chromadb

    from triage.ingest.index import COLLECTION, HNSW_EF_SEARCH, open_collection

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    old = client.get_or_create_collection(COLLECTION, configuration={"hnsw": {"space": "cosine", "ef_search": 100}},
                                          embedding_function=None)
    assert _ef(old) == 100
    assert _ef(open_collection(tmp_path)) == HNSW_EF_SEARCH


def test_building_an_index_also_writes_public_page_titles(tmp_path):
    import json

    index = tmp_path / "index"
    build_index([Chunk(id="a-0", page_id="a", page_title="A page", text="alpha", heading_path=["A"], url="https://u/a"),
                 Chunk(id="a-1", page_id="a", page_title="A page", text="beta", heading_path=["A"], url="https://u/a")],
                lambda texts: [[1.0, 0.0] for _ in texts], index)
    assert json.loads((tmp_path / "page_titles.json").read_text()) == {"a": ["A page", "https://u/a"]}
