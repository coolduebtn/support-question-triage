import hashlib
import math

import pytest

from triage.ingest.index import build_index
from triage.retrieval import Bm25Retriever, HybridRetriever, VectorRetriever, rrf
from triage.schemas import Chunk, Retrieved

DIM = 64


def fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * DIM
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def chunk(i, text):
    return Chunk(id=f"c{i}", page_id=f"p{i}", page_title=f"Page {i}", heading_path=[f"Page {i}"],
                 text=text, url=f"https://x/{i}")


CHUNKS = [chunk(0, "reset the publishing config with the reset target"),
          chunk(1, "configure the payment gateway for orders"),
          chunk(2, "error ZXQ7 appears when exporting invoices"),
          chunk(3, "upgrade ofbiz to a new release version"),
          chunk(4, "create a new party and assign a role")]


@pytest.fixture
def index(tmp_path):
    build_index(CHUNKS, fake_embed, tmp_path / "idx")
    return tmp_path / "idx"


def test_vector_returns_k_sorted_desc_with_cosine_kind(index):
    res = VectorRetriever(index, fake_embed).retrieve("reset publishing config", 3)
    assert len(res) == 3
    assert [r.score for r in res] == sorted((r.score for r in res), reverse=True)
    assert res[0].chunk.id == "c0" and res[0].score_kind == "cosine"
    assert res[0].vector_cosine == res[0].score


def test_identical_text_scores_about_1(index):
    res = VectorRetriever(index, fake_embed).retrieve(CHUNKS[1].text, 1)
    assert res[0].chunk.id == "c1" and res[0].score == pytest.approx(1.0, abs=1e-3)


def test_index_persists_and_reloads(index):
    again = VectorRetriever(index, fake_embed)
    assert again.retrieve("payment gateway", 1)[0].chunk.id == "c1"


def test_index_info_hash_changes_with_content(tmp_path):
    a = build_index(CHUNKS, fake_embed, tmp_path / "a")
    b = build_index(CHUNKS[:3], fake_embed, tmp_path / "b")
    assert a.hash != b.hash and a.n_chunks == 5


def test_empty_query_returns_empty(index):
    assert VectorRetriever(index, fake_embed).retrieve("   ", 3) == []
    assert Bm25Retriever(CHUNKS).retrieve("", 3) == []


def test_bm25_ranks_exact_term_first():
    res = Bm25Retriever(CHUNKS).retrieve("ZXQ7", 3)
    assert res[0].chunk.id == "c2" and res[0].score_kind == "bm25"


def test_bm25_k_larger_than_corpus():
    assert len(Bm25Retriever(CHUNKS).retrieve("config", 50)) <= 5


def test_rrf_known_example():
    out = dict(rrf([["a", "b"], ["b", "c"], ["b"]], k=60))
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61 + 1 / 61)
    assert out["a"] == pytest.approx(1 / 61) and out["c"] == pytest.approx(1 / 62)
    assert [i for i, _ in rrf([["a", "b"], ["b", "c"], ["b"]])] == ["b", "a", "c"]


def test_rrf_deduplicates_ids():
    assert len(rrf([["a", "a", "b"], ["a"]])) == 2


class Stub:
    def __init__(self, ids, kind, cos=None):
        self.ids, self.kind, self.cos = ids, kind, cos

    def retrieve(self, query, k):
        return [Retrieved(chunk=CHUNKS[int(i[1:])], score=1.0 / (n + 1), score_kind=self.kind,
                          vector_cosine=self.cos) for n, i in enumerate(self.ids[:k])]


def test_hybrid_includes_doc_found_only_by_bm25():
    h = HybridRetriever(Stub(["c0", "c1"], "cosine", 0.8), Stub(["c2", "c0"], "bm25"), pool=20)
    ids = [r.chunk.id for r in h.retrieve("q", 3)]
    assert "c2" in ids and ids[0] == "c0"


def test_hybrid_score_kind_is_rrf_and_keeps_vector_cosine():
    h = HybridRetriever(Stub(["c0", "c1"], "cosine", 0.8), Stub(["c2", "c0"], "bm25"))
    res = h.retrieve("q", 3)
    assert all(r.score_kind == "rrf" for r in res)
    assert next(r for r in res if r.chunk.id == "c0").vector_cosine == 0.8
    assert next(r for r in res if r.chunk.id == "c2").vector_cosine is None
