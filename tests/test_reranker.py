from triage.retrieval import Reranker, RerankedRetriever
from triage.schemas import Chunk, Retrieved


def r(i, cos):
    c = Chunk(id=f"c{i}", page_id=f"p{i}", page_title="t", heading_path=["t"], text=f"text {i}", url="u")
    return Retrieved(chunk=c, score=cos, score_kind="cosine", vector_cosine=cos)


class Base:
    def __init__(self):
        self.asked = []

    def retrieve(self, query, k):
        self.asked.append(k)
        return [r(i, 1 - i / 10) for i in range(k)]


def stub_scorer(order_by_text):
    return lambda query, texts: [order_by_text[t] for t in texts]


def test_reranker_reorders_by_scores():
    scorer = stub_scorer({"text 0": 0.1, "text 1": 0.9, "text 2": 0.5})
    res = RerankedRetriever(Base(), Reranker("stub", scorer=scorer), pool=3).retrieve("q", 3)
    assert [x.chunk.id for x in res] == ["c1", "c2", "c0"]


def test_reranked_kind_is_cross_encoder_and_keeps_vector_cosine():
    scorer = stub_scorer({f"text {i}": float(i) for i in range(5)})
    res = RerankedRetriever(Base(), Reranker("stub", scorer=scorer), pool=5).retrieve("q", 2)
    assert all(x.score_kind == "cross_encoder" for x in res)
    assert res[0].chunk.id == "c4" and res[0].vector_cosine == 1 - 4 / 10


def test_pool_larger_than_k_then_truncated():
    base = Base()
    scorer = stub_scorer({f"text {i}": float(i) for i in range(20)})
    res = RerankedRetriever(base, Reranker("stub", scorer=scorer), pool=20).retrieve("q", 5)
    assert base.asked == [20] and len(res) == 5


def test_empty_pool_returns_empty():
    class Empty:
        def retrieve(self, q, k):
            return []
    assert RerankedRetriever(Empty(), Reranker("stub", scorer=lambda q, t: []), pool=5).retrieve("q", 3) == []
