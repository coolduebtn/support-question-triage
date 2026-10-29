import json
from pathlib import Path
from typing import Protocol

import bm25s

from triage.ingest.index import EmbedFn, open_collection
from triage.schemas import Chunk, Retrieved


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[Retrieved]: ...


class VectorRetriever:
    def __init__(self, path: Path, embed_fn: EmbedFn):
        self.col = open_collection(path)
        self.embed_fn = embed_fn

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        if not query.strip():
            return []
        res = self.col.query(query_embeddings=self.embed_fn([query]), n_results=k,
                             include=["documents", "metadatas", "distances"])
        out = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            cos = 1.0 - float(dist)
            chunk = Chunk(id=cid, page_id=meta["page_id"], page_title=meta["page_title"], text=doc,
                          heading_path=json.loads(meta["heading_path"]), url=meta["url"])
            out.append(Retrieved(chunk=chunk, score=cos, score_kind="cosine", vector_cosine=cos))
        return out


class Bm25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.index = bm25s.BM25()
        self.index.index(bm25s.tokenize([c.text for c in chunks], stopwords="en", show_progress=False),
                         show_progress=False)

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        if not query.strip():
            return []
        tokens = bm25s.tokenize([query], stopwords="en", return_ids=False, show_progress=False)
        if not tokens[0]:
            return []
        docs, scores = self.index.retrieve(tokens, k=min(k, len(self.chunks)), show_progress=False)
        return [Retrieved(chunk=self.chunks[int(i)], score=float(s), score_kind="bm25")
                for i, s in zip(docs[0], scores[0]) if s > 0]


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        rank = 0
        for item in ranking:
            if item in seen:
                continue
            seen.add(item)
            rank += 1
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class HybridRetriever:
    def __init__(self, vector: Retriever, bm25: Retriever, pool: int = 20):
        self.vector, self.bm25, self.pool = vector, bm25, pool

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        v, b = self.vector.retrieve(query, self.pool), self.bm25.retrieve(query, self.pool)
        by_id = {r.chunk.id: r for r in b}
        by_id.update({r.chunk.id: r for r in v})
        cos = {r.chunk.id: r.vector_cosine for r in v}
        fused = rrf([[r.chunk.id for r in v], [r.chunk.id for r in b]])
        return [Retrieved(chunk=by_id[i].chunk, score=s, score_kind="rrf", vector_cosine=cos.get(i))
                for i, s in fused[:k]]


class Reranker:
    """Local cross-encoder. `scorer` can be injected for tests; otherwise sentence-transformers is loaded lazily."""

    def __init__(self, model_name: str, scorer=None):
        self.model_name, self._scorer = model_name, scorer

    def score(self, query: str, texts: list[str]) -> list[float]:
        if self._scorer is None:
            import torch
            from sentence_transformers import CrossEncoder

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = CrossEncoder(self.model_name, device=device)
            self._scorer = lambda q, ts: [float(s) for s in model.predict([(q, t) for t in ts])]
        return self._scorer(query, texts)


class RerankedRetriever:
    def __init__(self, base: Retriever, reranker: Reranker, pool: int = 20):
        self.base, self.reranker, self.pool = base, reranker, pool

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        candidates = self.base.retrieve(query, max(self.pool, k))
        if not candidates:
            return []
        scores = self.reranker.score(query, [c.chunk.text for c in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])[:k]
        return [Retrieved(chunk=c.chunk, score=s, score_kind="cross_encoder", vector_cosine=c.vector_cosine)
                for s, c in ranked]
