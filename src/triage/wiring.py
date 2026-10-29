"""Builds retrievers, policies and run configurations from names and settings (used by the CLI)."""
import hashlib
import tomllib
from pathlib import Path

from triage.schemas import Action, Decision, Retrieved


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_limits(path: Path = Path("configs/limits.toml")) -> dict:
    return tomllib.loads(Path(path).read_text())


def run_config(policy: str, retriever: str, clean_query: bool, k: int, system_model: str, prompt: str,
               snapshot_hash: str, index_hash: str, threshold: float | None, seed: int,
               judge_model: str | None = None) -> dict:
    return {"policy": policy, "retriever": retriever, "clean_query": clean_query, "k": k,
            "system_model": system_model, "prompt": prompt, "snapshot_hash": snapshot_hash,
            "index_hash": index_hash, "threshold": threshold, "seed": seed, "judge_model": judge_model}


def render_decision(decision: Decision, retrieved: list[Retrieved]) -> str:
    action = decision.action.value if isinstance(decision.action, Action) else str(decision.action)
    lines = [f"ACTION: {action.upper()}", f"REASON: {decision.reason}"]
    if decision.error_detail:
        lines.append(f"ERROR: {decision.error_detail}")
    if decision.missing_detail:
        lines.append(f"MISSING DETAIL: {decision.missing_detail}")
    cited = [r for r in retrieved if r.chunk.id in set(decision.citations)] or retrieved[:3]
    seen: set[str] = set()
    sources = [r for r in cited if not (r.chunk.url in seen or seen.add(r.chunk.url))]
    if sources:
        lines.append("PAGES: " + " | ".join(f"{r.chunk.page_title} ({r.chunk.url})" for r in sources))
    if decision.draft:
        lines.append(f"DRAFT REPLY (not sent):\n{decision.draft}")
    return "\n".join(lines)


def run_ask(text: str, retriever, policy, k: int = 5) -> str:
    if not text.strip():
        return render_decision(Decision(action=Action.ESCALATE, reason="empty question, nothing to triage"), [])
    retrieved = retriever.retrieve(text, k)
    return render_decision(policy.decide(text, retrieved), retrieved)


def build_retriever(name: str, cfg, embed_fn):
    from triage.ingest.index import load_chunks
    from triage.retrieval import Bm25Retriever, HybridRetriever, Reranker, RerankedRetriever, VectorRetriever

    idx = cfg.data_dir / "index"
    vector = VectorRetriever(idx, embed_fn)
    if name == "vector":
        return vector
    hybrid = HybridRetriever(vector, Bm25Retriever(load_chunks(idx)))
    if name == "hybrid":
        return hybrid
    models = {"rerank-minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2", "rerank-bge": "BAAI/bge-reranker-v2-m3"}
    if name in models:
        return RerankedRetriever(hybrid, Reranker(models[name]))
    raise ValueError(f"unknown retriever {name!r}")
