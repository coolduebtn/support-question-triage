import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import chromadb
from pydantic import BaseModel

from triage.schemas import Chunk

COLLECTION = "chunks"
BATCH = 128
EmbedFn = Callable[[list[str]], list[list[float]]]


class IndexInfo(BaseModel):
    hash: str
    n_chunks: int


def _collection(path: Path, create: bool):
    client = chromadb.PersistentClient(path=str(path / "chroma"))
    if create:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None)


def build_index(chunks: list[Chunk], embed_fn: EmbedFn, path: Path) -> IndexInfo:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    col = _collection(path, create=True)
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        col.add(ids=[c.id for c in batch], embeddings=embed_fn([c.text for c in batch]),
                documents=[c.text for c in batch],
                metadatas=[{"page_id": c.page_id, "page_title": c.page_title, "url": c.url,
                            "heading_path": json.dumps(c.heading_path)} for c in batch])
    body = "".join(json.dumps(c.model_dump(), sort_keys=True) + "\n" for c in chunks)
    (path / "chunks.jsonl").write_text(body)
    return IndexInfo(hash=hashlib.sha256(body.encode()).hexdigest(), n_chunks=len(chunks))


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**json.loads(line)) for line in (Path(path) / "chunks.jsonl").read_text().splitlines() if line]


def open_collection(path: Path):
    return _collection(Path(path), create=False)
