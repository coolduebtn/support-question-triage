import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Action(StrEnum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    ESCALATE = "escalate"


ACTIONS = [a.value for a in Action]


class Page(BaseModel):
    id: str
    title: str
    url: str
    body_html: str
    updated: str = ""


class Chunk(BaseModel):
    id: str
    page_id: str
    page_title: str
    heading_path: list[str]
    text: str
    url: str


class Thread(BaseModel):
    tid: str
    subject: str
    body: str
    n_replies: int
    epoch: int
    replies_text: list[str] = Field(default_factory=list)
    permalink: str = ""


class Question(BaseModel):
    qid: str
    thread_id: str
    text_raw: str
    text_clean: str
    permalink: str = ""
    split: Literal["dev", "test"] | None = None


class Retrieved(BaseModel):
    chunk: Chunk
    score: float
    score_kind: Literal["cosine", "bm25", "rrf", "cross_encoder"]
    vector_cosine: float | None = None


class Decision(BaseModel):
    action: Action | Literal["error"]
    reason: str = ""
    missing_detail: str | None = None
    citations: list[str] = Field(default_factory=list)
    draft: str | None = None
    error_detail: str | None = None
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cached: bool = False


def canonical_page_id(page_id: str) -> str:
    """Annotators sometimes cited a chunk id (`<page_id>-<n>`) where a page id was expected. Page ids never end in
    `-<digits>` (wiki ids are numeric, doc ids end in `.adoc`), so a trailing `-<digits>` is a chunk suffix."""
    return re.sub(r"-\d+$", "", page_id)


class Evidence(BaseModel):
    page_id: str
    heading: str
    quote: str

    @field_validator("page_id")
    @classmethod
    def _page_not_chunk(cls, v: str) -> str:
        return canonical_page_id(v)


class Annotation(BaseModel):
    qid: str
    annotator: str
    label: Action
    evidence: list[Evidence] = Field(default_factory=list)
    missing_detail: str | None = None
    reference_answer: str | None = None
    flags: list[str] = Field(default_factory=list)


class ReviewItem(BaseModel):
    qid: str
    annotations: list[Annotation]
    reason: Literal["disputed", "evidence_failed", "audit", "owner_blind"]


class GoldItem(BaseModel):
    qid: str
    text: str
    text_raw: str = ""
    label: Action
    evidence: list[Evidence]
    missing_detail: str | None
    reference_answer: str | None
    split: Literal["dev", "test"]
    source: Literal["agree", "adjudicated", "owner_blind", "owner_review", "synthetic_probe"]
    synthetic: bool = False
    derived_from: str | None = None


class RunResult(BaseModel):
    run_id: str
    config_hash: str
    split: str
    predictions: list[dict]
    metrics: dict
    n_errors: int
    cost_per_q: float
    latency_p50: float | None
