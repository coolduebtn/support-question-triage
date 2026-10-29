from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


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


class Evidence(BaseModel):
    page_id: str
    heading: str
    quote: str


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
    source: Literal["agree", "adjudicated", "owner_blind", "owner_review"]


class RunResult(BaseModel):
    run_id: str
    config_hash: str
    split: str
    predictions: list[dict]
    metrics: dict
    n_errors: int
    cost_per_q: float
    latency_p50: float | None
