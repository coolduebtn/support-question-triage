from typing import Literal

from pydantic import BaseModel, Field

from triage.schemas import Annotation


class Snippet(BaseModel):
    chunk_id: str
    page_id: str
    page_title: str
    heading: str
    text: str
    score: float = 0.0


class EvidenceOut(BaseModel):
    page_id: str
    heading: str
    quote: str


class AnnotationOut(BaseModel):
    label: Literal["answer", "clarify", "escalate"]
    evidence: list[EvidenceOut] = Field(default_factory=list)
    missing_detail: str | None = None
    reference_answer: str | None = None


class Entailment(BaseModel):
    supports: bool


class LabelRecord(BaseModel):
    qid: str
    text: str
    split: Literal["dev", "test"]
    annotations: list[Annotation]
    final: Annotation
    agreed: bool
    adjudicated: bool = False
    flags: list[str] = Field(default_factory=list)
