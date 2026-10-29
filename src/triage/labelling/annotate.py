from typing import Protocol

from triage.labelling.models import AnnotationOut, Snippet
from triage.schemas import Action, Annotation, Evidence, Question

MAX_SNIPPETS = 12
MAX_SNIPPET_CHARS = 1200

SYSTEM = """You label support questions for an IT support triage study over a documentation wiki.
Choose exactly one label:
- answer: the documentation snippets below contain the answer. Quote the exact sentence(s) as evidence.
- clarify: the documentation could answer it, but the question lacks a specific detail (version, environment,
  error text, which feature). Name that detail in missing_detail.
- escalate: the documentation does not cover it.
Rules: quotes must be copied verbatim from a snippet (at least 40 characters) with its page_id.
Never invent evidence. If the snippets do not support an answer, choose escalate or clarify.
For answer, also write a short reference_answer using only the quoted evidence."""


class AnnotatorTools(Protocol):
    def candidates(self, text: str) -> list[Snippet]: ...
    def search_wiki(self, query: str) -> list[Snippet]: ...


def build_prompt(question: Question, snippets: list[Snippet]) -> str:
    seen: set[str] = set()
    parts = []
    for s in snippets:
        if s.chunk_id in seen:
            continue
        seen.add(s.chunk_id)
        parts.append(f"[{s.chunk_id}] page_id={s.page_id} | {s.page_title} > {s.heading}\n{s.text[:MAX_SNIPPET_CHARS]}")
        if len(parts) >= MAX_SNIPPETS:
            break
    return f"QUESTION:\n{question.text_clean}\n\nDOCUMENTATION SNIPPETS:\n\n" + "\n\n".join(parts)


def annotate(question: Question, tools: AnnotatorTools, llm) -> Annotation:
    snippets = tools.candidates(question.text_clean) + tools.search_wiki(question.text_clean)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(question, snippets)}]
    out: AnnotationOut = llm.complete(messages, response_format=AnnotationOut).parsed
    return Annotation(qid=question.qid, annotator=llm.model, label=Action(out.label),
                      evidence=[Evidence(**e.model_dump()) for e in out.evidence],
                      missing_detail=out.missing_detail, reference_answer=out.reference_answer)



STOP = {"the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "is", "it", "do", "i", "how", "my", "we",
        "with", "that", "this", "can", "be", "are", "you", "not", "when", "what", "get", "has", "have"}


def _terms(text: str) -> set[str]:
    import re
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2}


class WikiTools:
    """Evidence sources for annotators: union of vector and BM25 top results, plus a plain keyword-coverage
    search over the whole snapshot (a different scoring from either retriever)."""

    def __init__(self, chunks, vector, bm25, k: int = 10):
        self.chunks, self.vector, self.bm25, self.k = chunks, vector, bm25, k
        self._terms = [(c, _terms(c.text)) for c in chunks]

    def _snippet(self, c, score: float = 0.0) -> Snippet:
        return Snippet(chunk_id=c.id, page_id=c.page_id, page_title=c.page_title,
                       heading=" > ".join(c.heading_path), text=c.text, score=score)

    def candidates(self, text: str) -> list[Snippet]:
        seen: dict[str, Snippet] = {}
        for r in self.vector.retrieve(text, self.k) + self.bm25.retrieve(text, self.k):
            seen.setdefault(r.chunk.id, self._snippet(r.chunk))
        return list(seen.values())

    def search_wiki(self, query: str) -> list[Snippet]:
        q = _terms(query)
        if not q:
            return []
        scored = [(len(q & ts) / len(q), c) for c, ts in self._terms if q & ts]
        scored.sort(key=lambda x: (-x[0], len(x[1].text)))
        return [self._snippet(c, s) for s, c in scored[:5]]
