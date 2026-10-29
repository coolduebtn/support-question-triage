from pydantic import BaseModel

from triage.schemas import Action, Chunk, Decision, GoldItem, Retrieved


class VerdictOut(BaseModel):
    passed: bool
    reason: str = ""


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool
    reason: str = ""


class ClaimsOut(BaseModel):
    claims: list[ClaimVerdict]


class JudgeVerdict(BaseModel):
    passed: bool
    reason: str = ""
    error: bool = False


CORRECT_SYSTEM = ("You grade a support answer against a reference answer. Pass only if the draft states the same "
                  "key facts or steps as the reference and contradicts nothing in it. Ignore style and length.")
CITATION_SYSTEM = ("Split the draft answer into its factual claims. For each claim say whether it is directly supported "
                   "by the cited documentation excerpts. Unsupported or invented details are not supported.")
FOLLOWUP_SYSTEM = ("A support agent asked a follow-up question. Pass only if the question asks for the specific missing "
                   "detail named below, and does not ask for unrelated information.")


def assert_judge_differs(system_model: str, judge_model: str) -> None:
    if system_model == judge_model:
        raise ValueError("the judge model must differ from the system model")


class Judge:
    """Binary, single-criterion rubrics. Parse or API failures retry, then become an error verdict (counted, not hidden)."""

    def __init__(self, llm, attempts: int = 2):
        self.llm, self.attempts = llm, attempts

    def _ask(self, system: str, user: str, fmt):
        last: Exception | None = None
        for _ in range(self.attempts):
            try:
                res = self.llm.complete([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                        response_format=fmt)
                if res.parsed is None:
                    raise ValueError("no structured output")
                return res.parsed
            except Exception as e:  # noqa: BLE001
                last = e
        raise RuntimeError(f"judge failed: {last!r}")

    def _verdict(self, system: str, user: str) -> JudgeVerdict:
        try:
            out: VerdictOut = self._ask(system, user, VerdictOut)
            return JudgeVerdict(passed=out.passed, reason=out.reason)
        except RuntimeError as e:
            return JudgeVerdict(passed=False, reason=str(e), error=True)

    def answer_correct(self, question: str, reference: str, draft: str) -> JudgeVerdict:
        return self._verdict(CORRECT_SYSTEM, f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nDRAFT:\n{draft}")

    def citation_supported(self, draft: str, cited_chunks: list[Chunk]) -> list[JudgeVerdict]:
        docs = "\n\n".join(f"[{c.id}] {c.text}" for c in cited_chunks)
        try:
            out: ClaimsOut = self._ask(CITATION_SYSTEM, f"DRAFT:\n{draft}\n\nCITED EXCERPTS:\n{docs}", ClaimsOut)
            return [JudgeVerdict(passed=c.supported, reason=f"{c.claim}: {c.reason}") for c in out.claims]
        except RuntimeError as e:
            return [JudgeVerdict(passed=False, reason=str(e), error=True)]

    def followup_targets_missing(self, followup: str, missing_detail: str) -> JudgeVerdict:
        return self._verdict(FOLLOWUP_SYSTEM, f"MISSING DETAIL: {missing_detail}\n\nFOLLOW-UP QUESTION:\n{followup}")

    def evaluate(self, item: GoldItem, decision: Decision, retrieved: list[Retrieved]) -> dict:
        out: dict = {"correct": None, "citation_support": None, "followup": None}
        if decision.action == Action.ANSWER and decision.draft:
            if item.label == Action.ANSWER and item.reference_answer:
                out["correct"] = self.answer_correct(item.text, item.reference_answer, decision.draft).passed
            cited = [r.chunk for r in retrieved if r.chunk.id in set(decision.citations)] or [r.chunk for r in retrieved]
            verdicts = [v for v in self.citation_supported(decision.draft, cited) if not v.error]
            if verdicts:
                out["citation_support"] = sum(v.passed for v in verdicts) / len(verdicts)
        if decision.action == Action.CLARIFY and item.label == Action.CLARIFY and decision.draft and item.missing_detail:
            out["followup"] = self.followup_targets_missing(decision.draft, item.missing_detail).passed
        return out
