import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext

from triage.llm import Price, Usage, cost_from_usage
from triage.policies.vanilla import format_context
from triage.schemas import Action, Decision, Retrieved

PROMPT_PATH = Path(__file__).parent / "prompts" / "decision_v1.md"


class DecisionOut(BaseModel):
    action: Action
    reason: str
    missing_detail: str | None = None
    citations: list[str] = Field(default_factory=list)
    draft: str | None = None


@dataclass
class Deps:
    chunk_ids: set[str]


def prompt_text() -> str:
    return PROMPT_PATH.read_text()


class LLMDecision:
    """Approach B: one typed, validated LLM decision over the retrieved excerpts (Pydantic AI)."""

    name = "B"

    def __init__(self, model_name: str, price: Price, cache_dir: Path, retries: int = 2, model_obj=None,
                 prompt: str | None = None):
        self.model_name, self.price, self.retries = model_name, price, retries
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_obj, self.prompt = model_obj, prompt or prompt_text()
        self._agent: Agent | None = None
        self.spent_usd = 0.0

    def _get_agent(self) -> Agent:
        if self._agent is None:
            agent = Agent(self.model_obj or f"openai:{self.model_name}", output_type=DecisionOut,
                          deps_type=Deps, instructions=self.prompt, retries=self.retries)

            @agent.output_validator
            def check(ctx: RunContext[Deps], out: DecisionOut) -> DecisionOut:
                if out.action == Action.CLARIFY and not (out.missing_detail or "").strip():
                    raise ModelRetry("clarify requires missing_detail naming the specific missing detail")
                if out.action == Action.ANSWER and not [c for c in out.citations if c in ctx.deps.chunk_ids]:
                    raise ModelRetry("answer must cite at least one excerpt id that appears in the documentation")
                return out

            self._agent = agent
        return self._agent

    def _key(self, question: str, retrieved: list[Retrieved], repeat: int) -> str:
        blob = json.dumps([self.model_name, self.prompt, question, [(r.chunk.id, r.chunk.text) for r in retrieved], repeat])
        return hashlib.sha256(blob.encode()).hexdigest()

    def decide(self, question: str, retrieved: list[Retrieved], repeat: int = 0) -> Decision:
        if not question.strip():
            return Decision(action=Action.ESCALATE, reason="empty question")
        path = self.cache_dir / f"{self._key(question, retrieved, repeat)}.json"
        if path.exists():
            return Decision(**{**json.loads(path.read_text()), "cached": True})
        start = time.perf_counter()
        try:
            result = self._get_agent().run_sync(
                f"QUESTION:\n{question}\n\nDOCUMENTATION EXCERPTS:\n{format_context(retrieved)}",
                deps=Deps(chunk_ids={r.chunk.id for r in retrieved}))
        except Exception as e:  # noqa: BLE001 - retries exhausted or API failure: counted, not hidden
            return Decision(action="error", reason="decision step failed", error_detail=repr(e),
                            latency_s=time.perf_counter() - start)
        usage = Usage(input_tokens=result.usage.input_tokens, output_tokens=result.usage.output_tokens)
        cost = cost_from_usage(usage, self.price)
        self.spent_usd += cost
        out = result.output
        decision = Decision(action=out.action, reason=out.reason, missing_detail=out.missing_detail,
                            citations=out.citations, draft=out.draft, cost_usd=cost,
                            latency_s=time.perf_counter() - start)
        path.write_text(decision.model_dump_json())
        return decision
