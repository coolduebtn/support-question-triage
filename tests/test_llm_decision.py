import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from triage.llm import Price
from triage.policies.llm_decision import LLMDecision
from triage.schemas import Action, Chunk, Retrieved

PRICE = Price(input_per_m=1.0, output_per_m=2.0)


def chunk(i, text="Run the reset target."):
    return Chunk(id=f"c{i}", page_id=f"p{i}", page_title=f"Page {i}", heading_path=[f"Page {i}"], text=text, url=f"https://x/{i}")


def retrieved(*ids):
    return [Retrieved(chunk=chunk(i), score=0.9, score_kind="cosine", vector_cosine=0.9) for i in ids]


class Scripted:
    """FunctionModel that replies with the next scripted output-tool arguments and records prompts."""

    def __init__(self, outputs):
        self.outputs, self.prompts, self.calls = list(outputs), [], 0

    def model(self):
        def fn(messages, info: AgentInfo):
            self.calls += 1
            self.prompts.append("\n".join(str(getattr(p, "content", "")) for m in messages for p in m.parts))
            args = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)],
                                 usage=RequestUsage(input_tokens=1000, output_tokens=500))
        return FunctionModel(fn)


def make(tmp_path, script, retries=2):
    return LLMDecision("fake-model", PRICE, tmp_path, retries=retries, model_obj=script.model())


ANSWER = {"action": "answer", "reason": "docs cover it", "citations": ["c1"], "draft": "Run the reset target. [c1]"}
CLARIFY = {"action": "clarify", "reason": "no version", "missing_detail": "OFBiz version", "draft": "Which version?"}


def test_valid_answer_passes(tmp_path):
    d = make(tmp_path, Scripted([ANSWER])).decide("How do I reset?", retrieved(1))
    assert d.action == Action.ANSWER and d.citations == ["c1"]


def test_answer_without_valid_citation_retried(tmp_path):
    s = Scripted([{**ANSWER, "citations": ["zzz"]}, ANSWER])
    d = make(tmp_path, s).decide("How do I reset?", retrieved(1))
    assert d.action == Action.ANSWER and s.calls == 2


def test_clarify_without_missing_detail_retried(tmp_path):
    s = Scripted([{**CLARIFY, "missing_detail": None}, CLARIFY])
    d = make(tmp_path, s).decide("It fails.", retrieved(1))
    assert d.action == Action.CLARIFY and d.missing_detail == "OFBiz version" and s.calls == 2


def test_exhausted_retries_returns_error_decision(tmp_path):
    s = Scripted([{**ANSWER, "citations": ["zzz"]}])
    d = make(tmp_path, s, retries=1).decide("How do I reset?", retrieved(1))
    assert d.action == "error" and d.error_detail


def test_prompt_contains_chunks_and_question(tmp_path):
    s = Scripted([ANSWER])
    make(tmp_path, s).decide("How do I reset the publishing config?", retrieved(1, 2))
    assert "How do I reset the publishing config?" in s.prompts[0]
    assert "[c1]" in s.prompts[0] and "[c2]" in s.prompts[0] and "Run the reset target." in s.prompts[0]


def test_empty_question_never_answers_and_makes_no_call(tmp_path):
    s = Scripted([ANSWER])
    d = make(tmp_path, s).decide("   ", retrieved(1))
    assert d.action == Action.ESCALATE and s.calls == 0


def test_usage_and_cost_recorded(tmp_path):
    d = make(tmp_path, Scripted([ANSWER])).decide("q?", retrieved(1))
    assert d.cost_usd == pytest.approx(0.002) and not d.cached and d.latency_s >= 0


def test_second_call_served_from_cache_with_same_cost(tmp_path):
    s = Scripted([ANSWER])
    pol = make(tmp_path, s)
    first = pol.decide("q?", retrieved(1))
    second = pol.decide("q?", retrieved(1))
    assert s.calls == 1 and second.cached and second.cost_usd == first.cost_usd and second.action == first.action


def test_different_repeat_is_not_served_from_cache(tmp_path):
    s = Scripted([ANSWER])
    pol = make(tmp_path, s)
    pol.decide("q?", retrieved(1), repeat=0)
    pol.decide("q?", retrieved(1), repeat=1)
    assert s.calls == 2
