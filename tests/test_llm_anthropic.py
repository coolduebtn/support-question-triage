from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from triage.llm import LLMClient, Price, RetryableError, backend_for


class Out(BaseModel):
    verdict: str


def reply(text, tin=10, tout=5, blocks=None):
    content = blocks if blocks is not None else [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(content=content, usage=SimpleNamespace(input_tokens=tin, output_tokens=tout),
                           stop_reason="end_turn")


class FakeClient:
    def __init__(self, result):
        self.result, self.calls = result, []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def make(result):
    from triage.llm_anthropic import AnthropicBackend

    c = FakeClient(result)
    return AnthropicBackend(client=c), c


def test_system_messages_move_to_system_param_and_temperature_is_not_sent():
    be, c = make(reply("hello"))
    text, parsed, usage = be.chat("claude-opus-5-5", [{"role": "system", "content": "be brief"},
                                                       {"role": "user", "content": "hi"}], 0, None)
    kw = c.calls[0]
    assert kw["system"] == "be brief" and kw["messages"] == [{"role": "user", "content": "hi"}]
    assert "temperature" not in kw and "tool_choice" not in kw
    assert text == "hello" and parsed is None and (usage.input_tokens, usage.output_tokens) == (10, 5)


def test_structured_output_uses_json_schema_and_parses_into_the_model():
    be, c = make(reply('{"verdict": "correct"}'))
    text, parsed, _ = be.chat("claude-opus-5-5", [{"role": "user", "content": "x"}], None, Out)
    fmt = c.calls[0]["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["properties"]["verdict"]
    assert parsed == Out(verdict="correct")


def test_thinking_blocks_are_ignored_when_reading_text():
    blocks = [SimpleNamespace(type="thinking", thinking="hmm"), SimpleNamespace(type="text", text="ok")]
    be, _ = make(reply("", blocks=blocks))
    assert be.chat("claude-opus-5-5", [{"role": "user", "content": "x"}], None, None)[0] == "ok"


def test_invalid_structured_text_returns_none_so_the_client_raises():
    be, _ = make(reply("not json"))
    assert be.chat("claude-opus-5-5", [{"role": "user", "content": "x"}], None, Out)[1] is None


def test_rate_limit_and_server_errors_are_retryable():
    import anthropic
    import httpx

    req = httpx.Request("POST", "https://x")
    for exc in (anthropic.RateLimitError("slow", response=httpx.Response(429, request=req), body=None),
                anthropic.InternalServerError("boom", response=httpx.Response(500, request=req), body=None)):
        be, _ = make(exc)
        with pytest.raises(RetryableError):
            be.chat("claude-opus-5-5", [{"role": "user", "content": "x"}], None, None)


def test_backend_for_routes_by_model_name():
    from triage.llm_anthropic import AnthropicBackend

    assert isinstance(backend_for("claude-opus-5-5", client=FakeClient(reply("x"))), AnthropicBackend)


def test_llmclient_with_claude_backend_caches_and_costs(tmp_path):
    be, c = make(reply('{"verdict": "ok"}', tin=1_000_000, tout=0))
    llm = LLMClient("claude-opus-5-5", tmp_path, Price(input_per_m=4.0, output_per_m=20.0), backend=be)
    a = llm.complete([{"role": "user", "content": "q"}], response_format=Out)
    b = llm.complete([{"role": "user", "content": "q"}], response_format=Out)
    assert a.cost_usd == pytest.approx(4.0) and b.cached and len(c.calls) == 1
