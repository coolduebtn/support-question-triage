import pytest
from pydantic import BaseModel

from triage.llm import (FakeLLM, LLMClient, ParamRejected, Price, RetryableError, Usage,
                        cost_from_usage)

PRICE = Price(input_per_m=1.0, output_per_m=2.0, embed_per_m=0.1)
MSG = [{"role": "user", "content": "hi"}]


class Backend:
    def __init__(self, fail=0, reject_temperature=False):
        self.chat_calls = 0
        self.fail = fail
        self.reject_temperature = reject_temperature
        self.temps = []

    def chat(self, model, messages, temperature, response_format):
        self.chat_calls += 1
        self.temps.append(temperature)
        if self.fail >= self.chat_calls:
            raise RetryableError("429")
        if self.reject_temperature and temperature is not None:
            raise ParamRejected("temperature")
        parsed = response_format(answer="ok") if response_format else None
        return "text", parsed, Usage(input_tokens=1000, output_tokens=500)

    def embed(self, model, texts):
        return [[float(len(t)), 1.0] for t in texts], 100


def client(tmp_path, backend):
    return LLMClient("m", tmp_path, PRICE, backend=backend, sleep=lambda s: None)


def test_cost_from_usage_known_example():
    assert cost_from_usage(Usage(input_tokens=1000, output_tokens=500), PRICE) == pytest.approx(0.002)


def test_cache_hit_keeps_nominal_cost_and_sets_cached_flag(tmp_path):
    b = Backend()
    c = client(tmp_path, b)
    first = c.complete(MSG)
    second = c.complete(MSG)
    assert (first.cached, second.cached) == (False, True)
    assert second.cost_usd == first.cost_usd == pytest.approx(0.002)
    assert second.latency_s == first.latency_s
    assert b.chat_calls == 1
    assert c.spent_usd == pytest.approx(0.002)


def test_different_repeat_values_are_different_cache_entries(tmp_path):
    b = Backend()
    c = client(tmp_path, b)
    c.complete(MSG, repeat=0)
    c.complete(MSG, repeat=1)
    assert b.chat_calls == 2


def test_cache_persists_across_clients(tmp_path):
    client(tmp_path, Backend()).complete(MSG)
    b2 = Backend()
    assert client(tmp_path, b2).complete(MSG).cached and b2.chat_calls == 0


def test_rate_limit_retried_three_times_then_raises(tmp_path):
    b = Backend(fail=5)
    with pytest.raises(RetryableError):
        client(tmp_path, b).complete(MSG)
    assert b.chat_calls == 3


def test_rate_limit_then_success(tmp_path):
    b = Backend(fail=2)
    assert client(tmp_path, b).complete(MSG).text == "text"
    assert b.chat_calls == 3


def test_temperature_dropped_for_models_that_reject_it(tmp_path):
    b = Backend(reject_temperature=True)
    c = client(tmp_path, b)
    c.complete(MSG, repeat=0)
    c.complete(MSG, repeat=1)
    assert b.temps == [0, None, None]


class Out(BaseModel):
    answer: str


def test_structured_output_roundtrips_through_cache(tmp_path):
    c = client(tmp_path, Backend())
    c.complete(MSG, response_format=Out)
    again = c.complete(MSG, response_format=Out)
    assert again.cached and again.parsed == Out(answer="ok")


def test_embed_cached_and_costed(tmp_path):
    c = client(tmp_path, Backend())
    vecs = c.embed(["ab", "abc"])
    assert vecs == [[2.0, 1.0], [3.0, 1.0]]
    assert c.embed(["ab", "abc"]) == vecs
    assert c.spent_usd == pytest.approx(100 / 1e6 * 0.1)


def test_fake_llm_scripted_order():
    f = FakeLLM(["one", Out(answer="two")])
    assert f.complete(MSG).text == "one"
    assert f.complete(MSG, response_format=Out).parsed == Out(answer="two")
    assert len(f.calls) == 2
    with pytest.raises(IndexError):
        f.complete(MSG)


def test_missing_structured_output_raises_and_is_not_cached(tmp_path):
    class NoParse(Backend):
        def chat(self, model, messages, temperature, response_format):
            self.chat_calls += 1
            return "refusal", None, Usage(input_tokens=1, output_tokens=1)

    b = NoParse()
    c = client(tmp_path, b)
    with pytest.raises(ValueError):
        c.complete(MSG, response_format=Out)
    with pytest.raises(ValueError):
        c.complete(MSG, response_format=Out)
    assert b.chat_calls == 2


def test_load_price_missing_model_names_the_model(tmp_path):
    from triage.llm import PriceMissing, load_price
    p = tmp_path / "prices.toml"
    p.write_text('[models."a"]\ninput_per_m = 1\noutput_per_m = 1\n')
    with pytest.raises(PriceMissing) as e:
        load_price("not-there", p)
    assert "not-there" in str(e.value)
