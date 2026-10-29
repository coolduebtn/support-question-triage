import hashlib
import json
import time
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class RetryableError(RuntimeError):
    """Transient API failure (rate limit, timeout, 5xx)."""


class ParamRejected(RuntimeError):
    """The model does not accept a request parameter (e.g. temperature)."""


class Price(BaseModel):
    input_per_m: float
    output_per_m: float
    embed_per_m: float = 0.0


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResult(BaseModel):
    text: str
    parsed: Any = None
    usage: Usage
    cost_usd: float
    cached: bool
    latency_s: float


def cost_from_usage(usage: Usage, price: Price) -> float:
    return usage.input_tokens / 1e6 * price.input_per_m + usage.output_tokens / 1e6 * price.output_per_m


class PriceMissing(ValueError):
    pass


def load_price(model: str, path: Path = Path("configs/prices.toml")) -> Price:
    models = tomllib.loads(Path(path).read_text())["models"]
    if model not in models:
        raise PriceMissing(f"no price for {model!r}; add it to {path} (check the OpenAI pricing page)")
    return Price(**models[model])


def _key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


class LLMClient:
    """Cached, cost-tracking wrapper. `cost_usd` on a result is the nominal cost of the call (kept
    on cache hits so per-question cost stays comparable); real money spent is `spent_usd`."""

    ATTEMPTS = 3

    def __init__(self, model: str, cache_dir: Path, price: Price, backend: Any = None, sleep=time.sleep):
        self.model, self.price, self.sleep = model, price, sleep
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._backend = backend
        self.spent_usd = 0.0
        self._temperature_ok = True

    @property
    def backend(self):
        if self._backend is None:
            self._backend = OpenAIBackend()
        return self._backend

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _retry(self, fn):
        for attempt in range(self.ATTEMPTS):
            try:
                return fn()
            except RetryableError:
                if attempt == self.ATTEMPTS - 1:
                    raise
                self.sleep(2 ** attempt)

    def complete(self, messages: list[dict], *, temperature: float | None = 0,
                 response_format: type[BaseModel] | None = None, repeat: int = 0) -> LLMResult:
        schema = response_format.__name__ if response_format else None
        path = self._path(_key("chat", self.model, messages, temperature, schema, repeat))
        if path.exists():
            d = json.loads(path.read_text())
            parsed = response_format.model_validate(d["parsed"]) if response_format and d["parsed"] is not None else None
            return LLMResult(text=d["text"], parsed=parsed, usage=Usage(**d["usage"]),
                             cost_usd=d["cost_usd"], cached=True, latency_s=d["latency_s"])
        start = time.perf_counter()

        def call():
            temp = temperature if self._temperature_ok else None
            try:
                return self.backend.chat(self.model, messages, temp, response_format)
            except ParamRejected:
                if temp is None:
                    raise
                self._temperature_ok = False
                return self.backend.chat(self.model, messages, None, response_format)

        text, parsed, usage = self._retry(call)
        if response_format is not None and parsed is None:
            raise ValueError(f"no structured output from {self.model} (refusal or truncation): {text[:80]!r}")
        latency = time.perf_counter() - start
        cost = cost_from_usage(usage, self.price)
        self.spent_usd += cost
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": text, "parsed": parsed.model_dump() if parsed is not None else None,
                                    "usage": usage.model_dump(), "cost_usd": cost, "latency_s": latency}))
        return LLMResult(text=text, parsed=parsed, usage=usage, cost_usd=cost, cached=False, latency_s=latency)

    def embed(self, texts: list[str]) -> list[list[float]]:
        path = self._path(_key("embed", self.model, texts))
        if path.exists():
            return json.loads(path.read_text())["vectors"]
        vectors, tokens = self._retry(lambda: self.backend.embed(self.model, texts))
        self.spent_usd += tokens / 1e6 * self.price.embed_per_m
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"vectors": vectors}))
        return vectors


class FakeLLM:
    """Scripted stand-in for tests: each call pops the next item (str, or a BaseModel as parsed output)."""

    def __init__(self, script: list):
        self.script, self.calls = list(script), []
        self.model = "fake"

    def complete(self, messages, *, temperature=0, response_format=None, repeat=0) -> LLMResult:
        self.calls.append({"messages": messages, "response_format": response_format, "repeat": repeat})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        parsed = item if isinstance(item, BaseModel) else None
        text = item if isinstance(item, str) else item.model_dump_json()
        return LLMResult(text=text, parsed=parsed, usage=Usage(), cost_usd=0.0, cached=False, latency_s=0.0)


class OpenAIBackend:
    """Thin adapter over the openai SDK. Not unit-tested offline: verified by `triage smoke` on key day."""

    def __init__(self):
        import openai

        self.openai = openai
        self.client = openai.OpenAI()

    def _wrap(self, fn):
        o = self.openai
        try:
            return fn()
        except (o.RateLimitError, o.APIConnectionError, o.APITimeoutError, o.InternalServerError) as e:
            raise RetryableError(str(e)) from e
        except o.BadRequestError as e:
            if "temperature" in str(e):
                raise ParamRejected("temperature") from e
            raise

    def chat(self, model, messages, temperature, response_format):
        kwargs: dict = {"model": model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if response_format is not None:
            r = self._wrap(lambda: self.client.chat.completions.parse(response_format=response_format, **kwargs))
            msg = r.choices[0].message
            usage = Usage(input_tokens=r.usage.prompt_tokens, output_tokens=r.usage.completion_tokens)
            return msg.content or "", msg.parsed, usage
        r = self._wrap(lambda: self.client.chat.completions.create(**kwargs))
        usage = Usage(input_tokens=r.usage.prompt_tokens, output_tokens=r.usage.completion_tokens)
        return r.choices[0].message.content or "", None, usage

    def embed(self, model, texts):
        r = self._wrap(lambda: self.client.embeddings.create(model=model, input=texts))
        return [d.embedding for d in r.data], r.usage.total_tokens
