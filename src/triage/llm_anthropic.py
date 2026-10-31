from pydantic import BaseModel, ValidationError

from triage.llm import RetryableError, Usage


class AnthropicBackend:
    """Claude adapter with the same `chat` interface as `OpenAIBackend`.

    Sampling parameters are never sent (the current Claude models reject them), and structured output
    uses `output_config.format` rather than forced tool use, which those models also reject.
    """

    MAX_TOKENS = 16000

    def __init__(self, client=None):
        import anthropic

        self.anthropic = anthropic
        self.client = client or anthropic.Anthropic()

    def chat(self, model, messages, temperature, response_format):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        kwargs: dict = {"model": model, "max_tokens": self.MAX_TOKENS,
                        "messages": [m for m in messages if m["role"] != "system"]}
        if system:
            kwargs["system"] = system
        if response_format is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema",
                                                  "schema": self.anthropic.transform_schema(response_format)}}
        a = self.anthropic
        try:
            r = self.client.messages.create(**kwargs)
        except (a.RateLimitError, a.APIConnectionError, a.APITimeoutError, a.InternalServerError) as e:
            raise RetryableError(str(e)) from e
        text = "".join(b.text for b in r.content if b.type == "text")
        usage = Usage(input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens)
        return text, _parse(response_format, text), usage

    def embed(self, model, texts):
        raise NotImplementedError("Anthropic has no embeddings endpoint; embeddings stay on OpenAI")


def _parse(response_format: type[BaseModel] | None, text: str):
    if response_format is None:
        return None
    try:
        return response_format.model_validate_json(text)
    except ValidationError:
        return None
