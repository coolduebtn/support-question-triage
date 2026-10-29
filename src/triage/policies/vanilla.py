from triage.schemas import Action, Decision, Retrieved

ANSWER_SYSTEM = """You answer IT support questions using only the documentation excerpts provided.
Cite the excerpt ids you used in square brackets, for example [123456-0]. Be concise."""


def format_context(retrieved: list[Retrieved]) -> str:
    return "\n\n".join(f"[{r.chunk.id}] {r.chunk.page_title} ({r.chunk.url})\n{r.chunk.text}" for r in retrieved)


class VanillaRAG:
    """Approach A: always answers."""

    name = "A"

    def __init__(self, llm, system_prompt: str = ANSWER_SYSTEM):
        self.llm, self.system_prompt = llm, system_prompt

    def decide(self, question: str, retrieved: list[Retrieved], repeat: int = 0) -> Decision:
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"QUESTION:\n{question}\n\nDOCUMENTATION:\n{format_context(retrieved)}"}]
        res = self.llm.complete(messages, repeat=repeat)
        return Decision(action=Action.ANSWER, reason="always answers", draft=res.text,
                        citations=[r.chunk.id for r in retrieved], cost_usd=res.cost_usd,
                        latency_s=res.latency_s, cached=res.cached)
