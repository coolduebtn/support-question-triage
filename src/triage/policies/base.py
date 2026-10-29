from typing import Protocol

from triage.schemas import Decision, Retrieved


class Policy(Protocol):
    name: str

    def decide(self, question: str, retrieved: list[Retrieved], repeat: int = 0) -> Decision: ...
