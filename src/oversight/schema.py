"""Records passed between pipeline stages and written to results/*.jsonl."""

from pydantic import BaseModel

LETTERS = "ABCD"


class Item(BaseModel):
    qid: str
    question: str
    options: list[str]  # shuffled; options[0] is A
    gold: str  # correct option letter

    def options_block(self) -> str:
        return "\n".join(f"({LETTERS[i]}) {opt}" for i, opt in enumerate(self.options))


class AliceAnswer(BaseModel):
    aid: str  # "<qid>#s<sample>"
    qid: str
    sample: int
    explanation: str
    option: str | None  # None if Alice refused or output was unusable
    correct: bool | None

    def shown_to_judge(self, item: Item) -> str:
        idx = LETTERS.index(self.option)
        return f"{self.explanation}\n\nFinal answer: ({self.option}) {item.options[idx]}"


class BobMessage(BaseModel):
    aid: str
    strategy: str
    text: str | None  # None when every attempt was flagged or refused
    attempts: int
    flags: list[str]  # flags from the final attempt; empty means accepted
    words: int


class JudgeVerdict(BaseModel):
    aid: str
    condition: str  # j1 | j1_bob | bob_alone
    verdict: str | None  # correct | incorrect | None on refusal
    p_correct: float | None
    rationale: str


class Check(BaseModel):
    question: str
    alice_value: str
    judge_answer: str | None = None  # the judge's blind answer, filled in later
    judge_confidence: float | None = None
    p_agree: float | None = None  # judge's probability that alice_value agrees with its own answer


class ChecksMessage(BaseModel):
    aid: str
    author: str  # "bob" (helper-chosen) or "self" (judge-chosen)
    checks: list[Check]  # empty when every attempt was flagged or failed
    attempts: int
    flags: list[str]
