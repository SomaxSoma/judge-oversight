"""Alice: answers each question, committing to one option. Her label is option == gold."""

import random

from oversight import prompts
from oversight.client import Client
from oversight.schema import LETTERS, AliceAnswer, Item

SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "final_option": {"type": "string", "enum": list(LETTERS)},
    },
    "required": ["explanation", "final_option"],
    "additionalProperties": False,
}


def answer(client: Client, cfg: dict, item: Item, sample: int) -> AliceAnswer:
    system, user = prompts.load("alice")
    out = client.complete(
        tag="alice",
        model=cfg["model"],
        system=system,
        user=user.format(question=item.question, options=item.options_block()),
        schema=SCHEMA,
        effort=cfg.get("effort"),
        max_tokens=cfg.get("max_tokens", 16000),
        sample=sample,
    ).json()
    option = out.get("final_option") if out else None
    if option not in set(LETTERS):  # also rejects None and multi-letter strings like "AB"
        option = None
    return AliceAnswer(
        aid=f"{item.qid}#s{sample}",
        qid=item.qid,
        sample=sample,
        explanation=str(out.get("explanation", "")) if out else "",
        option=option,
        correct=None if option is None else option == item.gold,
    )


def balance(answers: list[AliceAnswer], seed: int) -> list[AliceAnswer]:
    """Subsample to equal numbers of correct and incorrect answers."""
    usable = [a for a in answers if a.correct is not None]
    right = [a for a in usable if a.correct]
    wrong = [a for a in usable if not a.correct]
    n = min(len(right), len(wrong))
    rng = random.Random(seed)
    chosen = rng.sample(right, n) + rng.sample(wrong, n)
    return sorted(chosen, key=lambda a: a.aid)
