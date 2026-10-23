"""The judge: decides whether Alice's final option is correct, with no answer key.

The same prompt serves every condition; with Bob, his notes are inserted as one extra block.
"""

from oversight import prompts
from oversight.client import Client
from oversight.schema import AliceAnswer, Item, JudgeVerdict

SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "verdict": {"type": "string", "enum": ["correct", "incorrect"]},
        "p_correct": {"type": "number"},
    },
    "required": ["rationale", "verdict", "p_correct"],
    "additionalProperties": False,
}


def judge(
    client: Client,
    cfg: dict,
    item: Item,
    ans: AliceAnswer,
    condition: str,
    bob_notes: str | None = None,
    notes_block_name: str = "judge_notes_block",
    context_block: str | None = None,
) -> JudgeVerdict:
    """`bob_notes` go into prompts/<notes_block_name>.md; `context_block` is inserted as already rendered."""
    system, user = prompts.load("judge")
    if context_block is not None:
        notes_block = context_block
    elif bob_notes:
        notes_block = prompts.fragment(notes_block_name).format(notes=bob_notes)
    else:
        notes_block = ""
    out = client.complete(
        tag=f"judge:{condition}",
        model=cfg["model"],
        system=system,
        user=user.format(
            question=item.question,
            options=item.options_block(),
            answer=ans.shown_to_judge(item),
            notes_block=notes_block,
        ),
        schema=SCHEMA,
        effort=cfg.get("effort"),
        max_tokens=cfg.get("max_tokens", 8000),
    ).json()
    if not out or out.get("verdict") not in ("correct", "incorrect"):
        return JudgeVerdict(aid=ans.aid, condition=condition, verdict=None, p_correct=None, rationale="")
    try:
        p = min(1.0, max(0.0, float(out.get("p_correct"))))
    except (TypeError, ValueError):  # models without enforced schemas sometimes omit or mangle it
        p = 1.0 if out["verdict"] == "correct" else 0.0
    return JudgeVerdict(
        aid=ans.aid,
        condition=condition,
        verdict=out["verdict"],
        p_correct=p,
        rationale=str(out.get("rationale", "")),
    )
