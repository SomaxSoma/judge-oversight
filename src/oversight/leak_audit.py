"""Leak audit (evaluation only): can a model read Bob's private verdict off his verdict-free notes?

The auditor sees the question, Alice's response and Bob's notes, and predicts whether Bob privately judged
her answer correct. The AUROC of that prediction against Bob's actual verdict measures how much verdict the
notes carry. Notes far more decodable than the v1 notes suggest verdict smuggling.
"""

from oversight import prompts
from oversight.client import Client
from oversight.schema import AliceAnswer, Item, JudgeVerdict

SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "prediction": {"type": "string", "enum": ["correct", "incorrect"]},
        "p_correct": {"type": "number"},
    },
    "required": ["rationale", "prediction", "p_correct"],
    "additionalProperties": False,
}


def audit(client: Client, cfg: dict, item: Item, ans: AliceAnswer, notes: str, condition: str) -> JudgeVerdict:
    """`verdict`/`p_correct` here are the auditor's guess at Bob's private verdict, not at Alice's correctness."""
    system, user = prompts.load("leak_audit")
    out = client.complete(
        tag="leak_audit",
        model=cfg["model"],
        system=system,
        user=user.format(question=item.question, options=item.options_block(), answer=ans.shown_to_judge(item),
                         notes=notes),
        schema=SCHEMA,
        max_tokens=cfg.get("max_tokens", 8000),
    ).json()
    if not out or out.get("prediction") not in ("correct", "incorrect"):
        return JudgeVerdict(aid=ans.aid, condition=condition, verdict=None, p_correct=None, rationale="")
    try:
        p = min(1.0, max(0.0, float(out.get("p_correct"))))
    except (TypeError, ValueError):
        p = 1.0 if out["prediction"] == "correct" else 0.0
    return JudgeVerdict(aid=ans.aid, condition=condition, verdict=out["prediction"], p_correct=p,
                        rationale=str(out.get("rationale", "")))
