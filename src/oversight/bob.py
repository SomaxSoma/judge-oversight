"""Bob v1: one short critique of Alice's answer, with no verdict and no answer.

Bob never sees the reference. A message that fails the leak check is regenerated; if every
attempt fails, Bob's text is None and the item is left out of the paired comparison.
"""

from oversight import leakage, prompts
from oversight.client import Client
from oversight.schema import AliceAnswer, BobMessage, Item

STRATEGY = "v1_critique"
X4_STRATEGY = "v1_critique_x4_working"


def with_working(user: str, working: str) -> str:
    """The v1 critique prompt with Bob's private working added (the autoresearch "steered by working" design)."""
    out = user.replace(
        "Write notes for the grader",
        "Your own independent working (private; it may be wrong, and the grader never sees it):\n"
        f"<working>\n{working}\n</working>\n\n"
        "Use it only to decide which steps most need checking. Write notes for the grader",
        1,
    ).replace("- Do not give your own final answer.", "- Do not give your own final answer or reveal your working's conclusion.")
    assert "<working>" in out and "reveal your working's conclusion" in out, "bob_v1 prompt changed"
    return out


def critique(
    client: Client,
    bob_cfg: dict,
    leak_cfg: dict,
    item: Item,
    ans: AliceAnswer,
    max_regenerations: int = 1,
    use_llm_check: bool = True,
    working: str | None = None,
) -> BobMessage:
    """With `working`, Bob's private solution steers which steps he flags (the x4 notes)."""
    system, user = prompts.load("bob_v1")
    prompt = user.format(question=item.question, options=item.options_block(), answer=ans.shown_to_judge(item))
    if working is not None:
        prompt = with_working(prompt, working)
    text, flags = None, ["not_run"]
    attempt = 0
    for attempt in range(max_regenerations + 1):
        out = client.complete(
            tag="bob",
            model=bob_cfg["model"],
            system=system,
            user=prompt,
            effort=bob_cfg.get("effort"),
            max_tokens=bob_cfg.get("max_tokens", 16000),
            sample=attempt,
        )
        if out.text is None:
            text, flags = None, ["refusal"]
            continue
        text = out.text.strip()
        flags = leakage.regex_flags(text, item, ans.option)
        if not flags and use_llm_check:
            flags = leakage.llm_flags(client, leak_cfg, text, item, ans.option)
        if not flags:
            break
    accepted = not flags
    return BobMessage(
        aid=ans.aid,
        strategy=STRATEGY if working is None else X4_STRATEGY,
        text=text if accepted else None,
        attempts=attempt + 1,
        flags=flags,
        words=len(text.split()) if text else 0,
    )


FREE_STRATEGY = "free"


def free_critique(client: Client, bob_cfg: dict, item: Item, ans: AliceAnswer) -> BobMessage:
    """Control: Bob may state his verdict and preferred option. No leak check; it is the ceiling a
    verdict-free Bob is measured against."""
    system, user = prompts.load("bob_free")
    out = client.complete(
        tag="bob",
        model=bob_cfg["model"],
        system=system,
        user=user.format(question=item.question, options=item.options_block(), answer=ans.shown_to_judge(item)),
        effort=bob_cfg.get("effort"),
        max_tokens=bob_cfg.get("max_tokens", 16000),
    )
    text = out.text.strip() if out.text else None
    return BobMessage(
        aid=ans.aid,
        strategy=FREE_STRATEGY,
        text=text,
        attempts=1,
        flags=[] if text else ["refusal"],
        words=len(text.split()) if text else 0,
    )
