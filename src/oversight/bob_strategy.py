"""Bob's strategy: the one file the autoresearch agent edits (see autoresearch/program.md).

Stage 0 scores `verdict`: Bob judging Alice's answer on his own.
Stage 1 scores `notes`: what Bob tells the blind judge.
Everything else is frozen: the judge, the leak checks, the metrics, the data and the models.

`client` reaches only Bob's model and enforces the per-item budget (autoresearch/score.py); it
also accepts extra={"think": True}. `cfg` is Bob's model config, e.g. {"model": ..., "max_tokens": ...}.

Baseline: Bob v1 exactly as in the main pipeline, so its calls come from the cache.
"""

from oversight import prompts
from oversight.judge import judge
from oversight.schema import AliceAnswer, Item


def verdict(client, cfg: dict, item: Item, ans: AliceAnswer) -> float | None:
    """Stage 0: Bob's probability that Alice's final option is correct, or None if he gives none."""
    return judge(client, cfg, item, ans, "bob_alone").p_correct


def notes(client, cfg: dict, item: Item, ans: AliceAnswer, attempt: int) -> str | None:
    """Stage 1: notes for the judge, or None. attempt > 0 means the previous attempt was flagged."""
    system, user = prompts.load("bob_v1")
    return client.complete(
        tag="bob",
        model=cfg["model"],
        system=system,
        user=user.format(question=item.question, options=item.options_block(), answer=ans.shown_to_judge(item)),
        effort=cfg.get("effort"),
        max_tokens=cfg.get("max_tokens", 16000),
        sample=attempt,
    ).text
