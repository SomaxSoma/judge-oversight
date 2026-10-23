"""Write autoresearch/split.json once: a frozen, question-level dev/test split of GPQA-Diamond.

All of Alice's samples of a question land on the same side, so nothing about a test question is
seen during the search. Only GPQA record IDs are written, never question text.

    uv run python autoresearch/make_split.py            # refuses to overwrite an existing split
"""

import argparse
import json
import random
from pathlib import Path

import yaml

from oversight.data.gpqa import load_items

HERE = Path(__file__).resolve().parent
SEED = "autoresearch-split-v1"
N_DEV = 120  # of 198; the remaining 78 are the locked test split


def make_split(qids: list[str], n_dev: int = N_DEV, seed: str = SEED) -> dict:
    order = sorted(set(qids))
    random.Random(seed).shuffle(order)
    return {"seed": seed, "dev": order[:n_dev], "test": order[n_dev:]}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/autoresearch.yaml")
    p.add_argument("--out", default=str(HERE / "split.json"))
    p.add_argument("--n-dev", type=int, default=N_DEV)
    args = p.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists. The split is frozen; delete it by hand only before any experiment has run.")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    split = make_split([it.qid for it in load_items(cfg["data"])], args.n_dev)
    out.write_text(json.dumps(split, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(split['dev'])} dev, {len(split['test'])} test questions")


if __name__ == "__main__":
    main()
