"""GPQA loading. The dataset is gated: accept its terms on Hugging Face and set HF_TOKEN.

Keep the data local (data/ is git-ignored); GPQA asks that it not be published in plain text.
"""

import csv
import os
import random
from pathlib import Path

from oversight.schema import LETTERS, Item

HF_REPO = "Idavidrein/gpqa"
HF_FILE = "gpqa_diamond.csv"
INCORRECT_COLS = ["Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]


def download_gpqa_diamond(data_dir: str | Path = "data") -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_FILE,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
            local_dir=data_dir,
        )
    )


def load_csv(path: str | Path, seed: int = 0, limit: int | None = None) -> list[Item]:
    """Read a GPQA-format csv and shuffle each question's options with a per-question seed."""
    items = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            correct = row["Correct Answer"].strip()
            options = [correct] + [row[c].strip() for c in INCORRECT_COLS]
            qid = row.get("Record ID") or f"q{len(items)}"
            random.Random(f"{seed}-{qid}").shuffle(options)
            items.append(
                Item(qid=qid, question=row["Question"].strip(), options=options, gold=LETTERS[options.index(correct)])
            )
    if limit:
        items = items[:limit]
    return items


def load_items(data_cfg: dict) -> list[Item]:
    if data_cfg["source"] == "gpqa_diamond":
        # A local copy (e.g. decrypted from the Kaggle bundle) is used as is, with no network call.
        local = Path("data") / HF_FILE
        path = local if local.exists() else download_gpqa_diamond()
    elif data_cfg["source"] == "csv":
        path = data_cfg["csv_path"]
    else:
        raise ValueError(f"unknown data source {data_cfg['source']!r}")
    return load_csv(path, seed=data_cfg.get("seed", 0), limit=data_cfg.get("limit_questions"))
