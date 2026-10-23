"""Pushes notebooks/kaggle_run.ipynb as a new version of your Kaggle notebook and starts it.

    uv run python scripts/kaggle_push.py --kernel USER/NOTEBOOK --configs kaggle_solve_x4_jev --dry-run

The settings cell gets CONFIGS, MAX_ITEMS and GPQA_KEY (read from the git-ignored .gpqa_bundle_key that
scripts/make_bundle.py --gpqa-csv created). The filled-in notebook only exists in a temp folder, which is
deleted after the push, so the key never lands in the repo. GPU, internet and the dataset come from the
notebook's existing Kaggle settings (pulled first), so a version pushed here runs like one saved on the site.
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_run.ipynb"
KEY_FILE = ROOT / ".gpqa_bundle_key"


def kaggle_cli() -> str:
    found = os.environ.get("KAGGLE_CLI") or shutil.which("kaggle")
    if found:
        return found
    conda = Path.home() / "anaconda3" / "Scripts" / "kaggle.exe"
    if conda.exists():
        return str(conda)
    raise SystemExit("kaggle CLI not found: install it or set KAGGLE_CLI")


def fill_settings(nb: dict, configs: list[str], max_items: int | None, key: str | None) -> dict:
    hits = 0
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if "CONFIGS = [" not in src or "MAX_ITEMS = None" not in src:
            continue
        lines = []
        for line in src.splitlines(keepends=True):
            if line.startswith("CONFIGS = ["):
                line = f"CONFIGS = {json.dumps(configs)}\n"
            elif line.startswith("MAX_ITEMS = "):
                line = f"MAX_ITEMS = {max_items}\n"
            elif line.startswith("GPQA_KEY = "):
                line = f"GPQA_KEY = {json.dumps(key)}\n"
            lines.append(line)
        cell["source"] = lines
        hits += 1
    if hits != 1:
        raise SystemExit(f"expected exactly one settings cell, found {hits}")
    return nb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, help="your Kaggle notebook, USER/SLUG")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--max-items", type=int)
    ap.add_argument("--dry-run", action="store_true", help="fill in and check the notebook, but don't push")
    args = ap.parse_args()

    for name in args.configs:
        if not (ROOT / "configs" / f"{name}.yaml").exists():
            raise SystemExit(f"no such config: configs/{name}.yaml")
    key = KEY_FILE.read_text().strip() if KEY_FILE.exists() else None
    if key is None:
        print("warning: no .gpqa_bundle_key; the notebook will need the HF_TOKEN secret")

    kaggle = kaggle_cli()
    tmp = Path(tempfile.mkdtemp(prefix="kaggle-push-"))
    try:
        subprocess.run([kaggle, "kernels", "pull", args.kernel, "-p", str(tmp), "-m"], check=True,
                       stdout=subprocess.DEVNULL)
        meta = json.loads((tmp / "kernel-metadata.json").read_text(encoding="utf-8"))
        nb = fill_settings(json.loads(NOTEBOOK.read_text(encoding="utf-8")), args.configs, args.max_items, key)
        (tmp / meta["code_file"]).write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"{args.kernel}: CONFIGS={args.configs} MAX_ITEMS={args.max_items} key={'set' if key else 'none'} "
              f"gpu={meta.get('enable_gpu')} internet={meta.get('enable_internet')} machine={meta.get('machine_shape')} "
              f"datasets={meta.get('dataset_sources')}")
        if not args.dry_run:
            subprocess.run([kaggle, "kernels", "push", "-p", str(tmp)], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # the filled-in notebook holds the key


if __name__ == "__main__":
    main()
