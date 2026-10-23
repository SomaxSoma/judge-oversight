"""Builds ../oversight_bundle.zip for Kaggle: the code and the cache. It never adds plaintext GPQA, results or keys.

Run from the project root:  uv run python scripts/make_bundle.py [--cache-dir PATH] [--gpqa-csv PATH]

--gpqa-csv adds GPQA encrypted (Fernet/AES) as data_enc/gpqa_diamond.csv.fernet, so a Kaggle run needs no
Hugging Face secret. The key lives only in the git-ignored .gpqa_bundle_key (created on first use) and is
injected into the private notebook by scripts/kaggle_push.py. The plaintext csv is never bundled.
"""

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "oversight_bundle.zip"

# Top-level folders only. src/oversight/data/ is code and must ship; ./data/ holds the GPQA csv and must not.
SKIP_TOP = {".venv", "data", "results", ".git", "cache"}
SKIP_ANYWHERE = {"__pycache__", ".pytest_cache"}
SKIP_FILES = {".env", ".gpqa_bundle_key"}  # secrets never ship
REQUIRED = ["pyproject.toml", "src/oversight/run.py", "src/oversight/data/gpqa.py", "src/oversight/data/__init__.py",
            "src/oversight/checks.py", "src/oversight/bob_solve.py", "src/oversight/leak_audit.py",
            "prompts/judge.md", "prompts/judge_testimony_block.md", "prompts/checks_write.md",
            "prompts/check_answer.md", "prompts/checks_final.md", "prompts/leak_check_checks.md",
            "prompts/leak_audit.md", "configs/kaggle_checks_32b.yaml", "configs/kaggle_solve_x4_jev.yaml",
            "configs/kaggle_notes_x4_jev.yaml"]


def included(rel: Path) -> bool:
    return not (
        rel.parts[0] in SKIP_TOP
        or SKIP_ANYWHERE & set(rel.parts)
        or rel.name in SKIP_FILES
        or rel.suffix == ".tmp"
    )


KEY_FILE = ROOT / ".gpqa_bundle_key"


def encrypt_gpqa(csv_path: Path) -> bytes:
    from cryptography.fernet import Fernet

    if not KEY_FILE.exists():
        KEY_FILE.write_bytes(Fernet.generate_key())
    return Fernet(KEY_FILE.read_bytes().strip()).encrypt(csv_path.read_bytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "cache"))
    ap.add_argument("--gpqa-csv", help="bundle GPQA encrypted (key in .gpqa_bundle_key)")
    args = ap.parse_args()
    cache = Path(args.cache_dir).resolve()
    files = [f for f in sorted(ROOT.rglob("*")) if f.is_file() and included(f.relative_to(ROOT))]
    cached = [f for f in sorted(cache.rglob("*.json"))]
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, Path("judge-oversight") / f.relative_to(ROOT))
        for f in cached:  # the cache always lands at judge-oversight/cache/ in the bundle
            z.write(f, Path("judge-oversight") / "cache" / f.relative_to(cache))
        if args.gpqa_csv:
            z.writestr("judge-oversight/data_enc/gpqa_diamond.csv.fernet", encrypt_gpqa(Path(args.gpqa_csv)))

    names = set(zipfile.ZipFile(OUT).namelist())
    missing = [r for r in REQUIRED if f"judge-oversight/{r}" not in names]
    leaked = [n for n in names if n.endswith((".env", ".gpqa_bundle_key")) or n.startswith("judge-oversight/data/") or n.endswith(".csv") and "fixtures" not in n]
    if missing or leaked:
        OUT.unlink()
        raise SystemExit(f"bundle rejected; missing={missing} leaked={leaked}")
    n_cache = sum(n.startswith("judge-oversight/cache/") for n in names)
    print(f"wrote {OUT} | {len(names)} files | {OUT.stat().st_size / 1e6:.2f} MB | {n_cache} cached calls")


if __name__ == "__main__":
    main()
