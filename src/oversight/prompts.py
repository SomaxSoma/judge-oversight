"""Loads role prompts from prompts/<name>.md. A file holds the system prompt, a
`---USER---` line, then the user template with {placeholders}."""

from functools import cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
SEPARATOR = "---USER---"


@cache
def _read(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def load(name: str) -> tuple[str, str]:
    """Return (system, user_template)."""
    system, _, user = _read(name).partition(SEPARATOR)
    return system.strip(), user.strip()


def fragment(name: str) -> str:
    return _read(name)
