"""Checks that Bob's notes give neither a verdict nor an answer.

Stage 1 is a regex pass for explicit leaks; stage 2 is an LLM classifier for implicit ones.
"""

import re

from oversight import prompts
from oversight.client import Client
from oversight.schema import LETTERS, Item

_PATTERNS = {
    # Letters must be upper-case so "(a)"-style enumerations in Bob's notes don't trip it.
    "option_letter": re.compile(r"\b(?i:option|choice|answer)\s*\(?[A-D]\)?(?![\w-])|\([A-D]\)"),
    "answer_statement": re.compile(
        r"\b(?:correct|right|true|actual)\s+(?:answer|option|choice)\s+(?:is|would\s+be|should\s+be|must\s+be)\b",
        re.I,
    ),
    "verdict": re.compile(
        r"\b(?:the\s+)?(?:response|answer|alice|model|solution)(?:'s\s+(?:answer|conclusion|final\s+\w+))?"
        r"\s+(?:is|was|seems|appears|looks)\s+(?:\w+\s+)?(?:correct|incorrect|right|wrong|mistaken|accurate)\b",
        re.I,
    ),
}

LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "states_verdict": {"type": "boolean"},
        "states_answer": {"type": "boolean"},
    },
    "required": ["reason", "states_verdict", "states_answer"],
    "additionalProperties": False,
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def regex_flags(notes: str, item: Item, alice_option: str) -> list[str]:
    flags = [name for name, pat in _PATTERNS.items() if pat.search(notes)]
    # Quoting another option's full text points the judge at an alternative answer.
    text = _norm(notes)
    for i, opt in enumerate(item.options):
        o = _norm(opt)
        if LETTERS[i] != alice_option and len(o) >= 4 and o in text:
            flags.append(f"mentions_option_{LETTERS[i]}")
    return flags


def llm_flags(client: Client, cfg: dict, notes: str, item: Item, alice_option: str) -> list[str]:
    system, user = prompts.load("leak_check")
    out = client.complete(
        tag="leak_check",
        model=cfg["model"],
        system=system,
        user=user.format(question=item.question, options=item.options_block(), alice_option=alice_option, notes=notes),
        schema=LLM_SCHEMA,
        effort=cfg.get("effort"),
        max_tokens=cfg.get("max_tokens", 4000),
    ).json()
    if out is None:
        return ["leak_check_failed"]
    flags = []
    if out["states_verdict"]:
        flags.append("llm_verdict")
    if out["states_answer"]:
        flags.append("llm_answer")
    return flags
