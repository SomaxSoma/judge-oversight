import json

import pytest

from oversight import bob, leakage
from oversight.client import Completion, MockClient
from oversight.schema import AliceAnswer, Item

ITEM = Item(
    qid="t1",
    question="Which gas makes up most of Earth's atmosphere?",
    options=["oxygen", "nitrogen", "argon", "carbon dioxide"],
    gold="B",
)


@pytest.mark.parametrize(
    "notes",
    [
        "The correct answer is nitrogen-based reasoning; check it.",
        "I think option C fits better.",
        "Compare against (D) before deciding.",
        "The response is wrong about the partial pressures.",
        "Alice's answer is correct.",
        "Check whether carbon dioxide could dominate instead.",
    ],
)
def test_regex_catches_planted_leaks(notes):
    assert leakage.regex_flags(notes, ITEM, alice_option="A")


@pytest.mark.parametrize(
    "notes",
    [
        "Check the step that converts volume fraction to mole fraction; if that conversion is off, the ranking changes.",
        "The response asserts (a) the composition is uniform and (b) water vapour is negligible; verify both.",
        "Ask whether the cited percentage refers to dry air. The argument relies on it without justification.",
        "Verify that oxygen is really the most abundant component, as the response claims.",  # Alice's own option
    ],
)
def test_regex_passes_clean_critiques(notes):
    assert leakage.regex_flags(notes, ITEM, alice_option="A") == []


class LeakyThenCleanBob(MockClient):
    """Bob's first attempt names an option; the regeneration is clean."""

    def complete(self, *, tag, sample=0, **kw):
        if tag == "bob":
            self.calls.append(tag)
            text = "The answer is option B." if sample == 0 else "Check the unit conversion in step two."
            return Completion(text=text, stop_reason="end_turn")
        return super().complete(tag=tag, sample=sample, **kw)


class AlwaysLeakyBob(MockClient):
    def complete(self, *, tag, sample=0, **kw):
        if tag == "bob":
            self.calls.append(tag)
            return Completion(text="The response is wrong.", stop_reason="end_turn")
        return super().complete(tag=tag, sample=sample, **kw)


class LLMFlagsVerdict(MockClient):
    def complete(self, *, tag, sample=0, **kw):
        if tag == "leak_check":
            self.calls.append(tag)
            return Completion(
                text=json.dumps({"reason": "implies wrong", "states_verdict": True, "states_answer": False}),
                stop_reason="end_turn",
            )
        return super().complete(tag=tag, sample=sample, **kw)


ANS = AliceAnswer(aid="t1#s0", qid="t1", sample=0, explanation="Oxygen is most abundant.", option="A", correct=False)
CFG = {"model": "mock"}


def test_bob_regenerates_after_leak():
    msg = bob.critique(LeakyThenCleanBob(), CFG, CFG, ITEM, ANS, max_regenerations=1)
    assert msg.text == "Check the unit conversion in step two."
    assert msg.attempts == 2 and msg.flags == []


def test_bob_dropped_when_every_attempt_leaks():
    msg = bob.critique(AlwaysLeakyBob(), CFG, CFG, ITEM, ANS, max_regenerations=1)
    assert msg.text is None
    assert msg.attempts == 2 and "verdict" in msg.flags


def test_llm_stage_flags_implicit_verdict():
    msg = bob.critique(LLMFlagsVerdict(), CFG, CFG, ITEM, ANS, max_regenerations=0)
    assert msg.text is None and msg.flags == ["llm_verdict"]
