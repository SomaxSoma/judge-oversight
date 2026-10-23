import json

import pytest

from oversight import alice
from oversight.client import Completion
from oversight.schema import Item

ITEM = Item(qid="q", question="?", options=["a", "b", "c", "d"], gold="B")


class Fixed:
    def __init__(self, text):
        self.text = text

    def complete(self, **kw):
        return Completion(text=self.text, stop_reason="stop")


@pytest.mark.parametrize(
    "payload, option, correct",
    [
        ({"explanation": "x", "final_option": "B"}, "B", True),
        ({"explanation": "x", "final_option": "C"}, "C", False),
        ({"explanation": "x", "final_option": None}, None, None),
        ({"explanation": "x", "final_option": "AB"}, None, None),
        ({"explanation": "x"}, None, None),
    ],
)
def test_answer_handles_missing_or_bad_options(payload, option, correct):
    a = alice.answer(Fixed(json.dumps(payload)), {"model": "m"}, ITEM, 0)
    assert (a.option, a.correct) == (option, correct)


def test_answer_handles_unparseable_output():
    a = alice.answer(Fixed("not json"), {"model": "m"}, ITEM, 0)
    assert a.option is None and a.correct is None
