import json
import math
from pathlib import Path

import pytest
import yaml

from oversight import checks
from oversight import run as run_mod
from oversight.client import Completion, MockClient
from oversight.data.gpqa import load_csv
from oversight.schema import AliceAnswer, Check, ChecksMessage, Item

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "tests" / "fixtures" / "toy.csv"
ITEM = Item(qid="t1", question="Which gas makes up most of Earth's atmosphere?",
            options=["oxygen", "nitrogen", "argon", "carbon dioxide"], gold="B")
ANS = AliceAnswer(aid="t1#s0", qid="t1", sample=0, explanation="Oxygen is most abundant.", option="A", correct=False)
CFG = {"model": "mock"}


class Scripted(MockClient):
    """MockClient with fixed replies for chosen tags, and a log of every prompt per tag."""

    def __init__(self, replies: dict):
        super().__init__()
        self.replies, self.prompts = replies, {}

    def complete(self, *, tag, user, sample=0, **kw):
        self.prompts.setdefault(tag, []).append(user)
        key = (tag, sample) if (tag, sample) in self.replies else tag
        if key in self.replies:
            self.calls.append(tag)
            return Completion(text=json.dumps(self.replies[key]), stop_reason="stop")
        return super().complete(tag=tag, user=user, sample=sample, **kw)


def two_checks(*values):
    return {"checks": [{"question": f"What is quantity {i}?", "alice_value": v} for i, v in enumerate(values)]}


def test_bob_checks_regenerated_after_option_letter_leak():
    client = Scripted({("checks_write:bob", 0): two_checks("option C", "x"), ("checks_write:bob", 1): two_checks("3.2 eV", "x")})
    msg = checks.write_checks(client, CFG, ITEM, ANS, "bob", k=2, leak_cfg=CFG)
    assert msg.attempts == 2 and msg.flags == [] and [c.alice_value for c in msg.checks] == ["3.2 eV", "x"]


def test_bob_checks_dropped_when_ungrounded():
    client = Scripted({"checks_write:bob": two_checks("a", "b"),
                       "leak_check_checks": {"reason": "r", "states_verdict": False, "states_answer": False, "ungrounded": True}})
    msg = checks.write_checks(client, CFG, ITEM, ANS, "bob", k=2, leak_cfg=CFG, max_regenerations=1)
    assert msg.checks == [] and msg.flags == ["ungrounded"] and msg.attempts == 2


def test_self_checks_skip_leak_check_and_truncate_to_k():
    client = Scripted({"checks_write:self": two_checks("option C", "b", "c")})
    msg = checks.write_checks(client, CFG, ITEM, ANS, "self", k=2)
    assert len(msg.checks) == 2 and msg.flags == [] and "leak_check_checks" not in client.prompts


def test_blind_answer_never_sees_alice_answer_or_options():
    client = Scripted({"check_answer": {"answer": "nitrogen", "confidence": 0.9}})
    ans, conf = checks.answer_check(client, CFG, ITEM, Check(question="Which gas is most abundant in air?", alice_value="oxygen"))
    assert (ans, conf) == ("nitrogen", 0.9)
    prompt = client.prompts["check_answer"][0]
    assert "Oxygen is most abundant" not in prompt and "carbon dioxide" not in prompt and "oxygen" not in prompt


@pytest.mark.parametrize("p_agree, verdict", [((0.9, 0.8), "correct"), ((0.9, 0.2), "incorrect")])
def test_rule_verdict_needs_every_check_to_agree(p_agree, verdict):
    client = Scripted({"checks_final:bob": {
        "agreements": [{"check": i + 1, "p_agree": p} for i, p in enumerate(p_agree)],
        "rationale": "r", "verdict": "incorrect", "p_correct": 0.3}})
    msg = ChecksMessage(aid=ANS.aid, author="bob", attempts=1, flags=[],
                        checks=[Check(question="q1", alice_value="a", judge_answer="a"),
                                Check(question="q2", alice_value="b", judge_answer="c")])
    judge_v, rule_v = checks.final_verdicts(client, CFG, ITEM, ANS, msg)
    assert judge_v.condition == "checks_bob_judge" and judge_v.verdict == "incorrect"
    assert rule_v.condition == "checks_bob_rule" and rule_v.verdict == verdict
    assert rule_v.p_correct == pytest.approx(math.prod(p_agree))
    assert "Your own answer (given without seeing the response): c" in client.prompts["checks_final:bob"][0]


def test_missing_blind_answer_gives_no_verdict():
    msg = ChecksMessage(aid=ANS.aid, author="self", attempts=1, flags=[],
                        checks=[Check(question="q1", alice_value="a", judge_answer=None)])
    judge_v, rule_v = checks.final_verdicts(MockClient(), CFG, ITEM, ANS, msg)
    assert judge_v.verdict is None and rule_v.verdict is None and rule_v.condition == "checks_self_rule"


def test_pipeline_with_bob_and_self_checks(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["run"]["out_dir"] = str(tmp_path)
    cfg["analysis"]["n_bootstrap"] = 100
    cfg["controls"].update(checks_bob=True, checks_self=True)
    cfg["checks"] = {"k": 3}
    client = MockClient()
    summary = run_mod.run(cfg, client, load_csv(TOY, seed=0))

    judgments = [json.loads(l) for l in (tmp_path / "judgments.jsonl").read_text(encoding="utf-8").splitlines()]
    conds = {j["condition"] for j in judgments}
    assert {"checks_bob_rule", "checks_bob_judge", "checks_self_rule", "checks_self_judge"} <= conds
    rows = [json.loads(l) for l in (tmp_path / "checks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["author"] for r in rows} == {"bob", "self"}
    assert all(c["judge_answer"] and c["p_agree"] is not None for r in rows for c in r["checks"])
    ck = summary["checks"]
    assert ck["bob"]["with_checks"] == ck["self"]["with_checks"] == len(rows) // 2
    assert set(ck["bob_vs_self"]) == {"rule", "judge", "rule_itt", "judge_itt"}
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## Checks" in md and "Bob's checks, rule" in md and "ΔAUROC vs J1" in md
    # Bob writes with his model before any judge call; blind answers precede the final calls
    assert client.calls.index("checks_write:bob") < client.calls.index("judge:j1")
    assert max(i for i, t in enumerate(client.calls) if t == "check_answer") < client.calls.index("checks_final:bob")


def test_rule_flags_catch_vague_values_and_non_standalone_questions():
    ok = [Check(question="What is the pKa of acetic acid?", alice_value="4.76")]
    assert checks.rule_flags(ok) == []
    assert checks.rule_flags([Check(question="What is X?", alice_value="Not explicitly stated, but assumed")]) == ["vague_value"]
    assert checks.rule_flags([Check(question="What value does the response give for X?", alice_value="3")]) == ["not_standalone"]


def test_vague_bob_checks_are_regenerated_before_any_llm_audit():
    client = Scripted({("checks_write:bob", 0): {"checks": [{"question": "What is X?", "alice_value": "not stated"}]},
                       ("checks_write:bob", 1): {"checks": [{"question": "What is X?", "alice_value": "4.76"}]}})
    msg = checks.write_checks(client, CFG, ITEM, ANS, "bob", k=1, leak_cfg=CFG, max_regenerations=2)
    assert msg.attempts == 2 and msg.flags == [] and len(client.prompts["leak_check_checks"]) == 1


def test_itt_falls_back_to_j1_when_checks_are_missing(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["run"]["out_dir"] = str(tmp_path)
    cfg["analysis"]["n_bootstrap"] = 50
    cfg["controls"].update(checks_bob=True, checks_self=True)
    cfg["checks"] = {"max_regenerations": 0}

    class DropSome(MockClient):
        """The leak audit flags every other item, so some items lose Bob's checks."""
        def complete(self, *, tag, user, sample=0, **kw):
            if tag == "leak_check_checks" and len(user) % 2:
                self.calls.append(tag)
                return Completion(text=json.dumps({"reason": "r", "states_verdict": True, "states_answer": False,
                                                   "ungrounded": False}), stop_reason="stop")
            return super().complete(tag=tag, user=user, sample=sample, **kw)

    s = run_mod.run(cfg, DropSome(), load_csv(TOY, seed=0))
    plain = s["controls"]["checks_bob_judge"]["vs_j1"]["comparison"]["n"]
    itt = s["controls"]["checks_bob_judge_itt"]["vs_j1"]["comparison"]["n"]
    n_j1 = s["controls"]["checks_bob_judge_itt"]["vs_j1"]["j1"]["n"]
    assert s["checks"]["bob"]["dropped"] > 0
    assert plain < itt == n_j1  # ITT covers every item with a J1 verdict
    judgments = [json.loads(l) for l in (tmp_path / "judgments.jsonl").read_text(encoding="utf-8").splitlines()]
    j1 = {j["aid"]: j["verdict"] for j in judgments if j["condition"] == "j1"}
    dropped = [r["aid"] for r in map(json.loads, (tmp_path / "checks.jsonl").read_text(encoding="utf-8").splitlines())
               if r["author"] == "bob" and not r["checks"]]
    assert dropped and all(a in j1 for a in dropped)
    assert "ITT: J1 if no checks" in (tmp_path / "summary.md").read_text(encoding="utf-8")
