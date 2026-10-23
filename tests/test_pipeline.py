"""End-to-end dry run on the toy csv with MockClient (no API calls)."""

import json
from pathlib import Path

import yaml

from oversight import run as run_mod
from oversight.client import MockClient
from oversight.data.gpqa import load_csv

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "tests" / "fixtures" / "toy.csv"


def test_load_csv_shuffles_and_tracks_gold():
    items = load_csv(TOY, seed=0)
    assert len(items) == 10
    for it in items:
        assert sorted(it.options) == sorted(set(it.options))
        assert it.gold in "ABCD"
    # gold is not always A, i.e. options really are shuffled
    assert len({it.gold for it in items}) > 1
    assert load_csv(TOY, seed=0) == items


def test_dry_run_end_to_end(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["run"]["out_dir"] = str(tmp_path)
    cfg["analysis"]["n_bootstrap"] = 200
    client = MockClient()

    summary = run_mod.run(cfg, client, load_csv(TOY, seed=0))

    for name in ("alice.jsonl", "eval_set.jsonl", "bob.jsonl", "judgments.jsonl", "summary.json", "summary.md"):
        assert (tmp_path / name).exists(), name

    eval_set = [json.loads(l) for l in (tmp_path / "eval_set.jsonl").read_text(encoding="utf-8").splitlines()]
    n_correct = sum(a["correct"] for a in eval_set)
    assert eval_set and n_correct * 2 == len(eval_set)  # balanced 50/50

    h = summary["headline_j1_vs_j1_bob"]
    assert h["j1"]["n"] == h["j1_bob"]["n"] == h["comparison"]["n"] == len(eval_set)
    assert summary["bob_alone_vs_j1_bob"] is not None
    assert "Headline" in (tmp_path / "summary.md").read_text(encoding="utf-8")

    judgments = [json.loads(l) for l in (tmp_path / "judgments.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {j["condition"] for j in judgments} == {"j1", "j1_bob", "bob_alone"}
    # j1 and j1_bob see different prompts (Bob's notes), so the judge was really called twice
    assert client.calls.count("judge:j1") == client.calls.count("judge:j1_bob") == len(eval_set)


def test_cli_mock(tmp_path, capsys):
    run_mod.main(["run", "--config", str(ROOT / "configs" / "default.yaml"), "--csv", str(TOY),
                  "--out", str(tmp_path), "--mock"])
    assert "J1 + Bob v1" in capsys.readouterr().out


class RecordingMock(MockClient):
    """MockClient that also keeps each judge prompt, keyed by condition."""

    def __init__(self):
        super().__init__()
        self.prompts = {}

    def complete(self, *, tag, user, **kw):
        if tag.startswith("judge:"):
            self.prompts.setdefault(tag.split(":", 1)[1], []).append(user)
        return super().complete(tag=tag, user=user, **kw)


def test_controls_testimony_and_unconstrained_bob(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["run"]["out_dir"] = str(tmp_path)
    cfg["analysis"]["n_bootstrap"] = 200
    cfg["controls"].update(testimony=True, unconstrained_bob=True)
    client = RecordingMock()

    summary = run_mod.run(cfg, client, load_csv(TOY, seed=0))

    judgments = [json.loads(l) for l in (tmp_path / "judgments.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {j["condition"] for j in judgments} == {"j1", "j1_bob", "bob_alone", "j1_testimony", "j1_bob_free"}
    assert (tmp_path / "bob_free.jsonl").exists()

    # testimony carries Bob-alone's own verdict for that item and no notes block
    bob_alone = {j["aid"]: j for j in judgments if j["condition"] == "bob_alone"}
    assert all("judged the response's final option to be" in p for p in client.prompts["j1_testimony"])
    assert all("<assistant_notes>" not in p for p in client.prompts["j1_testimony"])
    verdict_words = {f"to be {j['verdict']}," for j in bob_alone.values()}
    assert all(any(w in p for w in verdict_words) for p in client.prompts["j1_testimony"])
    # unconstrained notes use their own block, which doesn't claim a verdict was forbidden
    assert all("including its own assessment" in p for p in client.prompts["j1_bob_free"])
    assert all("was not allowed to give a verdict" in p for p in client.prompts["j1_bob"])

    assert set(summary["controls"]) == {"j1_bob", "j1_testimony", "j1_bob_free"}
    assert "## Controls" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_controls_off_by_default(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["run"]["out_dir"] = str(tmp_path)
    cfg["analysis"]["n_bootstrap"] = 100
    summary = run_mod.run(cfg, MockClient(), load_csv(TOY, seed=0))
    assert summary["controls"] == {} and summary["bob_free"] is None
    assert not (tmp_path / "bob_free.jsonl").exists()


def test_subsample_by_question_keeps_whole_questions():
    from oversight.schema import AliceAnswer

    ev = [AliceAnswer(aid=f"q{q}#s{s}", qid=f"q{q}", sample=s, explanation="", option="A", correct=bool(s % 2))
          for q in range(20) for s in range(3)]
    sub = run_mod.subsample_by_question(ev, 10, seed=0)
    qids = {a.qid for a in sub}
    assert 10 <= len(sub) <= 12  # stops after the question that reaches n
    assert all(sum(a.qid == q for a in sub) == 3 for q in qids)  # whole questions only
    assert sub == run_mod.subsample_by_question(ev, 10, seed=0)  # deterministic
    assert {a.qid for a in run_mod.subsample_by_question(ev, 10, seed=1)} != qids


def test_screening_mode_skips_notes_condition(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["run"].update(out_dir=str(tmp_path), max_eval_items=6)
    cfg["analysis"]["n_bootstrap"] = 100
    cfg["controls"].update(testimony=True, j1_bob=False)
    summary = run_mod.run(cfg, MockClient(), load_csv(TOY, seed=0))
    judgments = [json.loads(l) for l in (tmp_path / "judgments.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {j["condition"] for j in judgments} == {"j1", "bob_alone", "j1_testimony"}
    assert summary["headline_j1_vs_j1_bob"] is None
    assert set(summary["controls"]) == {"j1_testimony"}
    assert len({j["aid"] for j in judgments}) <= 6 + 2


class SolveCountingMock(RecordingMock):
    """RecordingMock that also records the (user prompt, sample) of every bob_solve call."""

    def __init__(self):
        super().__init__()
        self.solves = []

    def complete(self, *, tag, user, sample=0, **kw):
        if tag == "bob_solve":
            self.solves.append((user, sample))
        return super().complete(tag=tag, user=user, sample=sample, **kw)


def test_solve_x4_and_its_testimony(tmp_path):
    from oversight import bob_solve

    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["run"]["out_dir"] = str(tmp_path)
    cfg["analysis"]["n_bootstrap"] = 100
    cfg["controls"].update(bob_solve_x4=True, testimony_x4=True)
    client = SolveCountingMock()

    summary = run_mod.run(cfg, client, load_csv(TOY, seed=0))

    eval_set = [json.loads(l) for l in (tmp_path / "eval_set.jsonl").read_text(encoding="utf-8").splitlines()]
    n_q = len({a["qid"] for a in eval_set})
    # one set of 4 solves per question, never per Alice sample, and Alice's answer is not in the prompt
    assert len(client.solves) == len(set(client.solves)) == 4 * n_q
    assert {s for _, s in client.solves} == {0, 1, 2, 3}
    assert not any("Final answer:" in u for u, _ in client.solves)

    judgments = [json.loads(l) for l in (tmp_path / "judgments.jsonl").read_text(encoding="utf-8").splitlines()]
    x4 = {j["aid"]: j for j in judgments if j["condition"] == "bob_solve_x4"}
    assert set(x4) == {a["aid"] for a in eval_set}
    for a in eval_set:
        j = x4[a["aid"]]
        votes = j["rationale"].removeprefix("votes ").split(",")
        assert j["p_correct"] == sum(v == a["option"] for v in votes) / len(votes)
        assert j["verdict"] == ("correct" if j["p_correct"] >= 0.5 else "incorrect")
    # testimony x4 carries that verdict and confidence
    assert len(client.prompts["j1_testimony_x4"]) == len(eval_set)
    assert all("judged the response's final option to be" in p for p in client.prompts["j1_testimony_x4"])

    assert {"bob_solve_x4", "j1_testimony_x4"} <= set(summary["controls"])
    assert summary["controls"]["j1_testimony_x4"]["vs_bob_solve_x4"] is not None
    pc = summary["per_condition"]["bob_solve_x4"]
    assert pc["n"] == len(eval_set) and len(pc["auroc_ci95"]) == 2
    assert bob_solve.verdict(
        run_mod.AliceAnswer(aid="q#s0", qid="q", sample=0, explanation="", option="B", correct=True),
        ["B", None, "A", "B"]).p_correct == 2 / 3


def test_majority_working_rules():
    from oversight.bob_solve import majority_working

    s = lambda o, r: {"final_option": o, "reasoning": r}  # noqa: E731
    assert majority_working([s("A", "a1"), s("B", "b1"), s("B", "b2"), s("C", "c1")]) == "b1"
    assert majority_working([s("A", "a1"), s("B", "b1"), s("B", "b2"), s("A", "a2")]) == "a1"  # tie: first valid
    assert majority_working([{}, s("Z", "bad"), s("C", "c1"), s("D", "d1")]) == "c1"  # tie of valid ones
    assert majority_working([{}, {}]) == ""


def test_x4_notes_and_leak_audit(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cfg["run"]["out_dir"] = str(tmp_path)
    cfg["analysis"]["n_bootstrap"] = 100
    cfg["controls"].update(bob_solve_x4=True, testimony_x4=True, notes_x4=True, leak_audit=True)
    client = SolveCountingMock()
    bob_prompts = []
    orig = client.complete

    def spy(*, tag, user, **kw):
        if tag == "bob":
            bob_prompts.append(user)
        return orig(tag=tag, user=user, **kw)

    client.complete = spy
    summary = run_mod.run(cfg, client, load_csv(TOY, seed=0))

    eval_set = [json.loads(l) for l in (tmp_path / "eval_set.jsonl").read_text(encoding="utf-8").splitlines()]
    n_q = len({a["qid"] for a in eval_set})
    assert len(client.solves) == 4 * n_q  # the notes reuse the x4 solves, no extra solve calls
    with_working = [p for p in bob_prompts if "<working>" in p]
    assert len(with_working) == len(eval_set) and all("mock working" in p for p in with_working)
    assert all("reveal your working's conclusion" in p for p in with_working)
    assert (tmp_path / "bob_x4.jsonl").exists() and (tmp_path / "leak_audit.jsonl").exists()

    judgments = [json.loads(l) for l in (tmp_path / "judgments.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(j["condition"] == "j1_bob_x4" for j in judgments) == len(eval_set)
    audits = [json.loads(l) for l in (tmp_path / "leak_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {a["condition"] for a in audits} == {"audit_v1", "audit_x4"} and len(audits) == 2 * len(eval_set)

    xn = summary["x4_notes"]
    assert xn["notes"]["accepted"] == len(eval_set)
    for key in ("vs_j1", "vs_testimony_x4", "vs_v1_notes", "pp_vs_j1"):
        assert xn[key]["comparison"]["n"] > 0, key
    la = summary["leak_audit"]
    assert set(la["bob_solve_x4"]) >= {"audit_x4", "audit_v1", "j1_no_notes", "x4_minus_v1"}
    assert "j1_bob_x4_itt" in summary["per_condition"]
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## Leak audit" in md and "## Verdict-free notes from the solve-x4 Bob" in md
