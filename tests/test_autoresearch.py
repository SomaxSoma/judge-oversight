"""The autoresearch scorer: split, budget rules, keep/discard bookkeeping, cache compatibility."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from oversight.client import Completion, MockClient, _CachedClient, _cache_key
from oversight.data.gpqa import load_csv

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "tests" / "fixtures" / "toy.csv"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "autoresearch" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


score = _load("score")
make_split = _load("make_split")


def test_split_is_deterministic_disjoint_and_complete():
    qids = [it.qid for it in load_csv(TOY)]
    s = make_split.make_split(qids, n_dev=7)
    assert s == make_split.make_split(list(reversed(qids)), n_dev=7)
    assert len(s["dev"]) == 7 and not set(s["dev"]) & set(s["test"])
    assert set(s["dev"]) | set(s["test"]) == set(qids)


def test_committed_split_covers_gpqa_diamond():
    s = json.loads((ROOT / "autoresearch" / "split.json").read_text(encoding="utf-8"))
    assert len(s["dev"]) == 120 and len(s["test"]) == 78 and not set(s["dev"]) & set(s["test"])


def test_auroc():
    assert score.auroc([True, False], [0.9, 0.1]) == 1.0
    assert score.auroc([True, False], [0.1, 0.9]) == 0.0
    assert score.auroc([True, False], [0.5, 0.5]) == 0.5
    assert score.auroc([True, True], [0.1, 0.9]) == 0.5  # one class only


ACFG = {"max_calls_per_item": 2, "max_tokens_cap": 4096, "allowed_extra": ["think"]}


def test_bob_client_enforces_rules():
    c = score.BobClient(MockClient(), "bob-model", ACFG)
    kw = dict(tag="bob", system="s", user="u", max_tokens=100)
    with pytest.raises(score.BobRuleError, match="only call"):
        c.complete(model="judge-model", **kw)
    with pytest.raises(score.BobRuleError, match="cap"):
        c.complete(model="bob-model", **(kw | {"max_tokens": 99999}))
    with pytest.raises(score.BobRuleError, match="not allowed"):
        c.complete(model="bob-model", extra={"num_ctx": 1}, **kw)
    c.complete(model="bob-model", extra={"think": True}, **kw)
    c.complete(model="bob-model", **kw)
    with pytest.raises(score.BobRuleError, match="more than 2"):
        c.complete(model="bob-model", **kw)
    leak_system = score.prompts.load("leak_check")[0]
    with pytest.raises(score.BobRuleError, match="leak checker"):
        score.BobClient(MockClient(), "bob-model", ACFG).complete(model="bob-model", tag="x", system=leak_system, user="u", max_tokens=100)


class _FakeBackend(_CachedClient):
    provider = "fake"
    supports_extra = True

    def _call(self, model, system, user, schema, effort, max_tokens, sample, extra=None):
        return Completion(text=json.dumps(extra), stop_reason="end_turn")


def test_cache_keys_without_extra_are_unchanged(tmp_path):
    c = _FakeBackend(tmp_path)
    c.complete(tag="t", model="m", system="s", user="u")
    legacy = _cache_key(provider="fake", model="m", system="s", user="u", schema=None, effort=None,
                        max_tokens=8000, sample=0)
    assert (tmp_path / legacy[:2] / f"{legacy}.json").exists()
    assert c.complete(tag="t", model="m", system="s", user="u", extra={"think": True}).text == '{"think": true}'
    assert len(list(tmp_path.rglob("*.json"))) == 2


def _run(tmp_path, *extra):
    split = tmp_path / "split.json"
    if not split.exists():
        split.write_text(json.dumps(make_split.make_split([it.qid for it in load_csv(TOY)], n_dev=7)))
    score.main(["--mock", "--no-git", "--csv", str(TOY), "--split-file", str(split), "--results-dir", str(tmp_path),
                "--dev-questions", "7", *extra])
    with open(tmp_path / "results.tsv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_loop_bookkeeping(tmp_path, capsys):
    with pytest.raises(SystemExit, match="stage-0 run"):
        _run(tmp_path, "--stage", "1", "--desc", "too early")

    rows = _run(tmp_path, "--stage", "0", "--desc", "baseline")
    assert rows[-1]["status"] == "baseline" and rows[-1]["split"] == "dev7"
    assert "decision:    baseline" in capsys.readouterr().out

    rows = _run(tmp_path, "--stage", "0", "--desc", "same again")
    assert rows[-1]["status"] == "discard" and rows[-1]["p_improve"] == "0.500"  # identical predictions

    rows = _run(tmp_path, "--stage", "1", "--desc", "baseline notes")
    assert rows[-1]["status"] == "baseline" and rows[-1]["ref"] == "nogit"
    assert float(rows[-1]["leak_drop"]) == 0.0
    assert (tmp_path / "runs" / "stage1_dev7_nogit_notes.jsonl").exists()

    rows = _run(tmp_path, "--stage", "1", "--log-crash", "--desc", "broke it")
    assert rows[-1]["status"] == "crash"


def test_test_split_is_locked(tmp_path):
    with pytest.raises(SystemExit, match="locked"):
        _run(tmp_path, "--stage", "0", "--split", "test", "--desc", "peek")
