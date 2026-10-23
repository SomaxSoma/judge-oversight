"""Pipeline: Alice answers -> balance -> Bob v1 notes -> judge in each condition -> metrics.

Conditions (all on the same balanced items):
  j1            blind judge, no Bob                                  (baseline)
  j1_bob        the same judge plus Bob's notes                      (Bob v1)
  bob_alone     Bob's model judging directly                         (control)
  j1_testimony  the judge plus only Bob-alone's verdict + confidence (control: pure deference)
  j1_bob_free   the judge plus notes from a Bob allowed a verdict    (control: unconstrained Bob)
  bob_solve_x4  Bob solves each question 4x without seeing Alice; p = share of solves matching her option
  j1_testimony_x4  the judge plus only bob_solve_x4's verdict + confidence
  j1_bob_x4     the judge plus verdict-free v1 notes written with Bob's private x4 working in view
  *_itt         intention-to-treat: an item without notes or checks keeps its J1 verdict
Leak audit (evaluation only, leak_audit.py): the judge's model guesses Bob's private x4 verdict from the notes.
  checks_bob_*  Bob writes k checks; the judge answers each blind, then decides (_judge) or a rule decides (_rule)
  checks_self_* the same, but the judge writes its own checks        (control: decomposition without Bob)

Every API call is cached, so a crashed or interrupted run resumes when re-run.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from oversight import alice, bob, bob_solve, checks, leak_audit, metrics, prompts
from oversight.client import AnthropicClient, Client, MockClient, OllamaClient, OpenAICompatClient
from oversight.data.gpqa import load_items
from oversight.judge import judge
from oversight.schema import AliceAnswer, BobMessage, ChecksMessage, Item, JudgeVerdict

CHECK_CONDITIONS = ("checks_bob_rule", "checks_bob_judge", "checks_self_rule", "checks_self_judge")
X4_CONDITIONS = ("bob_solve_x4", "j1_testimony_x4", "j1_bob_x4")
CONDITIONS = ("j1", "j1_bob", "bob_alone", "j1_testimony", "j1_bob_free") + CHECK_CONDITIONS + X4_CONDITIONS
NOTES_ITT = ("j1_bob", "j1_bob_x4")
CHECK_ITT = tuple(f"{c}_itt" for c in CHECK_CONDITIONS)
ITT_CONDITIONS = CHECK_ITT + tuple(f"{c}_itt" for c in NOTES_ITT)
CONTROL_CONDITIONS = ("j1_testimony", "j1_bob_free") + CHECK_CONDITIONS + CHECK_ITT + X4_CONDITIONS + ("j1_bob_x4_itt",)


def subsample_by_question(eval_set: list[AliceAnswer], n: int, seed: int) -> list[AliceAnswer]:
    """About n items from the eval set, taking whole questions in a seeded random order, so the subset
    reuses every cached call and keeps samples of one question together."""
    qids = sorted({a.qid for a in eval_set})
    random.Random(f"subsample-{seed}").shuffle(qids)
    chosen, count = set(), 0
    per_q = Counter(a.qid for a in eval_set)
    for q in qids:
        if count >= n:
            break
        chosen.add(q)
        count += per_q[q]
    return [a for a in eval_set if a.qid in chosen]


def run_checks(client, judge_cfg, by_qid, by_aid, msgs: list[ChecksMessage], workers: int) -> list[JudgeVerdict]:
    """The judge answers every check blind, then one final call per item gives p_agree and its verdict."""
    jobs = [(m, c) for m in msgs for c in m.checks]

    def answer(job):
        msg, c = job
        c.judge_answer, c.judge_confidence = checks.answer_check(client, judge_cfg, by_qid[by_aid[msg.aid].qid], c)

    pmap(answer, jobs, workers, "check_answer")
    out = []
    for pair in pmap(lambda msg: checks.final_verdicts(client, judge_cfg, by_qid[by_aid[msg.aid].qid],
                                                       by_aid[msg.aid], msg), msgs, workers, "checks_final"):
        out += list(pair)
    return out


def testimony_block(v: JudgeVerdict) -> str:
    return prompts.fragment("judge_testimony_block").format(verdict=v.verdict, confidence=f"{100 * v.p_correct:.0f}%")


def run_by_model(tasks: list, workers: int, fn: Callable) -> list:
    """Runs tasks grouped by model: a local GPU holds one model, and interleaving would reload it constantly."""
    out = []
    for model in dict.fromkeys(t["cfg"]["model"] for t in tasks):
        group = [t for t in tasks if t["cfg"]["model"] == model]
        out += pmap(fn, group, workers, f"judge[{model}]")
    return out


def pmap(fn: Callable, xs: Iterable, workers: int, label: str) -> list:
    xs = list(xs)
    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(fn, xs), 1):
            out.append(r)
            if i % 25 == 0 or i == len(xs):
                print(f"  {label}: {i}/{len(xs)}", file=sys.stderr)
    return out


def write_jsonl(path: Path, rows: Iterable) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")


def run(cfg: dict, client: Client, items: list[Item]) -> dict:
    out_dir = Path(cfg["run"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = cfg["run"].get("workers", 8)
    m = cfg["models"]
    by_qid = {it.qid: it for it in items}

    print(f"Alice: {len(items)} questions x {cfg['alice']['samples_per_question']} samples", file=sys.stderr)
    jobs = [(it, s) for it in items for s in range(cfg["alice"]["samples_per_question"])]
    answers: list[AliceAnswer] = pmap(lambda j: alice.answer(client, m["alice"], *j), jobs, workers, "alice")
    write_jsonl(out_dir / "alice.jsonl", answers)

    if cfg["alice"].get("balance", True):
        eval_set = alice.balance(answers, cfg["data"].get("seed", 0))
    else:
        eval_set = [a for a in answers if a.correct is not None]
    if not eval_set:
        raise SystemExit("No usable items after balancing: Alice got everything right or wrong.")
    if cfg["run"].get("max_eval_items"):
        eval_set = subsample_by_question(eval_set, cfg["run"]["max_eval_items"], cfg["data"].get("seed", 0))
    write_jsonl(out_dir / "eval_set.jsonl", eval_set)

    bcfg = cfg["bob"]
    bob_msgs: list[BobMessage] = pmap(
        lambda a: bob.critique(
            client, m["bob"], m["leak_checker"], by_qid[a.qid], a,
            max_regenerations=bcfg.get("max_regenerations", 1),
            use_llm_check=bcfg.get("use_llm_leak_check", True),
        ),
        eval_set, workers, "bob",
    )
    write_jsonl(out_dir / "bob.jsonl", bob_msgs)
    notes = {b.aid: b.text for b in bob_msgs}

    controls = cfg.get("controls", {})
    free_msgs: list[BobMessage] = []
    if controls.get("unconstrained_bob"):
        free_msgs = pmap(lambda a: bob.free_critique(client, m["bob"], by_qid[a.qid], a), eval_set, workers, "bob_free")
        write_jsonl(out_dir / "bob_free.jsonl", free_msgs)
    free_notes = {b.aid: b.text for b in free_msgs}

    def do(t: dict) -> JudgeVerdict:
        return judge(client, t["cfg"], by_qid[t["a"].qid], t["a"], t["cond"], bob_notes=t.get("notes"),
                     notes_block_name=t.get("block", "judge_notes_block"), context_block=t.get("context"))

    # Bob's own verdicts come first: testimony is built from them.
    verdicts: list[JudgeVerdict] = []
    if controls.get("bob_alone", True) or controls.get("testimony"):
        verdicts += run_by_model([{"a": a, "cond": "bob_alone", "cfg": m["bob"]} for a in eval_set], workers, do)
    bob_alone = {j.aid: j for j in verdicts if j.verdict is not None}

    # Solve x4: Bob answers each question N_SOLVES times without Alice, once per question.
    x4: dict[str, JudgeVerdict] = {}
    x4_msgs: list[BobMessage] = []
    if controls.get("bob_solve_x4") or controls.get("testimony_x4") or controls.get("notes_x4"):
        qids = list(dict.fromkeys(a.qid for a in eval_set))
        jobs = [(q, s) for q in qids for s in range(bob_solve.N_SOLVES)]
        solves = dict(zip(jobs, pmap(lambda j: bob_solve.solve_full(client, m["bob"], by_qid[j[0]], j[1]),
                                     jobs, workers, "bob_solve")))
        per_q = {q: [solves[(q, s)] for s in range(bob_solve.N_SOLVES)] for q in qids}
        x4_verdicts = [bob_solve.verdict(a, [bob_solve.option_of(x) for x in per_q[a.qid]]) for a in eval_set]
        verdicts += x4_verdicts
        x4 = {j.aid: j for j in x4_verdicts if j.verdict is not None}
        if controls.get("notes_x4"):
            working = {q: bob_solve.majority_working(per_q[q]) for q in qids}
            x4_msgs = pmap(
                lambda a: bob.critique(
                    client, m["bob"], m["leak_checker"], by_qid[a.qid], a,
                    max_regenerations=bcfg.get("max_regenerations", 1),
                    use_llm_check=bcfg.get("use_llm_leak_check", True),
                    working=working[a.qid],
                ),
                eval_set, workers, "bob_x4_notes",
            )
            write_jsonl(out_dir / "bob_x4.jsonl", x4_msgs)
    x4_notes = {b.aid: b.text for b in x4_msgs}

    # Checks written by Bob (his model's phase); the judge's own checks come in the judge's phase.
    ccfg = cfg.get("checks", {})
    k = ccfg.get("k", 3)
    check_msgs: list[ChecksMessage] = []
    if controls.get("checks_bob"):
        check_msgs += pmap(
            lambda a: checks.write_checks(client, m["bob"], by_qid[a.qid], a, "bob", k=k,
                                          leak_cfg=m.get("checks_leak_checker", m["leak_checker"]),
                                          max_regenerations=ccfg.get("max_regenerations", 2)),
            eval_set, workers, "checks_write[bob]",
        )

    tasks = [{"a": a, "cond": "j1", "cfg": m["judge"]} for a in eval_set]
    if controls.get("j1_bob", True):
        tasks += [{"a": a, "cond": "j1_bob", "cfg": m["judge"], "notes": notes[a.aid]}
                  for a in eval_set if notes[a.aid]]
    if controls.get("testimony"):
        tasks += [{"a": a, "cond": "j1_testimony", "cfg": m["judge"], "context": testimony_block(bob_alone[a.aid])}
                  for a in eval_set if a.aid in bob_alone]
    if controls.get("testimony_x4"):
        tasks += [{"a": a, "cond": "j1_testimony_x4", "cfg": m["judge"], "context": testimony_block(x4[a.aid])}
                  for a in eval_set if a.aid in x4]
    if x4_msgs:
        tasks += [{"a": a, "cond": "j1_bob_x4", "cfg": m["judge"], "notes": x4_notes[a.aid]}
                  for a in eval_set if x4_notes.get(a.aid)]
    if free_msgs:
        tasks += [{"a": a, "cond": "j1_bob_free", "cfg": m["judge"], "notes": free_notes[a.aid],
                   "block": "judge_free_notes_block"} for a in eval_set if free_notes.get(a.aid)]
    verdicts += run_by_model(tasks, workers, do)

    # Leak audit: the judge's model guesses Bob's private verdict from each set of accepted notes.
    audits: list[JudgeVerdict] = []
    if controls.get("leak_audit"):
        jobs = [(a, txt, cond) for a in eval_set
                for txt, cond in ((notes.get(a.aid), "audit_v1"), (x4_notes.get(a.aid), "audit_x4")) if txt]
        audits = pmap(lambda j: leak_audit.audit(client, m["judge"], by_qid[j[0].qid], j[0], j[1], j[2]),
                      jobs, workers, "leak_audit")
        write_jsonl(out_dir / "leak_audit.jsonl", audits)

    if controls.get("checks_self"):
        check_msgs += pmap(lambda a: checks.write_checks(client, m["judge"], by_qid[a.qid], a, "self", k=k),
                           eval_set, workers, "checks_write[self]")
    if check_msgs:
        verdicts += run_checks(client, m["judge"], by_qid, {a.aid: a for a in eval_set}, check_msgs, workers)
        write_jsonl(out_dir / "checks.jsonl", check_msgs)
    write_jsonl(out_dir / "judgments.jsonl", verdicts)

    summary = analyze(answers, eval_set, bob_msgs, verdicts, cfg["analysis"], free_msgs, check_msgs, x4_msgs, audits)
    summary["config"] = cfg
    summary["usage"] = getattr(client, "usage", {})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def notes_stats(msgs) -> dict:
    accepted = [b for b in msgs if b.text]
    return {
        "messages": len(msgs),
        "accepted": len(accepted),
        "dropped": len(msgs) - len(accepted),
        "drop_rate": metrics._rate(len(msgs) - len(accepted), len(msgs)),
        "regenerated": sum(b.attempts > 1 for b in msgs),
        "final_flags": dict(Counter(f for b in msgs for f in b.flags)),
        "mean_words": metrics._rate(sum(b.words for b in accepted), len(accepted)),
    }


def analyze(answers, eval_set, bob_msgs, verdicts, acfg, free_msgs=(), check_msgs=(), x4_msgs=(), audits=()) -> dict:
    n_boot, seed = acfg.get("n_bootstrap", 2000), acfg.get("seed", 0)
    label = {a.aid: a.correct for a in eval_set}
    qid = {a.aid: a.qid for a in eval_set}
    v = {c: {} for c in CONDITIONS + ITT_CONDITIONS}
    for j in verdicts:
        if j.verdict is not None:
            v[j.condition][j.aid] = j
    # Intention-to-treat: an item whose checks were dropped (leak filter) or failed keeps its J1 verdict, so
    # the comparison covers every item and the leak filter cannot select an easier subset.
    for cond in CHECK_CONDITIONS + NOTES_ITT:
        if v[cond]:
            v[f"{cond}_itt"] = {a: v[cond].get(a) or v["j1"][a] for a in v["j1"]}

    def cond_summary(cond: str, aids: list[str], with_auroc_ci: bool = False) -> dict | None:
        if not aids:
            return None
        labels = [label[a] for a in aids]
        preds = [v[cond][a].verdict == "correct" for a in aids]
        s = metrics.summarize(labels, preds, [v[cond][a].p_correct for a in aids])
        lo, hi = metrics.cluster_bootstrap_ci(
            lambda idx: metrics.accuracy([labels[i] for i in idx], [preds[i] for i in idx]),
            [qid[a] for a in aids], n_boot, seed,
        )
        s["accuracy_ci95"] = [lo, hi]
        s["auroc"] = metrics.auroc(labels, [v[cond][a].p_correct for a in aids])
        if with_auroc_ci:
            probs = [v[cond][a].p_correct for a in aids]
            s["auroc_ci95"] = list(metrics.cluster_bootstrap_ci(
                lambda idx: metrics.auroc([labels[i] for i in idx], [probs[i] for i in idx]) or 0.5,
                [qid[a] for a in aids], n_boot, seed,
            ))
        return s

    def paired(cond_a: str, cond_b: str) -> dict | None:
        aids = [a for a in label if a in v[cond_a] and a in v[cond_b]]
        if not aids:
            return None
        labels = [label[a] for a in aids]
        return {
            cond_a: cond_summary(cond_a, aids),
            cond_b: cond_summary(cond_b, aids),
            "comparison": metrics.paired_comparison(
                labels,
                [v[cond_a][a].verdict == "correct" for a in aids],
                [v[cond_b][a].verdict == "correct" for a in aids],
                [qid[a] for a in aids], n_boot, seed,
            ),
            "auroc_comparison": metrics.paired_auroc(
                labels, [v[cond_a][a].p_correct for a in aids], [v[cond_b][a].p_correct for a in aids],
                [qid[a] for a in aids], n_boot, seed,
            ),
        }

    def checks_stats(author: str) -> dict | None:
        msgs = [mm for mm in check_msgs if mm.author == author]
        if not msgs:
            return None
        kept = [mm for mm in msgs if mm.checks]
        rows = [(label[mm.aid], c) for mm in kept for c in mm.checks if mm.aid in label]
        agree = lambda want: [c.p_agree for l, c in rows if l == want and c.p_agree is not None]  # noqa: E731
        right, wrong = agree(True), agree(False)
        return {
            "items": len(msgs),
            "with_checks": len(kept),
            "dropped": len(msgs) - len(kept),
            "flags": dict(Counter(f for mm in msgs for f in mm.flags)),
            "regenerated": sum(mm.attempts > 1 for mm in msgs),
            "mean_checks": metrics._rate(sum(len(mm.checks) for mm in kept), len(kept)),
            "blind_answer_failures": sum(c.judge_answer is None for _, c in rows),
            # The verifiability frontier: do the judge's blind answers agree with Alice's values more when she is right?
            "mean_p_agree_alice_right": metrics._rate(sum(right), len(right)),
            "mean_p_agree_alice_wrong": metrics._rate(sum(wrong), len(wrong)),
        }

    def audit_summary(audits) -> dict:
        """AUROC of the auditor's guess against Bob's actual private verdicts (x4, and v1 Bob alone)."""
        au = {c: {j.aid: j for j in audits if j.condition == c and j.p_correct is not None}
              for c in ("audit_v1", "audit_x4")}
        out = {}
        for target in ("bob_solve_x4", "bob_alone"):
            row = {}
            for c, preds in list(au.items()) + [("j1_no_notes", v["j1"])]:
                aids = [a for a in label if a in preds and a in v[target]]
                if not aids:
                    continue
                y = [v[target][a].verdict == "correct" for a in aids]
                ps = [preds[a].p_correct for a in aids]
                lo, hi = metrics.cluster_bootstrap_ci(
                    lambda idx: metrics.auroc([y[i] for i in idx], [ps[i] for i in idx]) or 0.5,
                    [qid[a] for a in aids], n_boot, seed)
                agree = metrics._rate(sum((preds[a].verdict == "correct") == yy for a, yy in zip(aids, y)), len(aids))
                row[c] = {"n": len(aids), "auroc": metrics.auroc(y, ps), "auroc_ci95": [lo, hi], "agreement": agree}
            both = [a for a in label if a in au["audit_v1"] and a in au["audit_x4"] and a in v[target]]
            if both:
                row["x4_minus_v1"] = metrics.paired_auroc(
                    [v[target][a].verdict == "correct" for a in both], [au["audit_v1"][a].p_correct for a in both],
                    [au["audit_x4"][a].p_correct for a in both], [qid[a] for a in both], n_boot, seed)
                row["x4_minus_v1"]["n"] = len(both)
            out[target] = row
        out["failures"] = sum(j.p_correct is None for j in audits)
        return out

    usable = [a for a in answers if a.correct is not None]
    flag_counts = Counter(f for b in bob_msgs for f in b.flags)
    accepted = [b for b in bob_msgs if b.text]
    return {
        "alice": {
            "samples": len(answers),
            "unusable": len(answers) - len(usable),
            "accuracy": metrics._rate(sum(a.correct for a in usable), len(usable)),
            "eval_set": len(eval_set),
            "eval_set_questions": len({a.qid for a in eval_set}),
        },
        "bob": {
            "messages": len(bob_msgs),
            "accepted": len(accepted),
            "dropped": len(bob_msgs) - len(accepted),
            "drop_rate": metrics._rate(len(bob_msgs) - len(accepted), len(bob_msgs)),
            "regenerated": sum(b.attempts > 1 for b in bob_msgs),
            "final_flags": dict(flag_counts),
            "mean_words": metrics._rate(sum(b.words for b in accepted), len(accepted)),
        },
        "judge_failures": {c: sum(1 for j in verdicts if j.condition == c and j.verdict is None) for c in CONDITIONS},
        "full_eval_set": {c: cond_summary(c, [a for a in label if a in v[c]]) for c in ("j1", "bob_alone")},
        # Every condition on all the items it has a verdict for, with question-clustered CIs.
        "per_condition": {c: cond_summary(c, [a for a in label if a in v[c]], with_auroc_ci=True)
                          for c in CONDITIONS + ITT_CONDITIONS if v[c]},
        # Every condition against J1 on their shared items (scripts/aggregate_channels.py reads this).
        "vs_j1": {c: paired("j1", c) for c in CONDITIONS[1:] + ITT_CONDITIONS if v[c]},
        "headline_j1_vs_j1_bob": paired("j1", "j1_bob"),
        "bob_alone_vs_j1_bob": paired("bob_alone", "j1_bob"),
        # Each control against the baseline and against Bob alone (the most the judge could gain).
        "controls": {
            c: {"vs_j1": paired("j1", c), "vs_bob_alone": paired("bob_alone", c),
                **({"vs_bob_solve_x4": paired("bob_solve_x4", c)} if v["bob_solve_x4"] and c != "bob_solve_x4" else {})}
            for c in ("j1_bob",) + CONTROL_CONDITIONS if v[c]
        } if any(v[c] for c in CONTROL_CONDITIONS) else {},
        "checks": {
            "bob": checks_stats("bob"),
            "self": checks_stats("self"),
            "bob_vs_self": {kind: paired(f"checks_self_{kind}", f"checks_bob_{kind}")
                            for kind in ("rule", "judge", "rule_itt", "judge_itt")}
            if v["checks_bob_rule"] and v["checks_self_rule"] else None,
        } if check_msgs else None,
        "x4_notes": {
            "notes": notes_stats(x4_msgs),
            "v1_notes": notes_stats(bob_msgs),
            # Primary (ITT) and per-protocol (items with notes) comparisons.
            "vs_j1": paired("j1", "j1_bob_x4_itt"),
            "vs_testimony_x4": paired("j1_testimony_x4", "j1_bob_x4_itt"),
            "vs_v1_notes": paired("j1_bob_itt", "j1_bob_x4_itt"),
            "v1_notes_vs_j1": paired("j1", "j1_bob_itt"),
            "pp_vs_j1": paired("j1", "j1_bob_x4"),
            "pp_vs_v1_notes": paired("j1_bob", "j1_bob_x4"),
        } if x4_msgs else None,
        "leak_audit": audit_summary(audits) if audits else None,
        "bob_free": {
            "messages": len(free_msgs),
            "with_assessment_line": sum(bool(b.text and "assessment:" in b.text.lower()) for b in free_msgs),
            "mean_words": metrics._rate(sum(b.words for b in free_msgs), len(free_msgs)),
        } if free_msgs else None,
    }


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def _ci(ci) -> str:
    return f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def _auc(x) -> str:
    return "–" if x is None else f"{x:.3f}"


def _dauc(r) -> str:
    return f"{r['delta_auroc']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"


def render_markdown(s: dict) -> str:
    lines = ["# Judge accuracy: baseline (J1) vs Bob v1 (J1+Bob)", ""]
    a, b = s["alice"], s["bob"]
    lines += [
        f"Alice: {a['samples']} samples, accuracy {_pct(a['accuracy'])}, {a['unusable']} unusable. "
        f"Balanced eval set: {a['eval_set']} answers from {a['eval_set_questions']} questions (50% correct).",
        f"Bob: {b['accepted']}/{b['messages']} notes accepted, {b['dropped']} dropped for leaks or refusals "
        f"({_pct(b['drop_rate'])}), {b['regenerated']} regenerated, mean {b['mean_words'] or 0:.0f} words. "
        f"Final-attempt flags: {b['final_flags'] or 'none'}.",
        "",
    ]
    h = s["headline_j1_vs_j1_bob"]
    if h:
        c = h["comparison"]
        lines += [
            f"## Headline (paired, n = {c['n']})",
            "",
            "| Condition | Accuracy (95% CI) | Balanced acc. | False accept | False reject | Brier |",
            "|---|---|---|---|---|---|",
        ]
        for name, key in (("J1 (no Bob)", "j1"), ("J1 + Bob v1", "j1_bob")):
            r = h[key]
            lines.append(
                f"| {name} | {_pct(r['accuracy'])} {_ci(r['accuracy_ci95'])} | {_pct(r['balanced_accuracy'])} | "
                f"{_pct(r['false_accept_rate'])} | {_pct(r['false_reject_rate'])} | {r['brier']:.3f} |"
            )
        f = c["flips"]
        lines += [
            "",
            f"**Δ accuracy (J1+Bob − J1): {100 * c['delta_accuracy']:+.1f} pts**, 95% CI {_ci(c['ci95'])}, "
            f"McNemar p = {c['mcnemar_p']:.3g}.",
            f"Flips: {f['wrong_to_right']} wrong→right, {f['right_to_wrong']} right→wrong, "
            f"{f['both_right']} both right, {f['both_wrong']} both wrong.",
            "",
        ]
    ba = s["bob_alone_vs_j1_bob"]
    if ba:
        c = ba["comparison"]
        lines += [
            "## Control: does J1+Bob beat Bob judging alone?",
            "",
            f"Bob alone {_pct(ba['bob_alone']['accuracy'])} vs J1+Bob {_pct(ba['j1_bob']['accuracy'])} "
            f"(n = {c['n']}); Δ (J1+Bob − Bob alone) {100 * c['delta_accuracy']:+.1f} pts, 95% CI {_ci(c['ci95'])}, "
            f"McNemar p = {c['mcnemar_p']:.3g}.",
            "",
        ]
    if s.get("controls"):
        names = {"j1_bob": "J1 + Bob v1 notes (no verdict)", "j1_testimony": "J1 + Bob's verdict only (testimony)",
                 "j1_bob_free": "J1 + unconstrained Bob notes",
                 "checks_bob_rule": "Bob's checks, rule", "checks_bob_judge": "Bob's checks, judge decides",
                 "checks_self_rule": "Judge's own checks, rule", "checks_self_judge": "Judge's own checks, judge decides",
                 "bob_solve_x4": "Bob alone, solve x4 (Δ vs Bob alone = x4 − v1)",
                 "j1_testimony_x4": "J1 + solve-x4 verdict only (testimony x4)",
                 "j1_bob_x4": "J1 + x4-working notes (no verdict)"}
        names.update({f"{c}_itt": f"{names[c]} (ITT: J1 if no checks)" for c in CHECK_CONDITIONS})
        names.update({f"{c}_itt": f"{names[c]} (ITT: J1 if dropped)" for c in NOTES_ITT})
        lines += [
            "## Controls: how much of Bob's edge reaches the judge?",
            "",
            "| Condition | n | Accuracy | Δ vs J1 (95% CI, p) | Δ vs Bob alone (95% CI, p) | AUROC | ΔAUROC vs J1 (95% CI) | False accept | False reject |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for cond, r in s["controls"].items():
            a1, a2 = r["vs_j1"], r["vs_bob_alone"]
            c1, c2 = a1["comparison"], a2["comparison"] if a2 else None
            me = a1[cond]
            d2 = (f"{100 * c2['delta_accuracy']:+.1f} {_ci(c2['ci95'])}, p = {c2['mcnemar_p']:.3g}") if c2 else "–"
            lines.append(
                f"| {names[cond]} | {c1['n']} | {_pct(me['accuracy'])} | "
                f"{100 * c1['delta_accuracy']:+.1f} {_ci(c1['ci95'])}, p = {c1['mcnemar_p']:.3g} | {d2} | "
                f"{_auc(me.get('auroc'))} | {_dauc(a1['auroc_comparison'])} | "
                f"{_pct(me['false_accept_rate'])} | {_pct(me['false_reject_rate'])} |"
            )
        for cond, r in s["controls"].items():
            r3 = r.get("vs_bob_solve_x4")
            if r3:
                c3 = r3["comparison"]
                lines.append(f"- {names[cond]} vs Bob alone solve x4 (n = {c3['n']}): Δ accuracy "
                             f"{100 * c3['delta_accuracy']:+.1f} pts {_ci(c3['ci95'])}, p = {c3['mcnemar_p']:.3g}; "
                             f"Δ AUROC {_dauc(r3['auroc_comparison'])}.")
        if s.get("bob_free"):
            bf = s["bob_free"]
            lines += ["", f"Unconstrained Bob: {bf['messages']} notes, {bf['with_assessment_line']} with an "
                          f"'Assessment:' line, mean {bf['mean_words'] or 0:.0f} words."]
        lines.append("")
    if s.get("checks"):
        ck = s["checks"]
        lines += ["## Checks", ""]
        for author, name in (("bob", "Bob's checks"), ("self", "Judge's own checks")):
            st = ck.get(author)
            if st:
                lines.append(
                    f"- {name}: {st['with_checks']}/{st['items']} items with checks ({st['dropped']} dropped, "
                    f"{st['regenerated']} regenerated; flags {st['flags'] or 'none'}), mean {st['mean_checks'] or 0:.1f} checks. "
                    f"Mean p(agree) when Alice is right {_pct(st['mean_p_agree_alice_right'])} vs wrong "
                    f"{_pct(st['mean_p_agree_alice_wrong'])}; blind-answer failures {st['blind_answer_failures']}."
                )
        if ck.get("bob_vs_self"):
            for kind, r in ck["bob_vs_self"].items():
                if r:
                    c, au = r["comparison"], r["auroc_comparison"]
                    lines.append(
                        f"- Bob's vs judge's own checks ({kind}, n = {c['n']}): Δ accuracy {100 * c['delta_accuracy']:+.1f} pts "
                        f"{_ci(c['ci95'])}, McNemar p = {c['mcnemar_p']:.3g}; Δ AUROC {_dauc(au)}."
                    )
        lines.append("")
    xn = s.get("x4_notes")
    if xn:
        lines += ["## Verdict-free notes from the solve-x4 Bob", ""]
        for key, name in (("notes", "x4-working notes"), ("v1_notes", "v1 notes")):
            st = xn[key]
            lines.append(f"- {name}: {st['accepted']}/{st['messages']} accepted, {st['dropped']} dropped "
                         f"({_pct(st['drop_rate'])}), {st['regenerated']} regenerated, mean {st['mean_words'] or 0:.0f} words; "
                         f"final-attempt flags {st['final_flags'] or 'none'}.")
        for key, name in (("vs_j1", "x4 notes (ITT) − J1"), ("vs_testimony_x4", "x4 notes (ITT) − testimony x4"),
                          ("vs_v1_notes", "x4 notes (ITT) − v1 notes (ITT)"), ("v1_notes_vs_j1", "v1 notes (ITT) − J1"),
                          ("pp_vs_j1", "x4 notes (per protocol) − J1"),
                          ("pp_vs_v1_notes", "x4 notes − v1 notes (per protocol)")):
            r = xn.get(key)
            if r:
                c = r["comparison"]
                lines.append(f"- {name} (n = {c['n']}): Δ accuracy {100 * c['delta_accuracy']:+.1f} pts {_ci(c['ci95'])}, "
                             f"McNemar p = {c['mcnemar_p']:.3g}; Δ AUROC {_dauc(r['auroc_comparison'])}.")
        lines.append("")
    la = s.get("leak_audit")
    if la:
        lines += ["## Leak audit: can the judge's model read Bob's private verdict off the notes?", "",
                  "| Target verdict | Notes | n | AUROC (95% CI) | Agreement |", "|---|---|---|---|---|"]
        for target, tname in (("bob_solve_x4", "Bob x4"), ("bob_alone", "Bob v1 alone")):
            for c, cname in (("audit_x4", "x4-working notes"), ("audit_v1", "v1 notes"),
                             ("j1_no_notes", "none (J1's own p_correct)")):
                r = la.get(target, {}).get(c)
                if r:
                    lines.append(f"| {tname} | {cname} | {r['n']} | {_auc(r['auroc'])} "
                                 f"[{r['auroc_ci95'][0]:.3f}, {r['auroc_ci95'][1]:.3f}] | {_pct(r['agreement'])} |")
        for target, tname in (("bob_solve_x4", "Bob x4"), ("bob_alone", "Bob v1 alone")):
            d = la.get(target, {}).get("x4_minus_v1")
            if d:
                lines.append(f"\nAUROC x4 notes − v1 notes, target {tname} (n = {d['n']}): {_dauc(d)}.")
        lines += [f"\nAudit failures: {la['failures']}.", ""]
    if s.get("per_condition"):
        lines += ["## Every condition on its own items (question-clustered 95% CIs)", "",
                  "| Condition | n | Accuracy | AUROC |", "|---|---|---|---|"]
        for cond, r in s["per_condition"].items():
            if r:
                ci = r.get("auroc_ci95")
                au = f"{_auc(r['auroc'])} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci and r["auroc"] is not None else _auc(r["auroc"])
                lines.append(f"| {cond} | {r['n']} | {_pct(r['accuracy'])} {_ci(r['accuracy_ci95'])} | {au} |")
        lines.append("")
    full = s["full_eval_set"]
    if full.get("j1"):
        lines += [f"J1 on the full eval set (before dropping Bob-rejected items): {_pct(full['j1']['accuracy'])} "
                  f"(n = {full['j1']['n']}). Judge failures per condition: {s['judge_failures']}.", ""]
    if s.get("usage"):
        lines += ["## API usage (new calls this run; cached calls not counted)", ""]
        lines += [f"- {m}: {u['calls']} calls, {u['input_tokens']:,} in / {u['output_tokens']:,} out tokens"
                  for m, u in s["usage"].items()]
    return "\n".join(lines) + "\n"


def make_client(cfg: dict) -> Client:
    provider = cfg.get("provider", {"type": "anthropic"})
    if provider["type"] == "anthropic":
        return AnthropicClient(cfg["run"]["cache_dir"])
    if provider["type"] == "openai_compat":
        return OpenAICompatClient(cfg["run"]["cache_dir"], provider["base_url"], provider["api_key_env"])
    if provider["type"] == "ollama":
        return OllamaClient(
            cfg["run"]["cache_dir"],
            provider.get("base_url", "http://localhost:11434"),
            num_ctx=provider.get("num_ctx", 8192),
            model_options=provider.get("model_options"),
        )
    raise SystemExit(f"unknown provider type {provider['type']!r}")


def load_dotenv(path: str | Path = ".env") -> None:
    """Set KEY=VALUE lines from a git-ignored .env file, without overriding the real environment."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        key, sep, value = line.strip().partition("=")
        value = value.strip().strip("\"'")
        if sep and key and value and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value)


def prevent_sleep() -> None:
    """Ask Windows not to idle-sleep while this process runs (released automatically on exit).

    Uses the same per-process request media players make; no power settings are changed.
    A closed lid or a manual Sleep still sleeps.
    """
    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    p = argparse.ArgumentParser(prog="oversight")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the J1 vs J1+Bob experiment")
    r.add_argument("--config", default="configs/default.yaml")
    r.add_argument("--limit-questions", type=int, help="pilot on the first N questions")
    r.add_argument("--csv", help="use a GPQA-format csv instead of downloading GPQA-Diamond")
    r.add_argument("--out", help="results directory (overrides run.out_dir)")
    r.add_argument("--max-eval-items", type=int, help="pilot on a whole-question subset of the full eval set")
    r.add_argument("--mock", action="store_true", help="dry run with MockClient; no API calls")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.limit_questions:
        cfg["data"]["limit_questions"] = args.limit_questions
    if args.csv:
        cfg["data"].update(source="csv", csv_path=args.csv)
    if args.out:
        cfg["run"]["out_dir"] = args.out
    if args.max_eval_items:
        cfg["run"]["max_eval_items"] = args.max_eval_items

    client = MockClient() if args.mock else make_client(cfg)
    if not args.mock:
        prevent_sleep()
    summary = run(cfg, client, load_items(cfg["data"]))
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252; the report uses Δ and →
    print(render_markdown(summary))
    print(f"Results written to {cfg['run']['out_dir']}/", file=sys.stderr)


if __name__ == "__main__":
    main()
