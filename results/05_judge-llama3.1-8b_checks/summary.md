# Judge accuracy: baseline (J1) vs Bob v1 (J1+Bob)

Alice: 594 samples, accuracy 31.6%, 21 unusable. Balanced eval set: 362 answers from 181 questions (50% correct).
Bob: 345/362 notes accepted, 17 dropped for leaks or refusals (4.7%), 35 regenerated, mean 152 words. Final-attempt flags: {'option_letter': 12, 'mentions_option_B': 4, 'mentions_option_C': 2, 'mentions_option_A': 1, 'mentions_option_D': 2}.

## Headline (paired, n = 340)

| Condition | Accuracy (95% CI) | Balanced acc. | False accept | False reject | Brier |
|---|---|---|---|---|---|
| J1 (no Bob) | 50.6% [45.0, 56.5] | 50.5% | 46.6% | 52.4% | 0.388 |
| J1 + Bob v1 | 54.1% [47.9, 60.2] | 53.6% | 24.1% | 68.7% | 0.315 |

**Δ accuracy (J1+Bob − J1): +3.5 pts**, 95% CI [-3.5, 10.6], McNemar p = 0.319.
Flips: 67 wrong→right, 55 right→wrong, 117 both right, 101 both wrong.

## Control: does J1+Bob beat Bob judging alone?

Bob alone 61.4% vs J1+Bob 53.9% (n = 345); Δ (J1+Bob − Bob alone) -7.5 pts, 95% CI [-14.1, -0.6], McNemar p = 0.0232.

## Controls: how much of Bob's edge reaches the judge?

| Condition | n | Accuracy | Δ vs J1 (95% CI, p) | Δ vs Bob alone (95% CI, p) | AUROC | ΔAUROC vs J1 (95% CI) | False accept | False reject |
|---|---|---|---|---|---|---|---|---|
| J1 + Bob v1 notes (no verdict) | 340 | 54.1% | +3.5 [-3.5, 10.6], p = 0.319 | -7.5 [-14.1, -0.6], p = 0.0232 | 0.530 | +0.057 [-0.011, +0.127] | 24.1% | 68.7% |
| J1 + Bob's verdict only (testimony) | 351 | 53.3% | +2.8 [-2.6, 8.4], p = 0.373 | -7.6 [-15.1, -0.6], p = 0.0271 | 0.532 | +0.055 [-0.009, +0.121] | 54.5% | 38.9% |
| Bob's checks, rule | 225 | 50.2% | -0.4 [-10.9, 10.0], p = 1 | -13.6 [-22.1, -4.8], p = 0.00215 | 0.543 | +0.072 [-0.036, +0.178] | 22.7% | 75.7% |
| Bob's checks, judge decides | 250 | 55.6% | +4.4 [-3.4, 12.7], p = 0.324 | -9.4 [-17.2, -1.9], p = 0.0197 | 0.551 | +0.092 [+0.005, +0.181] | 56.3% | 32.3% |
| Judge's own checks, rule | 313 | 52.4% | +2.2 [-5.8, 10.4], p = 0.632 | -8.2 [-16.0, -0.3], p = 0.0382 | 0.536 | +0.055 [-0.036, +0.147] | 33.3% | 60.7% |
| Judge's own checks, judge decides | 346 | 52.0% | +1.4 [-4.8, 7.4], p = 0.714 | -9.4 [-16.1, -2.6], p = 0.00899 | 0.525 | +0.049 [-0.024, +0.124] | 62.9% | 32.7% |
| Bob's checks, rule (ITT: J1 if no checks) | 357 | 50.1% | -0.3 [-6.9, 6.4], p = 1 | -10.6 [-17.6, -4.0], p = 0.00167 | 0.512 | +0.041 [-0.045, +0.121] | 31.1% | 68.9% |
| Bob's checks, judge decides (ITT: J1 if no checks) | 357 | 53.5% | +3.1 [-2.6, 9.0], p = 0.324 | -7.3 [-13.9, -0.8], p = 0.0304 | 0.539 | +0.067 [+0.001, +0.133] | 52.8% | 40.1% |
| Judge's own checks, rule (ITT: J1 if no checks) | 357 | 52.4% | +2.0 [-5.6, 9.1], p = 0.632 | -8.4 [-15.6, -1.1], p = 0.0208 | 0.504 | +0.033 [-0.054, +0.119] | 33.9% | 61.6% |
| Judge's own checks, judge decides (ITT: J1 if no checks) | 357 | 51.8% | +1.4 [-4.7, 7.2], p = 0.714 | -9.0 [-15.9, -2.5], p = 0.0122 | 0.521 | +0.050 [-0.022, +0.120] | 61.7% | 34.5% |

## Checks

- Bob's checks: 263/362 items with checks (99 dropped, 225 regenerated; flags {'llm_verdict': 22, 'llm_answer': 7, 'ungrounded': 28, 'not_standalone': 35, 'mentions_option_C': 5, 'mentions_option_A': 4, 'vague_value': 7, 'mentions_option_B': 2, 'mentions_option_D': 4}), mean 3.0 checks. Mean p(agree) when Alice is right 52.2% vs wrong 49.8%; blind-answer failures 8.
- Judge's own checks: 362/362 items with checks (0 dropped, 0 regenerated; flags none), mean 3.0 checks. Mean p(agree) when Alice is right 64.0% vs wrong 60.6%; blind-answer failures 13.
- Bob's vs judge's own checks (rule, n = 205): Δ accuracy -3.4 pts [-12.1, 5.3], McNemar p = 0.5; Δ AUROC -0.005 [-0.107, +0.097].
- Bob's vs judge's own checks (judge, n = 250): Δ accuracy +4.8 pts [-2.4, 12.0], McNemar p = 0.235; Δ AUROC +0.036 [-0.048, +0.119].
- Bob's vs judge's own checks (rule_itt, n = 357): Δ accuracy -2.2 pts [-8.3, 4.3], McNemar p = 0.554; Δ AUROC +0.008 [-0.066, +0.086].
- Bob's vs judge's own checks (judge_itt, n = 357): Δ accuracy +1.7 pts [-4.4, 8.0], McNemar p = 0.645; Δ AUROC +0.018 [-0.048, +0.088].

## Every condition on its own items (question-clustered 95% CIs)

| Condition | n | Accuracy | AUROC |
|---|---|---|---|
| j1 | 357 | 50.4% [44.5, 56.2] | 0.471 [0.409, 0.537] |
| j1_bob | 345 | 53.9% [47.8, 60.1] | 0.532 [0.463, 0.603] |
| bob_alone | 362 | 60.8% [55.5, 66.0] | 0.622 [0.557, 0.682] |
| j1_testimony | 356 | 53.1% [47.5, 58.6] | 0.530 [0.464, 0.592] |
| checks_bob_rule | 228 | 50.0% [42.9, 57.2] | 0.541 [0.465, 0.614] |
| checks_bob_judge | 254 | 55.1% [48.1, 61.7] | 0.544 [0.474, 0.614] |
| checks_self_rule | 317 | 52.1% [46.2, 58.1] | 0.533 [0.468, 0.595] |
| checks_self_judge | 351 | 51.6% [45.8, 57.0] | 0.521 [0.454, 0.586] |
| checks_bob_rule_itt | 357 | 50.1% [45.0, 55.6] | 0.512 [0.449, 0.577] |
| checks_bob_judge_itt | 357 | 53.5% [47.7, 59.2] | 0.539 [0.480, 0.601] |
| checks_self_rule_itt | 357 | 52.4% [47.0, 57.9] | 0.504 [0.443, 0.563] |
| checks_self_judge_itt | 357 | 51.8% [46.5, 57.1] | 0.521 [0.458, 0.584] |
| j1_bob_itt | 357 | 53.8% [48.0, 59.6] | 0.531 [0.464, 0.598] |

J1 on the full eval set (before dropping Bob-rejected items): 50.4% (n = 357). Judge failures per condition: {'j1': 5, 'j1_bob': 0, 'bob_alone': 0, 'j1_testimony': 6, 'j1_bob_free': 0, 'checks_bob_rule': 134, 'checks_bob_judge': 108, 'checks_self_rule': 45, 'checks_self_judge': 11, 'bob_solve_x4': 0, 'j1_testimony_x4': 0, 'j1_bob_x4': 0}.
