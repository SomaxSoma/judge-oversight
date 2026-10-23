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

## Every condition on its own items (question-clustered 95% CIs)

| Condition | n | Accuracy | AUROC |
|---|---|---|---|
| j1 | 357 | 50.4% [44.5, 56.2] | 0.471 [0.409, 0.537] |
| j1_bob | 345 | 53.9% [47.8, 60.1] | 0.532 [0.463, 0.603] |
| bob_alone | 362 | 60.8% [55.5, 66.0] | 0.622 [0.557, 0.682] |
| j1_testimony | 356 | 53.1% [47.5, 58.6] | 0.530 [0.464, 0.592] |
| j1_bob_itt | 357 | 53.8% [48.0, 59.6] | 0.531 [0.464, 0.598] |

J1 on the full eval set (before dropping Bob-rejected items): 50.4% (n = 357). Judge failures per condition: {'j1': 5, 'j1_bob': 0, 'bob_alone': 0, 'j1_testimony': 6, 'j1_bob_free': 0, 'checks_bob_rule': 0, 'checks_bob_judge': 0, 'checks_self_rule': 0, 'checks_self_judge': 0, 'bob_solve_x4': 0, 'j1_testimony_x4': 0, 'j1_bob_x4': 0}.
