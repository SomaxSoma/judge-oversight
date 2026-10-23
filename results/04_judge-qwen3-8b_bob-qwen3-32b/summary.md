# Judge accuracy: baseline (J1) vs Bob v1 (J1+Bob)

Alice: 594 samples, accuracy 31.6%, 21 unusable. Balanced eval set: 362 answers from 181 questions (50% correct).
Bob: 345/362 notes accepted, 17 dropped for leaks or refusals (4.7%), 35 regenerated, mean 152 words. Final-attempt flags: {'option_letter': 12, 'mentions_option_B': 4, 'mentions_option_C': 2, 'mentions_option_A': 1, 'mentions_option_D': 2}.

## Headline (paired, n = 342)

| Condition | Accuracy (95% CI) | Balanced acc. | False accept | False reject | Brier |
|---|---|---|---|---|---|
| J1 (no Bob) | 53.5% [47.9, 59.0] | 53.7% | 56.0% | 36.5% | 0.423 |
| J1 + Bob v1 | 52.9% [47.0, 58.8] | 52.3% | 20.0% | 75.4% | 0.344 |

**Δ accuracy (J1+Bob − J1): -0.6 pts**, 95% CI [-8.7, 7.4], McNemar p = 0.935.
Flips: 73 wrong→right, 75 right→wrong, 108 both right, 86 both wrong.

## Control: does J1+Bob beat Bob judging alone?

Bob alone 61.2% vs J1+Bob 52.8% (n = 343); Δ (J1+Bob − Bob alone) -8.5 pts, 95% CI [-15.2, -2.0], McNemar p = 0.00758.

## Controls: how much of Bob's edge reaches the judge?

| Condition | n | Accuracy | Δ vs J1 (95% CI, p) | Δ vs Bob alone (95% CI, p) | AUROC | ΔAUROC vs J1 (95% CI) | False accept | False reject |
|---|---|---|---|---|---|---|---|---|
| J1 + Bob v1 notes (no verdict) | 342 | 52.9% | -0.6 [-8.7, 7.4], p = 0.935 | -8.5 [-15.2, -2.0], p = 0.00758 | 0.541 | -0.002 [-0.068, +0.069] | 20.0% | 75.4% |
| J1 + Bob's verdict only (testimony) | 360 | 55.8% | +1.9 [-3.4, 7.5], p = 0.51 | -5.0 [-9.6, 0.0], p = 0.0474 | 0.586 | +0.038 [-0.010, +0.087] | 45.3% | 43.0% |

## Every condition on its own items (question-clustered 95% CIs)

| Condition | n | Accuracy | AUROC |
|---|---|---|---|
| j1 | 361 | 54.0% [48.2, 59.6] | 0.550 [0.491, 0.610] |
| j1_bob | 343 | 52.8% [46.9, 58.6] | 0.539 [0.473, 0.607] |
| bob_alone | 362 | 60.8% [55.5, 66.0] | 0.622 [0.557, 0.682] |
| j1_testimony | 361 | 55.7% [50.3, 61.3] | 0.583 [0.524, 0.645] |
| j1_bob_itt | 361 | 53.5% [48.0, 59.4] | 0.554 [0.491, 0.620] |

J1 on the full eval set (before dropping Bob-rejected items): 54.0% (n = 361). Judge failures per condition: {'j1': 1, 'j1_bob': 2, 'bob_alone': 0, 'j1_testimony': 1, 'j1_bob_free': 0, 'checks_bob_rule': 0, 'checks_bob_judge': 0, 'checks_self_rule': 0, 'checks_self_judge': 0, 'bob_solve_x4': 0, 'j1_testimony_x4': 0, 'j1_bob_x4': 0}.
