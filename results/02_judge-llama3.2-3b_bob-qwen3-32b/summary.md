# Judge accuracy: baseline (J1) vs Bob v1 (J1+Bob)

Alice: 594 samples, accuracy 31.6%, 21 unusable. Balanced eval set: 362 answers from 181 questions (50% correct).
Bob: 345/362 notes accepted, 17 dropped for leaks or refusals (4.7%), 35 regenerated, mean 152 words. Final-attempt flags: {'option_letter': 12, 'mentions_option_B': 4, 'mentions_option_C': 2, 'mentions_option_A': 1, 'mentions_option_D': 2}.

## Headline (paired, n = 343)

| Condition | Accuracy (95% CI) | Balanced acc. | False accept | False reject | Brier |
|---|---|---|---|---|---|
| J1 (no Bob) | 50.4% [45.0, 55.8] | 50.4% | 47.1% | 52.1% | 0.382 |
| J1 + Bob v1 | 50.1% [44.4, 56.1] | 49.7% | 16.7% | 84.0% | 0.328 |

**Δ accuracy (J1+Bob − J1): -0.3 pts**, 95% CI [-7.7, 6.9], McNemar p = 1.
Flips: 75 wrong→right, 76 right→wrong, 97 both right, 95 both wrong.

## Control: does J1+Bob beat Bob judging alone?

Bob alone 61.4% vs J1+Bob 49.9% (n = 345); Δ (J1+Bob − Bob alone) -11.6 pts, 95% CI [-18.9, -4.3], McNemar p = 0.000995.

## Controls: how much of Bob's edge reaches the judge?

| Condition | n | Accuracy | Δ vs J1 (95% CI, p) | Δ vs Bob alone (95% CI, p) | AUROC | ΔAUROC vs J1 (95% CI) | False accept | False reject |
|---|---|---|---|---|---|---|---|---|
| J1 + Bob v1 notes (no verdict) | 343 | 50.1% | -0.3 [-7.7, 6.9], p = 1 | -11.6 [-18.9, -4.3], p = 0.000995 | 0.531 | +0.009 [-0.070, +0.087] | 16.7% | 84.0% |
| J1 + Bob's verdict only (testimony) | 352 | 50.6% | -0.6 [-7.7, 7.2], p = 0.936 | -9.7 [-16.8, -2.0], p = 0.0107 | 0.522 | -0.009 [-0.095, +0.083] | 45.8% | 53.1% |

## Every condition on its own items (question-clustered 95% CIs)

| Condition | n | Accuracy | AUROC |
|---|---|---|---|
| j1 | 360 | 50.8% [45.4, 56.3] | 0.524 [0.458, 0.588] |
| j1_bob | 345 | 49.9% [44.0, 55.5] | 0.527 [0.468, 0.590] |
| bob_alone | 362 | 60.8% [55.5, 66.0] | 0.622 [0.557, 0.682] |
| j1_testimony | 352 | 50.6% [45.0, 56.3] | 0.522 [0.461, 0.587] |
| j1_bob_itt | 360 | 50.6% [44.8, 56.1] | 0.539 [0.480, 0.600] |

J1 on the full eval set (before dropping Bob-rejected items): 50.8% (n = 360). Judge failures per condition: {'j1': 2, 'j1_bob': 0, 'bob_alone': 0, 'j1_testimony': 10, 'j1_bob_free': 0, 'checks_bob_rule': 0, 'checks_bob_judge': 0, 'checks_self_rule': 0, 'checks_self_judge': 0, 'bob_solve_x4': 0, 'j1_testimony_x4': 0, 'j1_bob_x4': 0}.
