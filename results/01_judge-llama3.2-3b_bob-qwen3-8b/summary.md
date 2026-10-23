# Judge accuracy: baseline (J1) vs Bob v1 (J1+Bob)

Alice: 594 samples, accuracy 31.6%, 21 unusable. Balanced eval set: 362 answers from 181 questions (50% correct).
Bob: 337/362 notes accepted, 25 dropped for leaks or refusals (6.9%), 43 regenerated, mean 141 words. Final-attempt flags: {'option_letter': 16, 'mentions_option_A': 4, 'mentions_option_C': 4, 'mentions_option_B': 7, 'mentions_option_D': 2}.

## Headline (paired, n = 333)

| Condition | Accuracy (95% CI) | Balanced acc. | False accept | False reject | Brier |
|---|---|---|---|---|---|
| J1 (no Bob) | 50.2% [44.9, 55.5] | 50.1% | 47.9% | 51.8% | 0.383 |
| J1 + Bob v1 | 51.4% [45.1, 57.5] | 50.8% | 12.4% | 86.0% | 0.340 |

**Δ accuracy (J1+Bob − J1): +1.2 pts**, 95% CI [-6.5, 8.7], McNemar p = 0.81.
Flips: 80 wrong→right, 76 right→wrong, 91 both right, 86 both wrong.

## Control: does J1+Bob beat Bob judging alone?

Bob alone 52.1% vs J1+Bob 51.5% (n = 334); Δ (J1+Bob − Bob alone) -0.6 pts, 95% CI [-9.7, 8.1], McNemar p = 0.94.

## Every condition on its own items (question-clustered 95% CIs)

| Condition | n | Accuracy | AUROC |
|---|---|---|---|
| j1 | 360 | 50.8% [45.4, 56.3] | 0.524 [0.458, 0.588] |
| j1_bob | 335 | 51.3% [45.2, 57.1] | 0.495 [0.432, 0.555] |
| bob_alone | 361 | 54.0% [48.2, 59.6] | 0.550 [0.491, 0.610] |
| j1_bob_itt | 360 | 51.9% [45.8, 57.8] | 0.516 [0.456, 0.580] |

J1 on the full eval set (before dropping Bob-rejected items): 50.8% (n = 360). Judge failures per condition: {'j1': 2, 'j1_bob': 2, 'bob_alone': 1, 'j1_testimony': 0, 'j1_bob_free': 0, 'checks_bob_rule': 0, 'checks_bob_judge': 0, 'checks_self_rule': 0, 'checks_self_judge': 0, 'bob_solve_x4': 0, 'j1_testimony_x4': 0, 'j1_bob_x4': 0}.
