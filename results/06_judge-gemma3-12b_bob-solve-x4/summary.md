# Judge accuracy: baseline (J1) vs Bob v1 (J1+Bob)

Alice: 594 samples, accuracy 31.6%, 21 unusable. Balanced eval set: 362 answers from 181 questions (50% correct).
Bob: 345/362 notes accepted, 17 dropped for leaks or refusals (4.7%), 35 regenerated, mean 152 words. Final-attempt flags: {'option_letter': 12, 'mentions_option_B': 4, 'mentions_option_C': 2, 'mentions_option_A': 1, 'mentions_option_D': 2}.

## Headline (paired, n = 342)

| Condition | Accuracy (95% CI) | Balanced acc. | False accept | False reject | Brier |
|---|---|---|---|---|---|
| J1 (no Bob) | 55.3% [49.3, 61.1] | 55.3% | 44.3% | 45.2% | 0.366 |
| J1 + Bob v1 | 53.2% [47.5, 59.0] | 52.8% | 25.3% | 69.0% | 0.346 |

**Δ accuracy (J1+Bob − J1): -2.0 pts**, 95% CI [-7.9, 4.1], McNemar p = 0.558.
Flips: 49 wrong→right, 56 right→wrong, 133 both right, 104 both wrong.

## Control: does J1+Bob beat Bob judging alone?

Bob alone 61.2% vs J1+Bob 53.1% (n = 343); Δ (J1+Bob − Bob alone) -8.2 pts, 95% CI [-14.6, -1.4], McNemar p = 0.0142.

## Controls: how much of Bob's edge reaches the judge?

| Condition | n | Accuracy | Δ vs J1 (95% CI, p) | Δ vs Bob alone (95% CI, p) | AUROC | ΔAUROC vs J1 (95% CI) | False accept | False reject |
|---|---|---|---|---|---|---|---|---|
| J1 + Bob v1 notes (no verdict) | 342 | 53.2% | -2.0 [-7.9, 4.1], p = 0.558 | -8.2 [-14.6, -1.4], p = 0.0142 | 0.506 | -0.026 [-0.083, +0.029] | 25.3% | 69.0% |
| J1 + Bob's verdict only (testimony) | 360 | 56.7% | +1.1 [-3.9, 6.3], p = 0.752 | -4.2 [-7.6, -0.9], p = 0.0135 | 0.578 | +0.037 [-0.017, +0.091] | 33.9% | 52.8% |
| Bob alone, solve x4 (Δ vs Bob alone = x4 − v1) | 360 | 68.3% | +12.8 [4.3, 21.3], p = 0.000287 | +7.5 [0.0, 14.9], p = 0.0282 | 0.742 | +0.202 [+0.112, +0.292] | 39.4% | 23.9% |
| J1 + solve-x4 verdict only (testimony x4) | 360 | 63.1% | +7.5 [1.4, 13.7], p = 0.00934 | +2.2 [-4.3, 9.0], p = 0.512 | 0.660 | +0.120 [+0.051, +0.186] | 31.7% | 42.2% |
- J1 + Bob v1 notes (no verdict) vs Bob alone solve x4 (n = 343): Δ accuracy -15.2 pts [-24.1, -6.7], p = 6.65e-05; Δ AUROC -0.232 [-0.316, -0.149].
- J1 + Bob's verdict only (testimony) vs Bob alone solve x4 (n = 361): Δ accuracy -11.6 pts [-19.6, -3.5], p = 0.000698; Δ AUROC -0.162 [-0.237, -0.087].
- J1 + solve-x4 verdict only (testimony x4) vs Bob alone solve x4 (n = 361): Δ accuracy -5.3 pts [-10.5, 0.0], p = 0.0271; Δ AUROC -0.081 [-0.128, -0.032].

## Every condition on its own items (question-clustered 95% CIs)

| Condition | n | Accuracy | AUROC |
|---|---|---|---|
| j1 | 360 | 55.6% [49.9, 61.1] | 0.541 [0.477, 0.606] |
| j1_bob | 343 | 53.1% [47.4, 58.8] | 0.506 [0.433, 0.581] |
| bob_alone | 362 | 60.8% [55.5, 66.0] | 0.622 [0.557, 0.682] |
| j1_testimony | 361 | 56.5% [51.0, 61.8] | 0.578 [0.514, 0.641] |
| bob_solve_x4 | 362 | 68.2% [61.5, 74.9] | 0.740 [0.665, 0.808] |
| j1_testimony_x4 | 361 | 62.9% [56.7, 69.1] | 0.658 [0.584, 0.725] |
| j1_bob_itt | 360 | 53.6% [48.3, 59.3] | 0.522 [0.453, 0.589] |

J1 on the full eval set (before dropping Bob-rejected items): 55.6% (n = 360). Judge failures per condition: {'j1': 2, 'j1_bob': 2, 'bob_alone': 0, 'j1_testimony': 1, 'j1_bob_free': 0, 'checks_bob_rule': 0, 'checks_bob_judge': 0, 'checks_self_rule': 0, 'checks_self_judge': 0, 'bob_solve_x4': 0, 'j1_testimony_x4': 1, 'j1_bob_x4': 0}.
