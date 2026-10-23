# Locked test split, autoresearch loop (judge llama3.1:8b, Bob qwen3:8b), run 2026-10-22

Stage-0 winner = solve x4 (`src/oversight/bob_solve.py`); v1 baseline = Bob v1 (`src/oversight/bob_strategy.py`). The test split is now spent for these claims.

## Test split only

Test split: 148 items, 70 questions; new calls {}

| Condition | AUROC (95% CI) |
|---|---|
| J1 (judge alone) | 0.419 [0.323, 0.516] |
| J1 + testimony | 0.479 [0.383, 0.584] |
| J1 + Bob v1 notes | 0.493 [0.392, 0.591] |
| Bob alone, v1 prompt | 0.532 [0.434, 0.624] |
| Bob alone, solve x4 (winner) | 0.669 [0.542, 0.790] |

| Comparison | ΔAUROC | 95% CI | P(Δ > 0) |
|---|---|---|---|
| Bob alone, solve x4 (winner) − J1 (judge alone) | +0.251 | [+0.074, +0.415] | 0.998 |
| Bob alone, solve x4 (winner) − Bob alone, v1 prompt | +0.137 | [-0.007, +0.272] | 0.973 |
| J1 + Bob v1 notes − J1 (judge alone) | +0.075 | [-0.030, +0.184] | 0.923 |
| J1 + Bob v1 notes − J1 + testimony | +0.014 | [-0.099, +0.134] | 0.606 |
| J1 + testimony − J1 (judge alone) | +0.061 | [-0.034, +0.157] | 0.904 |
| J1 + Bob v1 notes − Bob alone, solve x4 (winner) | -0.176 | [-0.324, -0.014] | 0.017 |

## Pooled dev120 + test (all 362 items, 181 questions)
Fair for the v1-notes comparisons (v1 was never tuned on dev). Bob x4 and testimony include the dev questions it was selected on.

Pooled: 362 items, 181 questions; new calls {}

| Condition | AUROC (95% CI) |
|---|---|
| J1 (judge alone) | 0.466 [0.400, 0.529] |
| J1 + testimony | 0.474 [0.411, 0.535] |
| J1 + Bob v1 notes | 0.529 [0.465, 0.594] |
| Bob alone, v1 prompt | 0.549 [0.485, 0.608] |
| Bob alone, solve x4 (winner) | 0.678 [0.601, 0.752] |

| Comparison | ΔAUROC | 95% CI | P(Δ > 0) |
|---|---|---|---|
| Bob alone, solve x4 (winner) − J1 (judge alone) | +0.212 | [+0.114, +0.304] | 1.000 |
| Bob alone, solve x4 (winner) − Bob alone, v1 prompt | +0.129 | [+0.043, +0.215] | 0.998 |
| J1 + Bob v1 notes − J1 (judge alone) | +0.063 | [-0.008, +0.129] | 0.962 |
| J1 + Bob v1 notes − J1 + testimony | +0.056 | [-0.014, +0.125] | 0.941 |
| J1 + testimony − J1 (judge alone) | +0.008 | [-0.059, +0.071] | 0.603 |
| J1 + Bob v1 notes − Bob alone, solve x4 (winner) | -0.148 | [-0.240, -0.051] | 0.002 |
