# Method

## Data and labels

- GPQA-Diamond: 198 hard multiple-choice science questions with an answer key. Options are shuffled with a fixed seed.
- Alice (`qwen2.5:7b`) answers each question 3 times. Alice is correct on 31.6% of answers.
- Label: the option from Alice is equal to the key.
- The eval set keeps 362 answers from 181 questions, 50% correct and 50% incorrect. A judge that always says "correct" gets 50%.

This is the sandwich setup (Bowman et al., 2022): the judge cannot check the answers, but the key exists for scoring.

## Roles

| Role | Model | Sees |
|---|---|---|
| Alice | `qwen2.5:7b` | the question |
| Judge | `llama3.2:3b`, `llama3.1:8b`, `qwen3:8b` or `gemma3:12b` | the question, the response from Alice, and one channel |
| Bob | `qwen3:32b` (thinking off); `qwen3:8b` in run 01 | the question and the response from Alice. Bob never sees the key |

## Conditions

| Condition | Description |
|---|---|
| `j1` | The judge alone. This is the baseline. |
| `bob_alone` | Bob judges Alice directly with the judge prompt (v1). |
| `bob_solve_x4` | Bob solves the question 4 times without Alice. p(correct) is the fraction of solves that agree with Alice. |
| `j1_bob` | The judge gets the notes from Bob: at most 150 words on the steps to check ([prompts/bob_v1.md](../prompts/bob_v1.md)). |
| `checks_bob_judge` | Bob writes 3 sub-questions. The judge answers each one blind, then decides. |
| `checks_self_judge` | The same, but the judge writes its own sub-questions. This control has no Bob. |
| `j1_testimony` | The judge gets only the verdict and confidence of Bob. This control shows if the judge uses Bob at all. |
| `j1_testimony_x4` | The same, with the solve-x4 verdict. |
| `j1_bob_x4` | Notes, but Bob sees his own private solve-x4 working. Not run yet. |

## Leak filter

Bob must not state or suggest a verdict or an option. [src/oversight/leakage.py](../src/oversight/leakage.py) checks each message:
1. A regex looks for option letters and verdict words.
2. An LLM (`qwen3:8b`) looks for a stated verdict or answer.
3. The filter regenerates a flagged message once. If it fails again, the filter drops it.

The primary analysis is intention-to-treat (ITT). A dropped item keeps its J1 verdict. Thus the filter cannot select easy items.

The leak audit ([src/oversight/leak_audit.py](../src/oversight/leak_audit.py)) is for evaluation only. The judge model reads the notes and guesses the private verdict of Bob. A high AUROC shows that the notes carry the verdict.

## Metrics

- Accuracy, and AUROC of p(correct).
- Δ against J1 on the same items.
- 95% CI from a bootstrap that resamples whole questions. The 3 answers to one question are correlated.
- Exact McNemar p for Δ accuracy.

## Autoresearch loop

[autoresearch/program.md](../autoresearch/program.md) lets an agent edit only [src/oversight/bob_strategy.py](../src/oversight/bob_strategy.py). The judge, filter, data and metrics stay frozen.
- Stage 0 scores Bob alone. The winner was solve x4.
- Stage 1 scores the notes. No note design beat v1.
- The locked test split (148 items, 70 questions, judge `llama3.1:8b`, Bob `qwen3:8b`) confirmed solve x4. AUROC was 0.669, against 0.419 for J1. See [results/07_autoresearch-test-split/report.md](../results/07_autoresearch-test-split/report.md).

## Caveats

- **Small screens:** pilots of 60 to 120 items showed large gains (up to +20 points) 3 times. Full runs did not confirm them.
- **Selection:** solve x4 was selected on dev questions. These questions are also in the 362-item eval set, so the solve-x4 gain over v1 can be too high.
- **Same family:** in run 04, the judge (`qwen3:8b`) and Bob (`qwen3:32b`) are the same family.
- **Re-analysis:** the results come from the cached model calls, analysed with the code in this repository. For run 05, the checks rows are slightly different from the first report (Bob checks +3.1 points, first report +3.6). A later change in the scoring of the checks caused this. The conclusion does not change.
