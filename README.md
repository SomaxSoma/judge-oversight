# Weak-judge oversight on GPQA

This project tests one question: can a strong model (Bob) make a weak judge more accurate if Bob must not give a verdict?

## Setup

- **Alice** (`qwen2.5:7b`) answers each GPQA-Diamond question 3 times.
- The eval set has 362 answers from 181 questions: 50% correct, 50% incorrect.
- **The judge** decides if the answer from Alice is correct. The judge has no answer key.
- **Bob** (`qwen3:32b`) is stronger than the judge. Bob sends the judge one message through a channel.
- Bob must not state a verdict or an option. A leak filter checks each message (regex, then an LLM). The filter regenerates a flagged message once, then drops it.
- Each channel is compared with the judge alone (J1) on the same items.

| Channel | What the judge gets | Verdict-free |
|---|---|---|
| Notes | A critique of at most 150 words: the steps to check | Yes |
| Checks | 3 short sub-questions. The judge answers them blind, then decides | Yes |
| Testimony | Only the verdict and confidence of Bob (control) | No |

Bob alone is the critic: Bob judges Alice directly. The gap between Bob alone and J1 is the most a channel can transfer.

## Results

Δ accuracy against J1, in points, with p from the exact McNemar test. All runs use all 362 items.

| Judge | J1 % | Bob alone | Notes | Checks | Testimony |
|---|---|---|---|---|---|
| llama3.2:3b (Bob qwen3:8b) | 50.8 | +3.1 | +1.1 (p 0.81) | – | – |
| llama3.2:3b | 50.8 | +9.7 | −0.3 (p 1.0) | – | −0.6 (p 0.94) |
| llama3.1:8b | 50.4 | +10.4 | +3.4 (p 0.32) | +3.1 (p 0.32) | +2.8 (p 0.37) |
| qwen3:8b | 54.0 | +6.6 | −0.6 (p 0.94) | – | +1.9 (p 0.51) |
| gemma3:12b | 55.6 | +5.3 | −1.9 (p 0.56) | – | +1.1 (p 0.75) |
| gemma3:12b, Bob solve x4 | 55.6 | **+12.8** | – | – | **+7.5 (p 0.009)** |

1. The 32B Bob alone is better than every judge alone, by 5 to 10 points.
2. No verdict-free channel gives a significant gain. Notes and checks give −2 to +3 points.
3. A stronger Bob procedure increases the gap to 12.8 points. In this procedure (solve x4), Bob solves each question 4 times without Alice.
4. With that gap, the bare verdict of Bob transfers: +7.5 points [+1.4, +13.7]. This is a verdict channel, so it is a control, not oversight.
5. Pilots of 120 items or fewer gave false gains 3 times. Make decisions only on full runs.

All channels with CIs and AUROC: [results/channels.md](results/channels.md). Method and caveats: [docs/method.md](docs/method.md).

**Next step:** verdict-free notes from the solve-x4 Bob, with a leak audit ([configs/kaggle_notes_x4_jev.yaml](configs/kaggle_notes_x4_jev.yaml)). The code is ready. The run waits for GPU time.

## Layout

```
src/oversight/      pipeline: roles, leak filter, metrics, CLI (run.py)
prompts/            all model prompts
configs/            one YAML file for each run
autoresearch/       agent loop that searched Bob strategies (program.md)
notebooks/          Kaggle notebook for the 32B runs
scripts/            Kaggle push, bundle, channel aggregate
results/            aggregate metrics for each run, and channels.md
tests/              unit tests and a dry run on toy data
```

## Run

```bash
uv sync
uv run pytest
uv run oversight run --mock --csv tests/fixtures/toy.csv --out results-mock
```

The last command is a dry run on toy data. It does not call a model.

For a real run:
1. Get access to GPQA at huggingface.co/datasets/Idavidrein/gpqa. Set `HF_TOKEN`.
2. Start Ollama on port 11434.
3. Run a config: `uv run oversight run --config configs/<name>.yaml`.

The 32B Bob needs a large GPU. See [docs/kaggle.md](docs/kaggle.md). The cache keeps every model call, so a stopped run continues from where it stopped.

To rebuild the channel tables from `results/*/summary.json`:

```bash
uv run python scripts/aggregate_channels.py
```

## Data

GPQA asks users not to publish its questions. This repository has no questions, no model responses and no cache. The `results/` folder has aggregate metrics only. Do not commit `data/`, `cache/` or `*.jsonl` files.
