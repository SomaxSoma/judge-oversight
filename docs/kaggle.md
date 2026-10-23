# Run on Kaggle

The 32B Bob needs about 20 GB of GPU memory. A free Kaggle notebook with GPU T4 x2 has 32 GB.

## Set up (one time)

1. Make a Kaggle notebook from [notebooks/kaggle_run.ipynb](../notebooks/kaggle_run.ipynb).
2. Set Accelerator to **GPU T4 x2**. Set **Internet on**. Internet needs a phone-verified account.
3. Make the bundle. It contains the code, the cache and an encrypted copy of GPQA:
   ```bash
   uv run --group kaggle python scripts/make_bundle.py --gpqa-csv data/gpqa_diamond.csv
   ```
   The key goes to `.gpqa_bundle_key`. Do not commit this file.
4. Upload `../oversight_bundle.zip` as a **private** Kaggle dataset. Attach it to the notebook.

## Start a run

```bash
uv run python scripts/kaggle_push.py --kernel USER/NOTEBOOK --configs kaggle_solve_x4_jev --dry-run
uv run python scripts/kaggle_push.py --kernel USER/NOTEBOOK --configs kaggle_solve_x4_jev
```

The script writes the config names and the key into a temporary copy of the notebook. Then it pushes a new version. The key does not go into the repository.

## Get the results

1. When the version is complete, download `oversight_outputs.zip` from its Output tab.
2. Unzip it into the project. It adds `results/<run>/` and new files in `cache/`.

## Limits

- One session runs for 12 hours or less. Every 30 minutes the notebook writes `oversight_checkpoint.zip`. Unzip it and run again to continue.
- The free GPU quota is 30 hours each week.
