# Experiments

Experiment scripts, run outputs, and evaluation for GPU CVT.

## Layout

- **output/** – Run outputs from `main` (mode 0 → `gpucvt/<mesh>/`, mode 1 → `freeze/<mesh>/`). OBJs, eval CSVs, frozen logs. Gitignored.
- **results/** – Eval CSVs used for plotting. Copy or symlink from `output/gpucvt/<mesh>/eval_iters.csv` → `results/<mesh>/baseline/`, and `output/freeze/<mesh>/eval_iters.csv` → `results/<mesh>/freeze/`, then run the plot script.
- **plots_eval/** – Plots from `scripts/plot_eval_iters.py` (default `--out-dir experiments/plots_eval`).
- **exp*.py, plot_*.py** – Experiment and comparison scripts.
- **runs.csv** – Run log written by `main` (append).
- **notes/** – NextStep, process.txt, etc.

## Quick run + plot

From repo root:

```bash
./main objs/stanford-bunny.obj 0    # mode 0 → experiments/output/gpucvt/stanford-bunny/
./main objs/stanford-bunny.obj 1   # mode 1 → experiments/output/freeze/stanford-bunny/
mkdir -p experiments/results/stanford-bunny/{baseline,freeze}
cp experiments/output/gpucvt/stanford-bunny/eval_iters.csv experiments/results/stanford-bunny/baseline/
cp experiments/output/freeze/stanford-bunny/eval_iters.csv experiments/results/stanford-bunny/freeze/
python3 scripts/plot_eval_iters.py --mesh stanford-bunny --no-show
```
