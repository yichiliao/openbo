# Families of related tasks

[← Guide index](README.md) · [README](../../README.md)

Before starting meta-learning or transfer learning, we need to create the settings for such tasks. Importantly, we need to work with **many related tasks** (variants of one base function): run one method across the family, save per-task trajectories, create persistent train/test splits, and plot mean/std log-regret with `plot_family_results.py`.

Run one method across a family of Branin variants:

```bash
uv run python scripts/run_family_benchmark.py --method bo_scratch_multistart --n-tasks 10 --test-id family10
```

Without `--test-id` the id is `default`, and the run is written to
`test_results/trajectories/default_bo_scratch_multistart_branin_all/`, which
overwrites the committed run of that name (5 tasks; see the last examples on this page).

Run a noisy family benchmark (noise_std=0.05, capped at optimum):

```bash
uv run python scripts/run_family_benchmark.py \
  --method bo_scratch_multistart \
  --base-function branin \
  --n-tasks 10 \
  --noisy \
  --test-id family10_noisy
```

Compare multiple methods on one family in a single plot run:

```bash
uv run python scripts/plot_family_results.py \
  --base-function branin \
  --methods random bo_scratch_grid bo_scratch_multistart bo_botorch \
  --n-tasks 10 \
  --n-evals 30 \
  --test-id compare_all_methods_branin10 \
  --results-dir test_results
```

Side note: you can compare multiple methods
on the same family setup in a single plotting run.

Run the same family comparison in noisy mode:

```bash
uv run python scripts/plot_family_results.py \
  --base-function branin \
  --methods random bo_scratch_grid bo_scratch_multistart bo_botorch \
  --n-tasks 10 \
  --n-evals 30 \
  --noisy \
  --test-id compare_all_methods_branin10_noisy \
  --results-dir test_results
```

Create a persistent train/test split:

```bash
uv run python scripts/create_family_split.py \
  --base-function branin \
  --n-tasks 50 \
  --train-ratio 0.8
```

Run benchmark only on test tasks from the saved split:

```bash
uv run python scripts/run_family_benchmark.py \
  --method bo_scratch_multistart \
  --split-path configs/family_splits/branin_split.json \
  --subset test \
  --test-id exp001 \
  --results-dir test_results
```

This saves per-task trajectories and summary JSON under a run subfolder named
`{test_id}_{method}_{base_function}_{subset}` (TAF methods append `_taf_m` or `_taf_r`):
- `test_results/trajectories/exp001_bo_scratch_multistart_branin_test/`
- one JSON per task (e.g. `test_task_000.json`)
- one run summary (`summary.json`)

Plot family results (mean/std log-regret and best-so-far log-regret):

```bash
uv run python scripts/plot_family_results.py
```

This writes (with the default `--test-id default` and `--results-dir test_results`):
- `test_results/plots/default_branin_family_mean_std_plot.png`
- `test_results/plots/default_branin_family_best_so_far_mean_std_plot.png`

Important:
- `plot_family_results.py` generates plots, but does **not** save per-task trajectories.
- Use `run_family_benchmark.py` when you want trajectory JSON files.

With `--test-id` and `--results-dir`, plots are saved under:
- `test_results/plots/`

Plot family results directly from a stored run folder (no optimizer rerun):

```bash
uv run python scripts/plot_family_results.py \
  --trajectory-run-dir test_results/trajectories/exp001_bo_scratch_multistart_branin_test \
  --test-id exp001_from_stored \
  --results-dir test_results
```

That mode plots **one** method per invocation (the method recorded in `summary.json`).

**Combine several prior `run_family_benchmark.py` runs** (same task family and length) into **one** figure matching `plot_family_results.py` styling—each run gets a legend name you choose (`folder_or_path:label`). Add **as many `--run` lines as you want** (two, three, or more methods on the same plot).

The `default_*_branin_all` runs below are committed in the repo (5 tasks each). All runs you
compare must have the same number of tasks, so do not overwrite one of them with a run of a
different size (for example by running the 10-task examples above without `--test-id`).

```bash
uv run python scripts/plot_family_from_benchmark_runs.py \
  --results-dir test_results \
  --plot-id compare_botorch_random \
  --run default_bo_botorch_branin_all:botorch \
  --run default_random_branin_all:random
```

or 

```bash
uv run python scripts/plot_family_from_benchmark_runs.py \
  --results-dir test_results \
  --plot-id compare_three \
  --run default_bo_scratch_multistart_branin_all:scratch \
  --run default_bo_botorch_branin_all:botorch \
  --run default_random_branin_all:random
```

Relative names resolve to `test_results/trajectories/<name>/`. Pass absolute paths if the runs live elsewhere. If trajectory lengths differ, the script trims to a common length and prints a warning. Optional `--output` / `--output-best-so-far` set explicit PNG paths.
