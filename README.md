# OpenMetaBO

OpenMetaBO is a research + teaching Python library for Bayesian optimization (BO) and meta-BO.

## Current scope

This repository currently includes a first vertical slice:

- synthetic test functions (`Branin`, `Sphere`, `Ackley`, `Rastrigin`, `Rosenbrock`, `Hartmann6`) in normalized input space `[0, 1]^d`
- random search baseline
- BO from scratch (NumPy GP + EI)
- BO with BoTorch (`SingleTaskGP` + `LogExpectedImprovement`)
- single-task and family benchmark runners
- trajectory persistence for single-task and family runs
- log-regret plotting (single-task and family mean/std, including best-so-far)
- reusable train/test family split generation and persistence


## Quickstart

Run all tests:

```bash
uv run pytest
```

## Single-function workflow

Run one benchmark on a single base function (default: `branin`).

Available methods:
- `random`
- `bo_scratch_grid`
- `bo_scratch_multistart`
- `bo_botorch`

```bash
uv run python scripts/run_benchmark.py --method random
uv run python scripts/run_benchmark.py --method bo_scratch_grid
uv run python scripts/run_benchmark.py --method bo_scratch_multistart
uv run python scripts/run_benchmark.py --method bo_botorch
```

Store outputs with experiment ids:

```bash
uv run python scripts/run_benchmark.py \
  --method bo_scratch_multistart \
  --function branin \
  --test-id exp_single_001 \
  --results-dir test_results
```

This saves:
- `test_results/trajectories/exp_single_001_bo_scratch_multistart_branin.json`

Generate a 2D search-trajectory plot on top of a function-value heatmap
(blue-to-red), with points colored by iteration (light gray to black):

```bash
uv run python scripts/run_benchmark.py \
  --method bo_botorch \
  --function branin \
  --n-evals 30 \
  --test-id xloc_demo \
  --results-dir test_results \
  --plot-x-locations
```

This writes:
- `test_results/plots/xloc_demo_bo_botorch_branin_x_locations.png`

### 2D heatmap + trajectory quick recipes

This feature is available only for 2D functions (e.g. `branin`, `sphere`, `ackley`, `rastrigin`, `rosenbrock`).
For non-2D functions (e.g. `hartmann6`), the script will skip this plot.

Run all three methods on the same 2D function so outputs are easy to compare:

```bash
uv run python scripts/run_benchmark.py \
  --method random \
  --function sphere \
  --n-evals 30 \
  --test-id xloc_sphere \
  --results-dir test_results \
  --plot-x-locations

uv run python scripts/run_benchmark.py \
  --method bo_scratch_multistart \
  --function sphere \
  --n-evals 30 \
  --test-id xloc_sphere \
  --results-dir test_results \
  --plot-x-locations

uv run python scripts/run_benchmark.py \
  --method bo_botorch \
  --function sphere \
  --n-evals 30 \
  --test-id xloc_sphere \
  --results-dir test_results \
  --plot-x-locations
```

These commands write:
- `test_results/plots/xloc_sphere_random_sphere_x_locations.png`
- `test_results/plots/xloc_sphere_bo_scratch_multistart_sphere_x_locations.png`
- `test_results/plots/xloc_sphere_bo_botorch_sphere_x_locations.png`

You can override the output path with `--plot-output`:

```bash
uv run python scripts/run_benchmark.py \
  --method bo_botorch \
  --function branin \
  --plot-x-locations \
  --plot-output test_results/plots/custom_branin_xloc.png
```

Compare methods on one function and plot log-regret:

```bash
uv run python scripts/plot_results.py
```

This writes `benchmark_y_values.png`.
The y-axis is `log10(optimal_value - y_at_iteration)` using the known optimum.

Plot from stored single-run trajectories (no optimizer rerun):

```bash
uv run python scripts/plot_results.py \
  --methods random bo_scratch_grid bo_scratch_multistart bo_botorch \
  --function branin \
  --trajectory-dir test_results/trajectories \
  --test-id exp_single_001 \
  --output test_results/plots/exp_single_001_from_stored.png
```

## Family-of-functions workflow

Run one method across a family of Branin variants:

```bash
uv run python scripts/run_family_benchmark.py --method bo_scratch_multistart --n-tasks 10
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

Create a persistent train/test split:

```bash
uv run python scripts/create_family_split.py \
  --n-tasks 50 \
  --train-ratio 0.8 \
  --output configs/family_splits/branin_split.json
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

This saves per-task trajectories and summary JSON under a run subfolder:
- `test_results/trajectories/exp001_bo_scratch_branin_test/`
- one JSON per task (e.g. `test_task_000.json`)
- one run summary (`summary.json`)

Plot family results (mean/std log-regret and best-so-far log-regret):

```bash
uv run python scripts/plot_family_results.py
```

This writes:
- `family_mean_std_plot.png`
- `family_best_so_far_mean_std_plot.png`

Important:
- `plot_family_results.py` generates plots, but does **not** save per-task trajectories.
- Use `run_family_benchmark.py` when you want trajectory JSON files.

With `--test-id` and `--results-dir`, plots are saved under:
- `test_results/plots/`

Plot family results directly from a stored run folder (no optimizer rerun):

```bash
uv run python scripts/plot_family_results.py \
  --trajectory-run-dir test_results/trajectories/exp001_bo_scratch_branin_test \
  --test-id exp001_from_stored \
  --results-dir test_results
```

## Results folder convention

By default, artifacts are organized under `test_results/`:

- `test_results/trajectories/`
  - single-function run JSONs:
    - `{test_id}_{method}_{function}.json`
  - family run folders:
    - `{test_id}_{method}_{base_function}_{subset}/`
    - containing one task file per task plus `summary.json`
- `test_results/plots/`
  - single-function and family plot PNGs

## Project structure

- `README.md` - project overview and usage.
- `pyproject.toml` - dependencies, build config, and project metadata.
- `configs/` - YAML configs and reusable artifacts.
  - `benchmark.yaml` - default benchmark config.
  - `methods/*.yaml` - method-level config placeholders.
  - `family_splits/*.json` - persisted train/test task-family splits.
- `src/metabo/` - main package.
  - `test_functions/synthetic.py` - base synthetic objectives and known optima.
  - `test_functions/tasks.py` - variant spec + transform wrapper for per-task perturbations.
  - `test_functions/registry.py` - function metadata registry and family builders.
  - `test_functions/families.py` - family generation, train/test split, save/load utilities.
  - `models/gp_scratch.py` - simple NumPy GP regression (fit/posterior).
  - `models/kernels.py` - kernel implementations (RBF and Matérn-5/2 ARD).
  - `acquisition/ei.py` - scratch Expected Improvement.
  - `optimizers/random_search.py` - random search baseline.
  - `optimizers/bo_scratch.py` - sequential BO with scratch GP + EI (grid or multi-start search).
  - `optimizers/bo_botorch.py` - sequential BO with BoTorch GP + LogEI.
  - `benchmarks/runner.py` - unified entrypoint for single-function benchmark runs.
- `scripts/` - command-line entrypoints.
  - `run_benchmark.py` - run a single-function benchmark (optionally with 2D heatmap + x-location plotting via `--plot-x-locations`).
  - `plot_results.py` - single-function comparison plot (rerun or from stored trajectories).
  - `create_family_split.py` - create and save train/test split for a task family.
  - `run_family_benchmark.py` - run one method over family tasks and save one JSON per task.
  - `plot_family_results.py` - family mean/std plots (rerun or from stored trajectories).
- `tests/` - test suite.
  - `test_functions_test.py` - synthetic functions, variants, and family split persistence tests.
  - `gp_scratch_test.py` - scratch GP fit/posterior tests.
  - `acquisition_test.py` - EI tests and BO smoke tests.
- `notebooks/` - teaching notebooks for BO concepts and step-by-step demos.

## Contact
This repo is created and maintained by Yi-Chi Liao (yichi.liao@inf.ethz.ch).