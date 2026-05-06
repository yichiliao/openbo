# OpenMetaBO

OpenMetaBO is a research + teaching Python library for Bayesian optimization (BO) and meta-BO.

## Current scope

This repository currently includes a first vertical slice:

- synthetic test functions (`Branin`, `Sphere`) in normalized input space `[0, 1]^d`
- random search baseline
- BO from scratch (NumPy GP + EI)
- BO with BoTorch (`SingleTaskGP` + `LogExpectedImprovement`)
- single-task and family benchmark runners
- log-regret plotting (single-task and family mean/std, including best-so-far)
- reusable train/test family split generation and persistence

The code is intentionally simple and teaching-oriented.

## Quickstart

Run all tests:

```bash
uv run pytest
```

## Single-function workflow

Run one benchmark on a single base function (default: `branin`):

```bash
uv run python scripts/run_benchmark.py --method random
uv run python scripts/run_benchmark.py --method bo_scratch
uv run python scripts/run_benchmark.py --method bo_botorch
```

Compare methods on one function and plot log-regret:

```bash
uv run python scripts/plot_results.py
```

This writes `benchmark_y_values.png`.
The y-axis is `log10(optimal_value - y_at_iteration)` using the known optimum.

## Family-of-functions workflow

Run one method across a family of Branin variants:

```bash
uv run python scripts/run_family_benchmark.py --method bo_scratch --n-tasks 10
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
  --method bo_scratch \
  --split-path configs/family_splits/branin_split.json \
  --subset test \
  --test-id exp001 \
  --results-dir test_results
```

This saves per-task trajectories and summary JSON to:
- `test_results/trajectories/`

Plot family results (mean/std log-regret and best-so-far log-regret):

```bash
uv run python scripts/plot_family_results.py
```

This writes:
- `family_mean_std_plot.png`
- `family_best_so_far_mean_std_plot.png`

With `--test-id` and `--results-dir`, plots are saved under:
- `test_results/plots/`

## Project structure

- `README.md` - project overview and usage.
- `pyproject.toml` - dependencies, build config, and project metadata.
- `configs/` - YAML configs and reusable artifacts.
  - `benchmark.yaml` - default benchmark config.
  - `methods/*.yaml` - method-level config placeholders.
  - `family_splits/*.json` - persisted train/test task-family splits.
- `src/metabo/` - main package.
  - `test_functions/synthetic.py` - base synthetic objectives (`Branin`, `Sphere`) and known optima.
  - `test_functions/tasks.py` - variant spec + transform wrapper for per-task perturbations.
  - `test_functions/registry.py` - function metadata registry and family builders.
  - `test_functions/families.py` - family generation, train/test split, save/load utilities.
  - `models/gp_scratch.py` - simple NumPy GP regression (fit/posterior).
  - `models/kernels.py` - RBF kernel implementation.
  - `acquisition/ei.py` - scratch Expected Improvement.
  - `optimizers/random_search.py` - random search baseline.
  - `optimizers/bo_scratch.py` - sequential BO with scratch GP + EI.
  - `optimizers/bo_botorch.py` - sequential BO with BoTorch GP + LogEI.
  - `benchmarks/runner.py` - unified entrypoint for single-function benchmark runs.
- `scripts/` - command-line entrypoints.
  - `run_benchmark.py` - run a single-function benchmark.
  - `plot_results.py` - single-function method comparison plot.
  - `create_family_split.py` - create and save train/test split for a task family.
  - `run_family_benchmark.py` - run one method over family tasks (or a saved split subset).
  - `plot_family_results.py` - family mean/std plots (raw and best-so-far log-regret).
- `tests/` - test suite.
  - `test_functions_test.py` - synthetic functions, variants, and family split persistence tests.
  - `gp_scratch_test.py` - scratch GP fit/posterior tests.
  - `acquisition_test.py` - EI tests and BO smoke tests.
- `notebooks/` - teaching notebooks for BO concepts and step-by-step demos.

## Contact
This repo is created and maintained by Yi-Chi Liao (yichi.liao@inf.ethz.ch).