# OpenBO

OpenBO is a research + teaching Python library for Bayesian optimization (BO) and meta-BO.

## Current scope

This repository currently includes:

- synthetic test functions (`Branin`, `Sphere`, `Ackley`, `Rastrigin`, `Rosenbrock`, `Hartmann6`) in normalized input space `[0, 1]^d`
- random search baseline
- BO from scratch (NumPy GP + EI)
- BO with BoTorch (`SingleTaskGP` + `LogExpectedImprovement`)
- single-task and family benchmark runners
- trajectory persistence for single-task and family runs
- log-regret plotting (single-task and family mean/std, including best-so-far)
- reusable train/test family split generation and persistence


## Quickstart

Set up the project environment first:

```bash
# Create/update .venv and install dependencies from uv.lock
uv sync
```

Run commands in either of these ways:

```bash
# Option A (recommended): run directly via uv without manual activation
uv run pytest
uv run python scripts/run_benchmark.py --help

# Option B: activate .venv, then run python/pytest normally
source .venv/bin/activate
python -m pytest
```

All commands in this README assume either `uv run ...` or an activated `.venv`.

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
- `bo_taf_m` (TAF with meta-feature similarity weights)
- `bo_taf_r` (TAF with ranking-agreement weights)

```bash
uv run python scripts/run_benchmark.py --method random
uv run python scripts/run_benchmark.py --method bo_scratch_grid
uv run python scripts/run_benchmark.py --method bo_scratch_multistart
uv run python scripts/run_benchmark.py --method bo_botorch
uv run python scripts/run_benchmark.py --method bo_taf_m --taf-run-dir meta-bo-training/taf-gps/branin_train_v1
uv run python scripts/run_benchmark.py --method bo_taf_r --taf-run-dir meta-bo-training/taf-gps/branin_train_v1
```

### Noise-free vs noisy objectives

Noise-free benchmarks are the default:

```bash
uv run python scripts/run_benchmark.py --method bo_scratch_multistart --function branin
```

Use `--noisy` to enable noisy observations globally for the selected benchmark run
(`noise_std=0.05`, `cap_at_optimum=True`):

```bash
uv run python scripts/run_benchmark.py \
  --method bo_scratch_multistart \
  --function branin \
  --noisy
```

Notes:
- Without `--noisy`, objectives are deterministic (`noise_std=0.0`).
- With `--noisy`, noise is seeded from `--seed` for reproducibility.
- Trajectory JSON files record `noisy`, `noise_std`, and `cap_at_optimum`.

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

`plot_results.py` supports two modes:
- **Rerun mode (default):** runs optimizers first, then plots.
- **Stored mode:** loads previously saved trajectory JSON files and plots directly.

Common useful flags:
- `--methods`: list of methods to compare.
- `--function`: base function name (e.g. `branin`, `sphere`, `hartmann6`).
- `--n-evals`: evaluation budget for rerun mode.
- `--seed`: random seed for rerun mode.
- `--noisy`: rerun mode only; uses `noise_std=0.05` with capped outputs.
- `--output`: output PNG path.
- `--trajectory-dir` + `--test-id`: switch to stored mode.

Plot from stored single-run trajectories (no optimizer rerun):

```bash
uv run python scripts/plot_results.py \
  --methods random bo_scratch_grid bo_scratch_multistart bo_botorch \
  --function branin \
  --trajectory-dir test_results/trajectories \
  --test-id exp_single_001 \
  --output test_results/plots/exp_single_001_from_stored.png
```

Example rerun command with explicit settings:

```bash
uv run python scripts/plot_results.py \
  --methods random bo_scratch_multistart bo_botorch \
  --function branin \
  --n-evals 30 \
  --seed 0 \
  --output test_results/plots/compare_branin_rerun.png
```

Example rerun command in noisy mode:

```bash
uv run python scripts/plot_results.py \
  --methods random bo_scratch_grid bo_scratch_multistart bo_botorch \
  --function branin \
  --n-evals 30 \
  --seed 0 \
  --noisy \
  --output test_results/plots/compare_all_methods_branin_noisy.png
```

Tip:
- If you benchmarked with `scripts/run_benchmark.py --noisy`, and want plots from those exact runs, use **stored mode** so the plot reflects the same noisy trajectories.

## Family-of-functions workflow

Run one method across a family of Branin variants:

```bash
uv run python scripts/run_family_benchmark.py --method bo_scratch_multistart --n-tasks 10
```

Run a noisy family benchmark (noise_std=0.05, capped at optimum):

```bash
uv run python scripts/run_family_benchmark.py \
  --method bo_scratch_multistart \
  --base-function branin \
  --n-tasks 10 \
  --noisy
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

## Meta-BO training and testing

Generally meta-Bayesian optimization workflow follows these three steps:
- Step 1: Generate a family of tasks using the same base function, and then split the training tasks and testing tasks.
- Step 2: Train the optimizer with the training tasks.
- Step 3: Test the optimizer's performance on test tasks. 


### Transfer Acquisition Function (TAF) workflow

Step 1: Create a persistent train/test split:

```bash
uv run python scripts/create_family_split.py \
  --base-function branin \
  --n-tasks 15 \
  --train-ratio 0.8
```

By default, the split is saved to:
- `configs/family_splits/{base_function}_split.json`

Step 2: Training + GP prediction visualization

Train TAF by running `bo_scratch_multistart` on the train subset and saving:
- per-task trajectories (`trajectories/`)
- final GP states (`gp_states/`)

```bash
uv run python scripts/train_taf.py \
  --split-path configs/family_splits/branin_split.json \
  --subset train \
  --run-id branin_train_v1
```

This writes to:
- `meta-bo-training/taf-gps/branin_train_v1/summary.json`
- `meta-bo-training/taf-gps/branin_train_v1/trajectories/*.json`
- `meta-bo-training/taf-gps/branin_train_v1/gp_states/*.json`

Then visualize GP predictions as 2D heatmaps (mean/std):

```bash
uv run python scripts/plot_taf_gp_predictions.py \
  --run-dir meta-bo-training/taf-gps/branin_train_v1
```

Heatmaps are saved under:
- `meta-bo-training/taf-gps/branin_train_v1/gp_predictions/`

Step 3: Testing 

You can now evaluate TAF as separate high-level methods:
- `bo_taf_m`: TAF-M weights from meta-feature similarity.
- `bo_taf_r`: TAF-R weights from ranking agreement on observed target points.

Single-function TAF evaluation:

```bash
uv run python scripts/run_benchmark.py \
  --method bo_taf_m \
  --function branin \
  --taf-run-dir meta-bo-training/taf-gps/branin_train_v1 \
  --test-id taf_m_single

uv run python scripts/run_benchmark.py \
  --method bo_taf_r \
  --function branin \
  --taf-run-dir meta-bo-training/taf-gps/branin_train_v1 \
  --test-id taf_r_single
```

Family benchmark evaluation on split test tasks:

```bash
uv run python scripts/run_family_benchmark.py \
  --method bo_taf_m \
  --split-path configs/family_splits/branin_split.json \
  --subset test \
  --taf-run-dir meta-bo-training/taf-gps/branin_train_v1 \
  --test-id taf_m_family

uv run python scripts/run_family_benchmark.py \
  --method bo_taf_r \
  --split-path configs/family_splits/branin_split.json \
  --subset test \
  --taf-run-dir meta-bo-training/taf-gps/branin_train_v1 \
  --test-id taf_r_family
```

Family plotting comparison:

```bash
uv run python scripts/plot_family_results.py \
  --methods bo_taf_m bo_taf_r \
  --split-path configs/family_splits/branin_split.json \
  --subset test \
  --taf-run-dir meta-bo-training/taf-gps/branin_train_v1 \
  --test-id taf_compare_family
```

Notes:
- Backward-compatible `bo_taf` still works and uses `--taf-weight-mode`.
- `bo_taf_m` / `bo_taf_r` are recommended for cleaner benchmark tracking.


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

Practical workflow convention:
- Keep using `test_results/` as the default for day-to-day experiments and quick iterations.
- Use `--results-dir benchmark_results` for large, milestone-style benchmark runs that you want to keep stable over time.
- Prefer unique `--test-id` values for archived runs in `benchmark_results/` to avoid accidental overwrite.

## Important benchmark results

We thoroughly compared the performance of our BO, implemented from scratch (`bo_scratch_grid` and `bo_scratch_multistart`), against BoTorch implementation (`bo_botorch`) on low-dimensional test functions. The results are stored in: 
- `benchmark_results/benchmark_scratch_botorch`
  - `/benchmark_BO_BoTorch_no_noise`: Benchmark our BO with BoTorch BO in functions without Gaussian noises in the output
  - `/benchmark_BO_BoTorch_with_noise`: Benchmark our BO with BoTorch BO in functions with Gaussian noises in the output
  - `/visual_search_behaviors_with_noise`: Detailed comparisons of the search behavior of different methods in these test functions


## Server-based Optimization Workflow

OpenBO supports a server-style optimization loop for external applications
that evaluate candidate designs outside this Python process.

Current server implementation:
- `bo_botorch` via WebSocket (`server_scripts/run_botorch_server.py`)

### 1) Start the optimizer server

```bash
uv run python server_scripts/run_botorch_server.py --host 127.0.0.1 --port 8765
```

### 2) WebSocket message protocol (JSON)

Client -> Server:
- `start`: create a new optimization session
- `suggest`: ask for the next design (optional after `start`, since server auto-sends first suggestion)
- `observe`: return evaluation result for the latest suggested design
- `status`: get current progress

Server -> Client:
- `suggest`: next design vector `x`
- `done`: optimization finished; returns full trajectory and best solution
- `status`: progress snapshot
- `error`: validation/protocol error

`start` payload example:

```json
{
  "type": "start",
  "bounds": [[0.0, 1.0], [0.0, 1.0]],
  "n_init": 5,
  "n_iter": 25,
  "seed": 0,
  "num_restarts": 5,
  "raw_samples": 64
}
```

`observe` payload example:

```json
{
  "type": "observe",
  "x": [0.12, 0.88],
  "y": -1.234
}
```

### 3) Minimal client loop (Python)

```python
import asyncio
import json
import websockets

async def objective(x):
    # Replace this with your external app evaluation.
    return -((x[0] - 0.25) ** 2 + (x[1] - 0.75) ** 2)

async def main():
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.send(json.dumps({
            "type": "start",
            "bounds": [[0.0, 1.0], [0.0, 1.0]],
            "n_init": 2,
            "n_iter": 8,
            "seed": 0
        }))

        msg = json.loads(await ws.recv())
        while msg["type"] != "done":
            if msg["type"] != "suggest":
                raise RuntimeError(msg)
            x = msg["x"]
            y = await objective(x)
            await ws.send(json.dumps({"type": "observe", "x": x, "y": y}))
            msg = json.loads(await ws.recv())

        print("best_value:", msg["best_value"])
        print("best_x:", msg["best_x"])

asyncio.run(main())
```

Notes:
- One WebSocket connection corresponds to one session.
- The server enforces ask/tell order (`suggest` then `observe`).
- `done` includes `x_values`, `y_values`, and `best_y_history` for logging/debugging.

### 4) Run provided fake Branin client

OpenBO includes a ready-to-run fake client that acts like an external application
and evaluates server-suggested points on Branin:

```bash
uv run python server_scripts/run_botorch_fake_client.py --uri ws://127.0.0.1:8765
```

Useful options:

```bash
uv run python server_scripts/run_botorch_fake_client.py \
  --uri ws://127.0.0.1:8765 \
  --n-init 2 \
  --n-iter 8 \
  --seed 0 \
  --save-json test_results/trajectories/fake_client_done.json
```


## Project structure

- `README.md` - project overview and usage.
- `pyproject.toml` - dependencies, build config, and project metadata.
- `configs/` - YAML configs and reusable artifacts.
  - `benchmark.yaml` - default benchmark config.
  - `methods/*.yaml` - method-level config placeholders.
  - `family_splits/*.json` - persisted train/test task-family splits.
- `src/openbo/` - main package (`import openbo`; the installable distribution name in `pyproject.toml` is `open-bo`, which `uv` maps to `openbo` via `[tool.uv.build-backend] module-name`).
  - `test_functions/synthetic.py` - synthetic objectives + optional Gaussian output noise and optimum capping.
  - `test_functions/transforms.py` - input transform helpers.
  - `test_functions/tasks.py` - task-variant spec + affine input/output wrappers (including variant-level noise/capping).
  - `test_functions/registry.py` - function metadata registry + optional noisy wrappers for single-function specs.
  - `test_functions/families.py` - family variant generation, train/test split, and split persistence.
  - `models/gp_scratch.py` - scratch GP with ARD kernels and per-step hyperparameter fitting.
  - `models/kernels.py` - RBF and Matérn-5/2 ARD kernels.
  - `models/botorch_gp.py`, `models/preference_gp.py` - placeholder model modules.
  - `acquisition/ei.py` - Expected Improvement implementations.
  - `acquisition/pi.py`, `acquisition/ucb.py`, `acquisition/taf.py`, `acquisition/conbo.py`, `acquisition/naf.py`, `acquisition/preference_acq.py` - placeholder acquisition modules.
  - `optimizers/random_search.py` - random-search baseline.
  - `optimizers/bo_scratch.py` - scratch BO loop (Sobol candidate scans + multistart L-BFGS-B EI maximization).
  - `optimizers/bo_botorch.py` - BoTorch BO loop (`SingleTaskGP` + `LogExpectedImprovement`).
  - `server_optimizers/bo_botorch_server.py` - WebSocket server adapter for BoTorch ask/tell optimization.
  - `optimizers/taf.py`, `optimizers/conbo.py`, `optimizers/naf.py`, `optimizers/pbo.py`, `optimizers/taf_pbo.py` - placeholder optimizer modules.
  - `benchmarks/runner.py` - single-function benchmark runner used by CLI scripts.
  - `benchmarks/seeds.py` - reproducibility helpers.
  - `benchmarks/metrics.py`, `benchmarks/plotting.py` - placeholder benchmark utilities.
- `scripts/` - command-line entrypoints.
  - `run_benchmark.py` - run one method on one function; supports `--noisy`, optional 2D x-location plotting, and TAF methods (`bo_taf_m`, `bo_taf_r`).
  - `plot_results.py` - single-function multi-method comparison plots (rerun or from stored trajectories), with optional `--noisy` rerun mode and TAF methods.
  - `create_family_split.py` - create and save train/test split for a task family.
  - `run_family_benchmark.py` - run one method across family tasks and save per-task trajectories; supports `--noisy`, TAF methods, and multi-mode TAF runs.
  - `plot_family_results.py` - family mean/std and best-so-far plots across methods (rerun or from stored trajectories); supports `--noisy` in rerun mode and TAF methods.
  - `plot_taf_gp_predictions.py` - render 2D GP mean/std heatmaps from saved TAF source-task GP states and trajectories.
  - `plot_taf_acquisition_heatmap.py` - visualize stored TAF acquisition query values and zero-mask behavior per iteration.
  - `aggregate_results.py` - placeholder aggregation script.
- `server_scripts/` - server-oriented command-line entrypoints.
  - `run_botorch_server.py` - run WebSocket server for external ask/tell optimization with `bo_botorch`.
  - `run_botorch_fake_client.py` - fake Branin client that exercises the server suggest/observe loop.
- `tests/` - test suite.
  - `test_functions_test.py` - synthetic functions, variants, and family split persistence tests.
  - `gp_scratch_test.py` - scratch GP fit/posterior tests.
  - `acquisition_test.py` - EI tests and BO smoke tests.
- `notebooks/` - teaching notebooks for BO concepts and step-by-step demos.

## Contact
This repo is created and maintained by Yi-Chi Liao (yichi.liao@inf.ethz.ch).