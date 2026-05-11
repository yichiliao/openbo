# OpenBO

OpenBO is a research + teaching Python library for Bayesian optimization (BO) and meta-BO.

## 0. Current scope

OpenBO targets **research and teaching** in **normalized** input space `[0, 1]^d`: synthetic objectives, standard BO, **transfer BO (TAF)** using saved source-task GPs, and optional **WebSocket** servers when the objective runs outside this process.

**Test functions and task families**

- Registered synthetics: `branin`, `sphere`, `ackley`, `rastrigin`, `rosenbrock`, `hartmann6` (dims per `registry`; Branin-style tasks are 2D).
- Optional **Gaussian output noise** and **capping at the optimum** for noisy benchmarks.
- **Affine variants** of a base function, **random families**, persisted **train/test splits** (`create_family_split`), and objectives built from those splits (`families` + `registry`).

**Optimizers and acquisition**

- **Random search** baseline.
- **BO from scratch**: NumPy GP + expected improvement; multistart L-BFGS-B search (and a grid-style variant) in `bo_scratch`.
- **BoTorch BO**: `SingleTaskGP` + `LogExpectedImprovement` in `bo_botorch`.
- **TAF** (`bo_taf`, `bo_taf_m`, `bo_taf_r`): transfer acquisition using source surrogates loaded from a directory of saved `gp_states/` + `trajectories/` (`bo_taf`).

**Benchmarks, trajectories, and plots**

- **Single-task** and **family** benchmark runners; JSON **trajectories** for single runs and per-task family runs.
- **Log-regret** plots (per-iteration and best-so-far; single-task and family mean ± std).
- Optional **2D search-path** plots over a function heatmap (`run_benchmark.py`, `run_fake_client.py`).
- TAF-specific tooling: **`train_taf`**, GP **prediction heatmaps**, **acquisition** query diagnostics (`plot_taf_*` scripts).

**Servers and integration**

- **Generic WebSocket BO server** (`run_bo_server`): `bo_scratch` or `bo_botorch`; optional **auto-save** of scratch trajectories and GP states for TAF.
- **Dedicated TAF WebSocket server** (`run_taf_server`): same `TAFSequentialOptimizer` as in-process TAF, ask/tell JSON protocol.
- **Fake Branin clients** (including `run_fake_client.py` and `scripts/run_botorch_fake_client.py` for BoTorch-oriented tests) and **manual family** client scripts (train variants on scratch, test variants on TAF, optional regret plots).

**Out of scope for now**

- Several other optimizer / acquisition filenames under `src/openbo` (e.g. `conbo`, `naf`, `pbo`) are **stubs** for future work; they are **not** exposed through the benchmark or server CLIs documented here.


## 1. Installation and quickstart

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

### Table of contents

- [0. Current scope](#0-current-scope)
- [1. Installation and quickstart](#1-installation-and-quickstart)
- [2. Single-function workflow](#2-single-function-workflow)
- [3. Family-of-functions workflow](#3-family-of-functions-workflow)
- [4. Meta-BO training and testing](#4-meta-bo-training-and-testing)
- [5. Server-based Optimization Workflow](#5-server-based-optimization-workflow)
- [6. End-to-end TAF workflow: family split → scratch GPs → TAF](#6-end-to-end-taf-workflow-family-split-scratch-gps-taf)
- [7. Results folder convention](#7-results-folder-convention)
- [8. Important benchmark results](#8-important-benchmark-results)
- [9. Project structure](#9-project-structure)
- [Contact](#contact)

## 2. Single-function workflow

We build everything from here: Run and compare optimizers on a **single** named test function (e.g. Branin): command-line benchmarks, trajectory JSON, optional noise and 2D search plots, and log-regret figures via `plot_results.py`.

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

## 3. Family-of-functions workflow

Before starting meta-learning or transfer learning, we need to create the settings for such tasks. Importantly, we need to work with **many related tasks** (variants of one base function): run one method across the family, save per-task trajectories, create persistent train/test splits, and plot mean/std log-regret with `plot_family_results.py`.

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

## 4. Meta-BO training and testing

Here we describe the **meta-BO loop** in-repo: define a task family and split, train on source tasks (e.g. scratch BO saving GPs and trajectories), then evaluate transfer methods such as TAF on held-out tasks using the same benchmark scripts as §2–3.

Generally meta-Bayesian optimization workflow follows these three steps:
- Step 1: Generate a family of tasks using the same base function, and then split the training tasks and testing tasks.
- Step 2: Train the optimizer with the training tasks.
- Step 3: Test the optimizer's performance on test tasks. 


### 4.1. Transfer Acquisition Function (TAF) workflow

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
  --methods random bo_taf_m bo_taf_r bo_botorch \
  --split-path configs/family_splits/branin_split.json \
  --subset test \
  --taf-run-dir meta-bo-training/taf-gps/branin_train_v1 \
  --test-id taf_compare_family
```

Notes:
- Backward-compatible `bo_taf` still works and uses `--taf-weight-mode`.
- `bo_taf_m` / `bo_taf_r` are recommended for cleaner benchmark tracking.


## 5. Server-based Optimization Workflow

Here, we provide the server-based interface for any systems to communicate with our optimizers via WebSOckets. With the steps below, you can run the same BO backends **over WebSockets** when your objective lives outside this process (simulator, service, or other language): start a server, speak the JSON ask/tell protocol, and use the included fake client for a quick smoke test. 

OpenBO supports a server-style optimization loop for external applications
that evaluate candidate designs outside this Python process.

Current server implementation:
- Generic BO server via WebSocket (`server_scripts/run_bo_server.py`)
- Supported backends in generic server: `bo_botorch`, `bo_scratch`
- Dedicated TAF server via WebSocket (`server_scripts/run_taf_server.py`)

### Step 1. Start the optimizer server

Use one of the backend-specific config files:
- `configs/server_optimizers/bo_server_botorch.yaml`
- `configs/server_optimizers/bo_server_scratch.yaml`
- `configs/server_optimizers/bo_taf_server.yaml`

Start server with BoTorch backend:

```bash
uv run python server_scripts/run_bo_server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --config-path configs/server_optimizers/bo_server_botorch.yaml
```

Start server with scratch backend:

```bash
uv run python server_scripts/run_bo_server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --config-path configs/server_optimizers/bo_server_scratch.yaml
```

Start dedicated TAF server:

```bash
uv run python server_scripts/run_taf_server.py \
  --host 127.0.0.1 \
  --port 8766 \
  --config-path configs/server_optimizers/bo_taf_server.yaml
```

The server reads the config file at session start.
Example BoTorch config:

```yaml
optimizer: bo_botorch
input_dim: 2
y_range: [-350.0, 1.0]
n_init_default: 2
n_iter_default: 8
num_restarts_default: 5
raw_samples_default: 64
```

Example scratch config:

```yaml
optimizer: bo_scratch
input_dim: 2
y_range: [-350.0, 1.0]
n_init_default: 2
n_iter_default: 8
auto_save_scratch_artifacts: true
scratch_artifacts_dir: meta-bo-training/server-scratch-artifacts
```

Example TAF config:

```yaml
input_dim: 2
y_range: [-350.0, 1.0]
taf_run_dir: meta-bo-training/taf-gps/branin_train_v1
taf_weight_mode_default: taf_m  # or taf_r
taf_rho_default: 1.0
n_init_default: 0
n_iter_default: 8
```

Assumptions for server optimization:
- input is normalized to `[0, 1]^d`
- optimization is **maximization**
- you must set `input_dim` and `y_range` in the config file for your application
- current default config is Branin-oriented (`y = -Branin(x)`), so `y_range` is set for flipped Branin scale
- choose backend with `optimizer: bo_botorch` or `bo_scratch` for the generic server
- use dedicated TAF server for `bo_taf` and set `taf_run_dir` in config
- for `bo_scratch`, you can auto-save TAF-compatible `trajectories/*.json` and `gp_states/*.json`
  using `auto_save_scratch_artifacts: true`

### Step 2. WebSocket message protocol (JSON)

Client -> Server:
- `start`: create a new optimization session
- `suggest`: ask for the next design (optional after `start`, since server auto-sends first suggestion)
- `observe`: return evaluation result for the latest suggested design
- `status`: get current progress
- `stop`: stop current session early and return partial results

Server -> Client:
- `suggest`: next design vector `x`
- `done`: optimization finished; returns full trajectory and best solution
- `status`: progress snapshot
- `error`: validation/protocol error

`start` payload example:

```json
{
  "type": "start",
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

`stop` payload example:

```json
{
  "type": "stop",
  "reason": "user_cancelled"
}
```

### Step 3. Minimal client loop (Python)

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
- `stop` returns a `stopped` payload with partial trajectory and best-so-far values.
- `observe.y` must stay within configured `y_range`.

### Step 4. Run provided fake Branin client

OpenBO includes a ready-to-run fake client that acts like an external application
and evaluates server-suggested points on Branin:

```bash
uv run python server_scripts/run_fake_client.py --uri ws://127.0.0.1:8765
```

Useful options:

```bash
uv run python server_scripts/run_fake_client.py \
  --uri ws://127.0.0.1:8765 \
  --n-init 2 \
  --n-iter 8 \
  --seed 0 \
  --save-json test_results/trajectories/fake_client_done.json
```

For dedicated TAF server testing:

```bash
uv run python server_scripts/run_fake_client.py \
  --uri ws://127.0.0.1:8766 \
  --n-init 0 \
  --n-iter 10 \
  --seed 0 \
  --save-json test_results/trajectories/fake_client_taf_done.json \
  --plot-x-locations
```

This writes `test_results/trajectories/fake_client_taf_done_x_locations.png` next to the JSON
(same 2D heatmap + iteration-colored points as `scripts/run_benchmark.py --plot-x-locations`).
Use `--plot-output path/to/plot.png` to choose the PNG path; Branin is assumed (`input_dim: 2`).

## 6. End-to-end TAF workflow: family split → scratch GPs → TAF

Tutorial and example of using TAF for your own applications. Walk through the **manual** end-to-end pipeline for transfer BO: create a split, train source tasks through the scratch server (saving `gp_states/` and `trajectories/`), point the TAF server at that run directory, optimize test tasks, and optionally plot regret—substituting your own client where the examples use Branin or family helpers. 

### Step 1. Create a train/test split

From the repo root:

```bash
uv run python scripts/create_family_split.py \
  --base-function branin \
  --n-tasks 15 \
  --train-ratio 0.8 \
  --family-seed 0 \
  --output configs/family_splits/branin_split_15.json
```

The command prints `n_train` and `n_test` (expect 9 and 6). The JSON contains
`train_variants` and `test_variants` used in the next steps.

### Step 2. Configure `run_bo_server` (scratch) to auto-save TAF-style artifacts

Use a scratch server YAML that enables artifact export and points at a **single**
directory that will later be your TAF `taf_run_dir` (GPs and trajectories are written
under `gp_states/` and `trajectories/` inside that directory). Example:

```yaml
optimizer: bo_scratch
input_dim: 2
y_range: [-350.0, 1.0]
n_init_default: 5
n_iter_default: 25
auto_save_scratch_artifacts: true
scratch_artifacts_dir: meta-bo-training/taf-gps/branin_server_end_to_end
```

Save it as e.g. `configs/server_optimizers/bo_taf_server_train_end_to_end.yaml` and create
`meta-bo-training/taf-gps/branin_server_end_to_end` if you want an empty parent folder (the server
creates `gp_states` and `trajectories` on first save).

### Step 3. Start the generic BO server (scratch) and keep it running

Pick ports that are free (if `8767` is already in use, choose another port).

```bash
uv run python server_scripts/run_bo_server.py \
  --host 127.0.0.1 \
  --port 8767 \
  --config-path configs/server_optimizers/bo_taf_server_train_end_to_end.yaml
```

### Step 4. Run one WebSocket client per **training** variant (9 sessions)

For each training task you must:

- Open a **new** WebSocket connection (one session per variant).
- Send a `start` message that includes a **unique** `task_name` (e.g. `train_task_000`, …,
  `train_task_008`) so saved `gp_states/<name>.json` and `trajectories/<name>.json` do
  not overwrite each other. The server also uses `n_init`, `n_iter`, and `seed` from
  `start` (or defaults from the YAML).
- On each `suggest`, compute `y` with **that task’s** objective, not the base Branin
  registry function.

From the repo root, with `run_bo_server` already running (adjust `--uri` if you did not
use port `8765`):

```bash
uv run python server_scripts/run_manual_family_train_clients.py \
  --split-path configs/family_splits/branin_split_15.json \
  --uri ws://127.0.0.1:8767 \
  --n-init 5 \
  --n-iter 25 \
  --seed-base 0
```

Implementation: `server_scripts/run_manual_family_train_clients.py` (loads the split,
builds train specs with prefix `train_task`, sends `task_name` per session).

After all nine sessions, you should see nine files under
`meta-bo-training/taf-gps/branin_server_end_to_end/gp_states/` (and matching files under
`trajectories/`).

You can optionally visualize GP predictions as 2D heatmaps (mean/std):

```bash
uv run python scripts/plot_taf_gp_predictions.py \
  --run-dir meta-bo-training/taf-gps/branin_server_end_to_end
```

### Step 5. Stop the scratch server for training

Stop the `run_bo_server` process (Ctrl+C in the terminal where it runs).

### Step 6. Configure and start the TAF server for testing

Set `taf_run_dir` in your TAF server YAML to the **same** directory you used for
`scratch_artifacts_dir` (the folder that now contains `gp_states/` and
`trajectories/`). Example `configs/server_optimizers/bo_taf_server.yaml` (or a copy)
with:

```yaml
taf_run_dir: meta-bo-training/branin_server_end_to_end
```

Start the TAF server on a **different** port if you will run it on the same machine
immediately after the scratch server (for example `8766`):

```bash
uv run python server_scripts/run_taf_server.py \
  --host 127.0.0.1 \
  --port 8766 \
  --config-path configs/server_optimizers/bo_taf_server_test_end_to_end.yaml
```

### Step 7. Run one client per **test** variant

TAF does not require `task_name` in `start` for source filenames, but you must still
evaluate the **test** task’s Branin variant for each session. Use `test_task` as the
prefix when building specs (names `test_task_000` … `test_task_005`).

```bash
uv run python server_scripts/run_manual_family_test_clients.py \
  --split-path configs/family_splits/branin_split_15.json \
  --uri ws://127.0.0.1:8766 \
  --n-init 0 \
  --n-iter 30 \
  --seed-base 100 \
  --save-results-dir test_results/branin_server_end_to_end/test_sessions
```

The optional `--save-results-dir` writes one JSON file per test session (the terminal
`done` / `stopped` server payload), including `best_y_history`, for Step 8 plots.
If you already ran Step 7 without it, run Step 7 again with `--save-results-dir` (or
save trajectories yourself from client logs).

Implementation: `server_scripts/run_manual_family_test_clients.py`.

### Step 8. Visualize performance from trajectories (optional)

You already have **training** trajectories on disk: under your scratch artifact directory,
`trajectories/train_task_000.json`, … (each includes `y_values` and `best_so_far`).
For **test** runs, use JSON saved via `--save-results-dir` in Step 7 (each file lists
`best_y_history` from the server).

The plotting script uses the **same family split JSON** as Step 1 to recover each task’s
**optimal value**, then plots **log10(regret)** for maximization, matching
`scripts/plot_family_results.py`:

`log10(max(optimal_value - best_so_far_y, 1e-12))` versus iteration (1-based).

**How to read train vs test on the same figure (important):** the green curve averages
**scratch BO on train variants**; the purple curve averages **TAF on test variants**. Those
are **different tasks** drawn from the same random family, so similar mean log-regret is
**normal** — you are not comparing two methods on the **same** targets. This plot checks
that both stages behave sensibly; it does **not** by itself show transfer gain. To see
transfer, compare methods on the **same** test tasks (for example
`scripts/plot_family_results.py --split-path ... --subset test` with `bo_scratch` and
`bo_taf`, or run scratch `run_bo_server` sessions on the test variants with the same
budget and plot those trajectories alongside TAF).

**Mean ± 1 std** (default): one figure (9×5 in.), same presentation as
`scripts/plot_family_results.py` — mean line plus shaded band per group; **train** (scratch)
and **test** (TAF) on the same axes when both directories are given.

```bash
uv run python server_scripts/plot_manual_family_trajectories.py \
  --split-path configs/family_splits/branin_split_15.json \
  --train-trajectories-dir meta-bo-training/taf-gps/branin_server_end_to_end/trajectories \
  --test-results-dir test_results/branin_server_end_to_end/test_sessions \
  --output test_results/branin_server_end_to_end/log_regret_mean_std.png
```

**Per-task curves** (optional):

```bash
uv run python server_scripts/plot_manual_family_trajectories.py \
  --split-path configs/family_splits/branin_split_15.json \
  --train-trajectories-dir meta-bo-training/branin_server_end_to_end/trajectories \
  --test-results-dir test_results/branin_server_end_to_end/test_sessions \
  --plot-mode per_task \
  --output test_results/manual_family/log_regret_per_task.png
```

Either `--train-trajectories-dir` or `--test-results-dir` may be omitted if you only want
one panel. Task names in the JSON must match `train_task_*` / `test_task_*` from the same
split (same `--task-prefix` as the manual client scripts if you changed them).

Other ideas (not wired in this script): inspect raw `y_values` in the JSON, compare
`final_best` across tasks, or run `scripts/plot_taf_gp_predictions.py` on your
`gp_states/` folder for 2D GP heatmaps.

**Summary**

| Stage | Server | Client objective | `start` extras |
|------|--------|------------------|----------------|
| Train sources | `run_bo_server` (scratch, auto-save on) | Each **train** variant | Required unique `task_name` |
| Test targets | `run_taf_server` | Each **test** variant | Optional fields only |

### Server notes and troubleshooting

Server config precedence:
- `input_dim` and `y_range` are read from `--config-path` YAML at session start.
- The `start` message does not set bounds/dimension in server mode.
- Bounds are internally assumed as normalized `[0, 1]^d` using configured `input_dim`.
- Optional for scratch backend:
  - `auto_save_scratch_artifacts: true`
  - `scratch_artifacts_dir: <path>`
  - optional `start.task_name` to control output filename (`<task_name>.json`).

Scratch -> TAF two-stage flow:
- Run generic server with `optimizer: bo_scratch` and `auto_save_scratch_artifacts: true`.
- Evaluate tasks via your client; server saves `trajectories/` and `gp_states/`.
- Set `taf_run_dir` in `configs/server_optimizers/bo_taf_server.yaml` to that directory.
- Run dedicated TAF server (`run_taf_server.py`) for target-task optimization.

Stop semantics:
- `{"type":"stop"}` stops only the current optimization session.
- The server process stays alive and can accept new client sessions.

`y_range` validation:
- If client sends `observe.y` outside configured `y_range`, server replies with:
  - `{"type":"error","message":"...outside configured y_range..."}`

Expected happy-path:
- Start server and run fake client.
- You should see client logs ending with terminal `type=done`
  (or `type=stopped` if you explicitly stop).

Troubleshooting:
- **Port in use**: change `--port` when starting server.
- **Dimension mismatch**: update `input_dim` in config to match your application input size.
- **Frequent y-range errors**: widen `y_range` in config to include your observed objective scale.

Architecture note:
- `openbo.optimizers.bo_botorch` and `openbo.optimizers.bo_scratch` are served by `server_optimizers/bo_server.py`.
- `openbo.optimizers.bo_taf` is served by `server_optimizers/bo_taf_server.py` to keep TAF-specific config isolated. Random init is sampled once (same RNG draw pattern as `bootstrap()`), committed as one batched `observe`, and `n_iter` matches `run_bo_taf` (including the extra internal budget step when `n_init > 0`).


## 7. Results folder convention

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

## 8. Important benchmark results

**Purpose:** Point to **saved benchmark campaigns** (scratch vs BoTorch, with and without output noise) and visual search-behavior comparisons you can browse without rerunning optimizers.

We thoroughly compared the performance of our BO, implemented from scratch (`bo_scratch_grid` and `bo_scratch_multistart`), against BoTorch implementation (`bo_botorch`) on low-dimensional test functions. The results are stored in: 
- `benchmark_results/benchmark_scratch_botorch`
  - `/benchmark_BO_BoTorch_no_noise`: Benchmark our BO with BoTorch BO in functions without Gaussian noises in the output
  - `/benchmark_BO_BoTorch_with_noise`: Benchmark our BO with BoTorch BO in functions with Gaussian noises in the output
  - `/visual_search_behaviors_with_noise`: Detailed comparisons of the search behavior of different methods in these test functions




## 9. Project structure

- `README.md` - project overview and usage.
- `pyproject.toml` - dependencies, build config, and project metadata.
- `configs/` - YAML configs and reusable artifacts.
  - `benchmark.yaml` - default benchmark config.
  - `methods/*.yaml` - method-level config placeholders.
  - `family_splits/*.json` - persisted train/test task-family splits.
  - `server_optimizers/*.yaml` - server runtime configs (e.g. input dimension and y-range constraints).
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
  - `server_optimizers/bo_server.py` - generic WebSocket server adapter for ask/tell optimization backends (`bo_botorch` and `bo_scratch`).
  - `server_optimizers/bo_taf_server.py` - dedicated WebSocket server adapter for TAF ask/tell optimization (`bo_taf`).
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
  - `run_bo_server.py` - run generic WebSocket server for external ask/tell optimization (`bo_botorch` or `bo_scratch` via config/start message).
  - `run_fake_client.py` - fake Branin client that exercises the server suggest/observe loop; optional `--plot-x-locations` (2D, same style as `run_benchmark.py`).
  - `run_family_scratch_then_taf_demo.py` - standalone demo: `create_family_split.py` (15 Branin variants, 9 train / 6 test), scratch server to save source GPs, then TAF server on test variants.
  - `run_manual_family_train_clients.py` - README §4c Step 4: one websocket session per train variant from a saved split (for use with `run_bo_server` + auto-save).
  - `run_manual_family_test_clients.py` - README §4c Step 7: one session per test variant (for use with `run_taf_server`); optional `--save-results-dir` for Step 8 plots.
  - `plot_manual_family_trajectories.py` - README §4c Step 8: plot log10(regret) (same definition as `plot_family_results`) from train `trajectories/*.json` and/or saved test session JSON; requires `--split-path` for optima; default **mean ± 1 std** on one panel (style-aligned with `plot_family_results`).
- `tests/` - test suite.
  - `test_functions_test.py` - synthetic functions, variants, and family split persistence tests.
  - `gp_scratch_test.py` - scratch GP fit/posterior tests.
  - `acquisition_test.py` - EI tests and BO smoke tests.
- `notebooks/` - teaching notebooks for BO concepts and step-by-step demos.

## Contact

This repo is created and maintained by Yi-Chi Liao (yichi.liao@inf.ethz.ch).