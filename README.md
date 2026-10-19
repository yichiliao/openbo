# OpenBO

OpenBO is a research + teaching Python library for Bayesian optimization (BO): standard BO, transfer and meta-BO (TAF, TAF+, continual BO, MetaBO-style neural acquisition functions) and multi-objective BO.

**Website: [https://www.csie.ntu.edu.tw/~yichi/openbo-website/](https://www.csie.ntu.edu.tw/~yichi/openbo-website/index.html)**. The website has the overview: [installation and a first run](https://www.csie.ntu.edu.tw/~yichi/openbo-website/get-started.html), [what each method does and the paper it follows](https://www.csie.ntu.edu.tw/~yichi/openbo-website/methods.html), [interactive demos](https://www.csie.ntu.edu.tw/~yichi/openbo-website/demos.html), [benchmark results](https://www.csie.ntu.edu.tw/~yichi/openbo-website/benchmarks.html) and [how to cite OpenBO and the papers](https://www.csie.ntu.edu.tw/~yichi/openbo-website/publications.html). This README is the reference: every script, flag, server protocol and archived run.

### Table of contents

- [0. Current scope](#0-current-scope)
- [1. Installation](#1-installation)
- [2. Single-function workflow](#2-single-function-workflow)
- [3. Family-of-functions workflow](#3-family-of-functions-workflow)
- [4. Meta-BO training and testing](#4-meta-bo-training-and-testing)
- [5. Server-based Optimization Workflow](#5-server-based-optimization-workflow)
- [6. End-to-end TAF workflow: family split → scratch GPs → TAF](#6-end-to-end-taf-workflow-family-split--scratch-gps--taf)
- [7. Multi-objective BO](#7-multi-objective-bo)
- [8. Continual BO (ConBO)](#8-continual-bo-conbo)
- [9. TAF+: transfer across users with several objectives](#9-taf-transfer-across-users-with-several-objectives)
- [10. NAF (MetaBO): a learned acquisition function](#10-naf-metabo-a-learned-acquisition-function)
- [11. Results folder convention](#11-results-folder-convention)
- [12. Archived benchmark results](#12-archived-benchmark-results)
- [13. Project structure](#13-project-structure)
- [License and citation](#license-and-citation)

## 0. Current scope

The methods, test functions and servers are summarized on the website's [Methods](https://www.csie.ntu.edu.tw/~yichi/openbo-website/methods.html) and [Get started](https://www.csie.ntu.edu.tw/~yichi/openbo-website/get-started.html) pages, and each module is listed in Section 13.

**Out of scope for now**

- Several modules under `src/openbo` are **stubs or placeholders** for future work and are **not** exposed through the benchmark or server CLIs documented here:
  - placeholder functions: `acquisition/pi.py`, `acquisition/ucb.py`, `models/botorch_gp.py`, `benchmarks/metrics.py`, `benchmarks/plotting.py`, `scripts/aggregate_results.py`
- `configs/methods/*.yaml` and `configs/benchmark.yaml` are placeholders; no script reads them. Scripts are configured through command-line flags (and server YAMLs for the WebSocket servers).
- NAF: MetaBO's GP-prior and real-world tasks are not implemented.
- The teaching notebooks are still empty.

## 1. Installation

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

All commands in this README assume either `uv run ...` or an activated `.venv`, and are run **from the repository root** (default config and output paths are relative).

The conventions (maximization everywhere, inputs in `[0, 1]^d`, array shapes, the regret metric) are on the website's [Get started](https://www.csie.ntu.edu.tw/~yichi/openbo-website/get-started.html#conventions) page. Known optima are in `KNOWN_OPTIMA` (`openbo.test_functions.synthetic`).

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

Plot from stored single-run trajectories (no optimizer rerun). Stored mode needs one
trajectory per method under the same `--test-id`; the example above saved only
`bo_scratch_multistart`, so first store the other three:

```bash
for m in random bo_scratch_grid bo_botorch; do
  uv run python scripts/run_benchmark.py --method $m --function branin \
    --test-id exp_single_001 --results-dir test_results
done
```

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
uv run python scripts/run_family_benchmark.py --method bo_scratch_multistart --n-tasks 10 --test-id family10
```

Without `--test-id` the id is `default`, and the run is written to
`test_results/trajectories/default_bo_scratch_multistart_branin_all/`, which
overwrites the committed run of that name (5 tasks; see the last examples of this section).

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

## 4. Meta-BO training and testing

Here we describe the **meta-BO loop** in-repo: define a task family and split, train on source tasks (e.g. scratch BO saving GPs and trajectories), then evaluate transfer methods such as TAF on held-out tasks using the same benchmark scripts as §2–3.

NAF (MetaBO) follows the same three steps with its own family, training script and benchmark; see Section 10.

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

Note: this TAF differs from the original (Wistuba et al.) in how a source's improvement is measured. Each source is compared with a fixed 90% quantile of its own data, through a softplus, instead of with its best prediction over the points the target has already evaluated. So the sources keep pulling even after the target has explored their optima, and with many sources TAF can stall late in a run (see the ConBO benchmark in Section 12). `bo_taf_m` also weights sources by meta-feature similarity, not by predictive variance as in the original TAF-ME.

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
- TAF loads one source GP per `gp_states/<task>.json` file under `--taf-run-dir`, including subfolders (`gp_states/<sub>/<task>.json` is paired with `trajectories/<sub>/<task>.json`). A task without a matching trajectory is skipped with a warning. Two tasks with identical data (for example a backup copy of a run kept inside `gp_states/`) both load, and each repeat triggers a warning because it counts the same task twice. Only `gp_states/` is scanned. Check the number of sources (`n_sources` in the run metadata) if results look off.
- The committed `meta-bo-training/taf-gps/branin_train_v1` has all 12 tasks at the top level of `gp_states/` and `trajectories/`, the same layout `train_taf.py` writes, so re-running the training command overwrites the files instead of adding a second copy. (Earlier, 11 of the 12 tasks sat in a nested `train_task_0/` subfolder and the loader read only the top level, so TAF runs pointed at this directory used a **single** source. The output of the TAF fake client and server with their default config, which point at this run, came from that single-source version.)

## 5. Server-based Optimization Workflow

Every optimizer family has a WebSocket server, for objectives that live outside this process (a study app, a simulator, another language). One WebSocket connection is one session. The website's [Get started](https://www.csie.ntu.edu.tw/~yichi/openbo-website/get-started.html#servers) page lists the servers with their default ports and configs, and has a minimal client. This section is the full reference.

### Step 1. Start the optimizer server

Use one of the backend-specific config files:
- `configs/server_optimizers/bo_server_botorch.yaml`
- `configs/server_optimizers/bo_server_scratch.yaml`
- `configs/server_optimizers/bo_taf_server.yaml`
- `configs/server_optimizers/mobo_server.yaml`
- `configs/server_optimizers/conbo_server.yaml`
- `configs/server_optimizers/taf_plus_server.yaml`
- `configs/server_optimizers/naf_server.yaml` (Branin; also `naf_server_goldstein_price.yaml` and `naf_server_hartmann3.yaml` for the other two families)

`run_bo_server.py` has no usable default config (its fallback path,
`configs/server_optimizers/bo_server.yaml`, does not exist), so always pass
`--config-path` explicitly.

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

Start the dedicated MOBO server (the default config is set up for OKA2):

```bash
uv run python server_scripts/run_mobo_server.py \
  --host 127.0.0.1 \
  --port 8768 \
  --config-path configs/server_optimizers/mobo_server.yaml
```

Start the dedicated ConBO server. The population lives in the config's
`memory_dir` and survives restarts:

```bash
uv run python server_scripts/run_conbo_server.py \
  --host 127.0.0.1 \
  --port 8769 \
  --config-path configs/server_optimizers/conbo_server.yaml
```

Start the dedicated TAF+ server. The population is read from the config's
`population_dir` and is never changed by a session:

```bash
uv run python server_scripts/run_taf_plus_server.py \
  --host 127.0.0.1 \
  --port 8770 \
  --config-path configs/server_optimizers/taf_plus_server.yaml
```

Start the dedicated NAF server. The trained AF is read from the config's
`af_dir` and is never changed by a session:

```bash
uv run python server_scripts/run_naf_server.py \
  --host 127.0.0.1 \
  --port 8771 \
  --config-path configs/server_optimizers/naf_server.yaml
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

For `bo_scratch`, search settings (`n_candidates`, `n_starts`,
`search_strategy`, `kernel_type`, `optimize_hyperparameters`) are read only
from the `start` message; the `*_default` keys in `bo_server_scratch.yaml` are
currently ignored by the generic server. The TAF server does honor its
`*_default` keys.

Example TAF config (the shipped `bo_taf_server.yaml`):

```yaml
input_dim: 2
y_range: [-350.0, 1.0]
taf_run_dir: meta-bo-training/taf-gps/branin_train_v1
n_init_default: 0
n_iter_default: 10
n_candidates_default: 512
n_starts_default: 8
search_strategy_default: multistart
kernel_type_default: matern52
optimize_hyperparameters_default: true
taf_weight_mode_default: taf_r  # or taf_m
taf_rho_default: 1.0
source_only_warmup_iters_default: 3
track_acquisition_default: true
```

Example MOBO config (`mobo_server.yaml`, OKA2). `y_range` holds one
`[min, max]` per objective, and the hypervolume reference box defaults to it:

```yaml
input_dim: 3
y_range: [[-3.2, 3.1416], [-5.1, 0.0]]
model: slf            # independent | mt | slf
acquisition: ceipv    # ceipv | ieipv
hyperparameters: map  # map | slice
n_init_default: 5
n_iter_default: 20
n_candidates_default: 1024
```

Example ConBO config (`conbo_server.yaml`). The `conbo:` settings are used only
when `memory_dir` holds no population yet; a saved population keeps the
settings it was created with:

```yaml
input_dim: 2
y_range: [-350.0, 1.0]   # one scalar per evaluation, maximized
memory_dir: meta-bo-training/conbo-memory/branin_demo
n_evaluations_default: 20
conbo:
  r0: 6                          # random starts for user 1, decreasing by dr per user
  dr: 2
  alpha1: 5.0                    # population weight held at 1 for 5 evaluations,
  alpha2: 0.2                    # then falls by 0.2 per evaluation
  replay_variance_fraction: 0.05 # scale-free replay filter; or replay_max_variance (raw units)
  population_epochs: 1200
  online_epochs: 15
  candidate_grid_points: 40
  seed: 0
```

Example TAF+ config (`taf_plus_server.yaml`, set up for the CHI 2024 App. B
sphere users and their archived population):

```yaml
input_dim: 4
y_range: [[-12.0, 2.0], [-12.0, 2.0], [-12.0, 2.0]]   # one [min, max] per objective
population_dir: benchmark_results/taf_plus_sims/populations/v1_sim2_r01_s0
objective_weights_default: [0.3, 0.5, 0.2]   # alpha; must sum to 1
decay_default: true
d1_default: 8          # decay (d1, d2); this population's search result
d2_default: 0.1
weight_mode_default: poe   # poe | taf_r
rho_default: 1.0
n_iter_default: 10
n_candidates_default: 1024
refine_suggestion_default: true
first_point_default: population   # population | centre
```

`population_dir` must hold a population written by
`TAFPlusPopulation.save` (Section 9.2) or by
`scripts/run_taf_plus_benchmark.py --method population` (under
`test_results/taf_plus/populations/`). Its input dimension and number of
objectives must match `input_dim` and `y_range`. The new user's GPs use
the population's kernel.

Example NAF config (`naf_server.yaml`, set up for the archived Branin AF):

```yaml
input_dim: 2
y_range: [-12.0, 2.0]   # Branin-family tasks stay inside about [-10.3, 1.16]
af_dir: benchmark_results/naf_branin/training/branin_metabo50_v1
af_name: best_af        # best_af (best training return) | af (last iteration)
n_iter_default: null    # null = the AF's training budget T (30 here)
```

`af_dir` must hold a `scripts/train_naf.py` output folder (`<af_name>.pt`
and `naf_config.json`), and the AF's input dimension must match `input_dim`.
The GP hyperparameters and the budget come from the AF's own config.

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
- `suggest`: not needed. The server sends the next design after `start` and after each `observe`, and a `suggest` sent while a design is pending returns an `error`.
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

### Step 3. Per-server protocols and client loops (Python)

A minimal client for the generic BO server is on the website's Get started page. Notes that hold for every server:

- One WebSocket connection corresponds to one session.
- The server enforces ask/tell order (`suggest` then `observe`).
- `done` includes `x_values`, `y_values`, and `best_y_history` for logging/debugging.
- `stop` returns a `stopped` payload with partial trajectory and best-so-far values.
- `observe.y` must stay within configured `y_range`.

#### MOBO server protocol

The MOBO server (`run_mobo_server.py`) uses the same message types as the
generic BO server, but `observe.y` is a **list with one value per objective**
(all maximized):

```json
{
  "type": "observe",
  "x": [0.31, 0.72, 0.05],
  "y": [1.84, -3.97]
}
```

- The config's `y_range` holds one `[min, max]` per objective. Every value must
  lie inside its range, and a scalar `y` or one of the wrong length is rejected.
- `y_range` also sets the hypervolume reference box. `v_ref` defaults to the
  minimums and `w_ref` to the maximums; both can be set in the YAML.
- `start` may override `model` (`independent` / `mt` / `slf`), `acquisition`
  (`ceipv` / `ieipv`), `hyperparameters` (`map` / `slice`), `n_candidates`,
  `n_rescore`, `n_init` (at least 1), `n_iter` and `seed`.
- `done` and `stopped` return `x_values`, `y_values` (n × L), `pareto_x`,
  `pareto_y`, and `hv_history` (one entry per observation).
- `server_scripts/run_mobo_fake_client.py --function oka2` is a ready-made
  client (see Section 7).

#### MOBO client loop

Like the generic BO loop, but `y` is a list with one value per objective:

```python
import asyncio
import json
import websockets

async def evaluate(x):
    # Replace with your app: one value per objective, all maximized,
    # each inside the config's y_range.
    return [-(x[0] - 0.3) ** 2, -(x[1] - 0.7) ** 2]

async def main():
    async with websockets.connect("ws://127.0.0.1:8768") as ws:
        await ws.send(json.dumps({"type": "start", "n_init": 5, "n_iter": 10, "seed": 0}))
        msg = json.loads(await ws.recv())
        while msg["type"] == "suggest":
            y = await evaluate(msg["x"])
            await ws.send(json.dumps({"type": "observe", "x": msg["x"], "y": y}))
            msg = json.loads(await ws.recv())
        if msg["type"] == "error":
            raise RuntimeError(msg["message"])
        print("Pareto front:", msg["pareto_y"])
        print("final hypervolume:", msg["hv_history"][-1])

asyncio.run(main())
```

#### ConBO server protocol

The ConBO server (`run_conbo_server.py`) uses the same message types as the
generic BO server, with a scalar `y` checked against `y_range`. What differs is
that **one session is one user**, and the population outlives the session:

- **One user at a time.** ConBO must finish users in the order it started them,
  so while a session is open, another `start` gets an `error`. Run participants
  one after another.
- **`start`** may set `user_name` (letters, digits, `_`, `.`, `-`; it must be new
  to the memory; default `user_000`, `user_001`, ...) and `n_evaluations` (this
  user's budget). There is no `n_init`: the user's random starts follow the
  population's schedule (`r0`, `dr`).
- **`suggest`** carries `x`, `phase` (`random` or `bo`), `population_weight`
  (`null` during random starts, then the population's share of the
  acquisition), `user_name` and `user_index`.
- **`done`** arrives when the budget is used up. The user is stored first
  (replay retraining plus save, about 10 s), and `done` reports `stored`,
  `n_users_in_memory` and the `replay` report, besides the usual `x_values`,
  `y_values`, `best_x`, `best_value` and `best_y_history`.
- **`stop`** discards the user by default (the population is unchanged). Send
  `{"type": "stop", "store": true}` to keep a partial session with at least one
  observation. A dropped connection also discards the user.
- `status` reports `n_users_in_memory`, `n_random` and the current
  `population_weight`.
- The memory directory uses the TAF layout, so it can also be passed to the TAF
  server as `taf_run_dir`.

#### ConBO client loop

The same loop as the generic BO client; run it once per participant:

```python
import asyncio
import json
import websockets

async def run_trial(x):
    # Replace with your experiment: one scalar, maximized, inside y_range.
    return -((x[0] - 0.25) ** 2 + (x[1] - 0.75) ** 2)

async def main(user_name):
    async with websockets.connect("ws://127.0.0.1:8769") as ws:
        await ws.send(json.dumps({"type": "start", "user_name": user_name, "n_evaluations": 15}))
        msg = json.loads(await ws.recv())
        while msg["type"] == "suggest":
            # msg["phase"] is "random" or "bo"; msg["population_weight"] shows the handover
            y = await run_trial(msg["x"])
            await ws.send(json.dumps({"type": "observe", "x": msg["x"], "y": y}))
            msg = json.loads(await ws.recv())
        if msg["type"] == "error":
            raise RuntimeError(msg["message"])
        print("best:", msg["best_value"], msg["best_x"])
        print("users in memory:", msg["n_users_in_memory"])

asyncio.run(main("participant_01"))
```

#### TAF+ server protocol

The TAF+ server (`run_taf_plus_server.py`) uses the same message types as the
MOBO server: `observe.y` is a list with one value per objective, each checked
against its `y_range`. **One session is one new user**:

- **`start`** may override `objective_weights` (one per objective, summing to
  1), `decay`, `d1`, `d2`, `n_iter`, `n_candidates`, `weight_mode`, `rho`,
  `first_point`, `refine_suggestion` and `seed`. There is no `n_init`: the
  first design comes from the population.
- **`suggest`** carries `x`, `phase` (`population` for the first design, then
  `taf_plus`), `iteration` and `decay` (`null` for the first design, then
  `δ(t)`, the population's weight factor).
- **`done`** and **`stopped`** carry `x_values`, `y_values` (`n × L`),
  `weighted_y` (`y @ objective_weights`), `best_so_far`, `decay_history`,
  `objective_weights` and `population_users`.
- `status` reports `n_observations`, `best_weighted_y` and
  `n_population_users`.
- The population is loaded at each `start` and never changed: finished users
  are not added to it. To grow it, add their data with
  `TAFPlusPopulation.add_user` and save it again.

#### TAF+ client loop

```python
import asyncio
import json
import websockets

async def run_trial(x):
    # Replace with your experiment: one value per objective, all maximized.
    return [1.0 - 8.0 * ((x[0] - 0.55) ** 2 + (x[1] - 0.4) ** 2), 0.0, 0.0]

async def main():
    async with websockets.connect("ws://127.0.0.1:8770") as ws:
        await ws.send(json.dumps({"type": "start", "objective_weights": [0.3, 0.5, 0.2], "n_iter": 10}))
        msg = json.loads(await ws.recv())
        while msg["type"] == "suggest":
            # msg["phase"] is "population" or "taf_plus"; msg["decay"] is delta(t)
            y = await run_trial(msg["x"])
            await ws.send(json.dumps({"type": "observe", "x": msg["x"], "y": y}))
            msg = json.loads(await ws.recv())
        if msg["type"] == "error":
            raise RuntimeError(msg["message"])
        print("best weighted value:", msg["best_so_far"][-1])

asyncio.run(main())
```

#### NAF server protocol

The NAF server (`run_naf_server.py`) uses the same message types as the TAF
server, with a scalar `observe.y` checked against `y_range`. **One session is
one task**:

- **`start`** may set `n_iter`, from 1 up to the AF's training budget `T`
  (default: the config's `n_iter_default`, or `T`). There is no `n_init` and
  no seed: the AF chooses every point, greedily, so a session is deterministic
  given the observed values.
- **`suggest`** carries `x`, `iteration`, `n_iter` and `budget` (`T`).
- **`done`** and **`stopped`** carry `x_values`, `y_values`, `best_so_far`,
  `budget` and `af_dir`.
- `status` reports `n_observations`, `has_pending` and `best_y`.
- **The objective must belong to the family the AF was trained on, with values
  on that family's scale.** The AF reads the GP's raw posterior mean and
  standard deviation, and the GP uses the family's fixed hyperparameters, so
  rescaled or unrelated objectives give the AF inputs it never saw. For your
  own application, train an AF on a family that models it (Section 10.3).

#### NAF client loop

```python
import asyncio
import json
import websockets

async def run_trial(x):
    # Replace with your experiment: one value, maximized, on the training family's scale.
    return 1.0 - 8.0 * ((x[0] - 0.55) ** 2 + (x[1] - 0.15) ** 2)

async def main():
    async with websockets.connect("ws://127.0.0.1:8771") as ws:
        await ws.send(json.dumps({"type": "start", "n_iter": 15}))
        msg = json.loads(await ws.recv())
        while msg["type"] == "suggest":
            y = await run_trial(msg["x"])
            await ws.send(json.dumps({"type": "observe", "x": msg["x"], "y": y}))
            msg = json.loads(await ws.recv())
        if msg["type"] == "error":
            raise RuntimeError(msg["message"])
        print("best value:", msg["best_so_far"][-1])

asyncio.run(main())
```

### Step 4. Run the provided fake clients

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

For the MOBO server (`mobo_server.yaml`, OKA2):

```bash
uv run python server_scripts/run_mobo_fake_client.py --function oka2 --n-init 5 --n-iter 10
```

For the ConBO server, the fake client runs a sequence of simulated users
(shifted, rescaled Branin variants), one session each, and prints each user's
regret as the population grows. User names must be new to the memory, so pick
a fresh `--prefix` for each run:

```bash
uv run python server_scripts/run_conbo_fake_client.py --n-users 4 --n-evaluations 20 --prefix sim
```

For the TAF+ server, the fake client simulates one new sphere user
(CHI 2024 App. B) with similarity range `--range` and prints each
objective vector. It saves the server's final message plus the user's
noise-free regret to `test_results/taf_plus/taf_plus_server_smoke.json`:

```bash
uv run python server_scripts/run_taf_plus_fake_client.py --range 0.1 --user-seed 100
```

For the NAF server, the fake client draws one task from MetaBO's Branin
family (`--task-seed`), prints each point with its regret, and saves the
server's final message plus the task and its simple regret to
`test_results/naf/naf_server_smoke.json`. A full 30-evaluation session takes
about a second:

```bash
uv run python server_scripts/run_naf_fake_client.py --task-seed 100
```

**The fake clients write to `test_results/`, which holds committed files.** Two of the
commands in this section overwrite them: `run_naf_fake_client.py` (default output
`test_results/naf/naf_server_smoke.json`) and the TAF example above (`--save-json
test_results/trajectories/fake_client_taf_done.json` and its `_x_locations.png`). Pass
`--output <path>` (or another `--save-json`) to write elsewhere, or restore the files with
`git checkout -- test_results`.

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

The command prints `n_train` and `n_test` (expect 12 and 3 with `--train-ratio 0.8`). The JSON contains
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

### Step 4. Run one WebSocket client per **training** variant (12 sessions)

For each training task you must:

- Open a **new** WebSocket connection (one session per variant).
- Send a `start` message that includes a **unique** `task_name` (e.g. `train_task_000`, …,
  `train_task_011`) so saved `gp_states/<name>.json` and `trajectories/<name>.json` do
  not overwrite each other. The server also uses `n_init`, `n_iter`, and `seed` from
  `start` (or defaults from the YAML).
- On each `suggest`, compute `y` with **that task’s** objective, not the base Branin
  registry function.

From the repo root, with `run_bo_server` already running (adjust `--uri` if you did not
use port `8767`):

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

After all twelve sessions, you should see twelve files under
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
`trajectories/`). `configs/server_optimizers/bo_taf_server_test_end_to_end.yaml`
already contains:

```yaml
taf_run_dir: meta-bo-training/taf-gps/branin_server_end_to_end
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
prefix when building specs (names `test_task_000` … `test_task_002`).

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
  --train-trajectories-dir meta-bo-training/taf-gps/branin_server_end_to_end/trajectories \
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

## 7. Multi-objective BO

Multi-objective BO with correlated objectives, following Shah & Ghahramani (ICML 2016); the website's Methods page has the overview. All objectives are **maximized**, inputs live in `[0, 1]^d`, and observations have shape `(n, L)`. The design decisions and the paper reproduction notes are in [`docs/plans/mobo.md`](docs/plans/mobo.md).

### 7.1 The pieces

- **Model** (`models/multi_output_gp.py`): `MultiOutputGP(mode=...)` models the
  objectives jointly and returns, at each input, a mean vector and an `L × L`
  covariance across objectives.
  - `independent`: one GP per objective.
  - `mt`: a multi-task GP with one kernel shared by all objectives.
  - `slf` (default): a semiparametric latent factor model with one kernel per
    latent function. It was the best model in the paper.
- **Hyperparameters**: MAP by default. `hyperparameters="slice"` follows the
  paper: it draws 10 hyperparameter samples per iteration by slice sampling
  and averages the acquisition over them.
- **Acquisition** (`acquisition/mobo.py`): the expected increase in Pareto volume.
  - **CEIPV** uses the predicted correlations between objectives.
  - **IEIPV** treats the objectives as independent and is exact in that case.
  - `eipv_mc` is a Monte Carlo ground truth for tests.
- **Baselines**: BoTorch **qLogNEHVI** with one independent GP per objective
  (`optimizers/mobo_botorch.py`), and random search.
- **Problems** (`test_functions/multi_objective.py`):

  | Problem | d | Objectives | Notes |
  | --- | --- | --- | --- |
  | `oka2` | 3 | 2 | analytic true hypervolume |
  | `vlmop3` | 2 | 3 | no closed-form front |
  | `dtlz1a` | 6 | 2 | analytic true hypervolume |
  | `slf_gp` | 3 | 2 | correlated GP draws with correlation `rho` |

  Each problem has a loose `v_ref` (the worst value over the domain) and a
  finite upper corner `w_ref`.

Three implementation notes, each verified by tests (see the plan for details):

- **The paper's printed λ is a typo.** The moment-matched mean of the CEIPV
  ramp is `(2w + v)/3`, not the printed `½(2w + v)`.
- **The paper's EIPV formula undercounts.** Read literally, eq. 3 credits
  only part of the hypervolume gain. CEIPV here uses an exact cell-local
  decomposition of the gain. The literal form remains available as
  `eipv_form="paper"`.
- **Predictions above `w_ref`.** Thin overflow layers above `w_ref` count
  predictive mass there, with the gain capped at `w_ref`.

### 7.2 Run in Python

The website's Methods page has a `run_mobo` example on OKA2.

`MOBOSequentialOptimizer` exposes the same `bootstrap()` / `suggest()` /
`observe(x, y)` / `result()` ask/tell interface as the single-objective
optimizers.

### 7.3 Benchmark and plot

```bash
# One method, one problem, several seeds (paper protocol: 5 initial points + 100 iterations)
uv run python scripts/run_mobo_benchmark.py --method mobo_ceipv_slf --problem oka2 \
  --n-iter 100 --seeds 0 1 2 3 4 --test-id oka2_demo

# Correlated synthetic problem: objectives drawn from an SLF GP with correlation rho
uv run python scripts/run_mobo_benchmark.py --method mobo_botorch --problem slf_gp --rho -0.5 \
  --n-iter 100 --seeds 0 1 2 --test-id slf_demo

# Hypervolume and log-hypervolume-regret curves, plus final Pareto fronts
uv run python scripts/plot_mobo_results.py --problem oka2 --test-id oka2_demo
```

Methods:

| Method | What it runs |
| --- | --- |
| `mobo_ceipv_slf` | CEIPV with the SLF model (MAP) |
| `mobo_ceipv_slf_slice` | CEIPV with the SLF model, slice-sampled hyperparameters (the paper's setup) |
| `mobo_ceipv_mt` | CEIPV with the multi-task model |
| `mobo_ieipv` | IEIPV with independent GPs (correlation ablation) |
| `mobo_botorch` | BoTorch qLogNEHVI (baseline) |
| `random` | uniform random search |

Results go to `test_results/mobo/trajectories/<test_id>_<method>_<problem>.json`,
where SLF-GP problems are labelled like `slf_gp_rho-0.50_p0`. Plots go to
`test_results/mobo/plots/`. `plot_mobo_results.py --test-id a b c --output-id all`
merges seeds that were saved under several test ids.

Cost: with three objectives (`vlmop3`), the 3-objective SLF GP fit dominates the run
time. `--n-rescore N` scores
the candidate pool with IEIPV and rescores only the top `N` with CEIPV, which
helps when the Pareto front is large.

### 7.4 Run through the server

```bash
# Terminal 1 (port 8768; the default config is set up for OKA2)
uv run python server_scripts/run_mobo_server.py

# Terminal 2
uv run python server_scripts/run_mobo_fake_client.py --function oka2 --n-init 5 --n-iter 10
```

The protocol is described in Section 5 ("MOBO server protocol").

## 8. Continual BO (ConBO)

Continual BO across a sequence of users, following Liao et al. (CHI 2025); the website's Methods page has the overview. The design decisions, deviations from the paper and measured costs are in [`docs/plans/conbo.md`](docs/plans/conbo.md).

### 8.1 The pieces

- **Population model** (`models/bnn.py`): `PopulationBNN`, an MC-dropout
  network (3 × 100 ReLU) with mean and log-variance heads. Its predictive
  variance combines the predicted noise and the spread over dropout samples.
- **Generative replay** (`models/conbo_replay.py`): finished users are stored
  as GPs, not as a growing dataset. After each user, the BNN is retrained on
  the stored GPs' predictions over a fixed input set. Predictions whose
  variance is above a threshold are dropped, so uncertain users don't teach
  the population. By default the threshold is scale-free
  (`replay_variance_fraction=0.05`); set `replay_max_variance` (`λ`, in raw
  output units) for the paper's fixed threshold.
- **Mixed acquisition** (`acquisition/conbo.py`):
  `a(x) = w·EI_pop(x) + (1 − w)·EI_user(x)`, with `w` held at 1 for the first
  `α₁` evaluations of a user and then falling by `α₂` per evaluation. The
  defaults (`α₁ = 5`, `α₂ = 0.2`) hand over fully by the 10th evaluation.
- **Random starts**: user `u` gets `r_u = max(0, r₀ − (u − 1)·d_r)` random
  points first. Early users explore while the population model is still weak.
- **Optimizers** (`optimizers/conbo.py`):
  - `ConBOPopulation` holds the BNN and the memory of finished users, and can
    be saved to and loaded from a directory.
  - `ConBOSequentialOptimizer` runs one user, ask/tell.
  - `run_conbo_sequence` loops over users.

### 8.2 Run in Python

The website's Methods page has a `run_conbo_sequence` example with the paper's Test 1 settings (15 simulated users).

One user at a time, for example inside a study session. The population
persists between sessions:

```python
import numpy as np

from openbo.optimizers.conbo import ConBOPopulation

def run_trial(x):
    """Stand-in for your experiment: maximized, shape (1,). Here a Branin-like peak."""
    return np.array([-((x[0, 0] - 0.25) ** 2 + (x[0, 1] - 0.75) ** 2)])

population = ConBOPopulation.load("test_results/conbo/memory_demo")
opt = population.new_user("participant_16")
for _ in range(30):
    x = opt.suggest()          # shape (1, d)
    y = run_trial(x)           # your experiment; returns shape (1,)
    opt.observe(x, y)
report = population.finish_user(opt)   # store the user and retrain by replay
population.save()
print(report.n_inputs_kept, report.kept_fraction)
```

Notes:
- Each user adapts a **private copy** of the population model. If a session is
  abandoned (never passed to `finish_user`), the population is unchanged.
- Users must be finished **in the order they were started**.
- `finish_user` takes about 10 s on a CPU (1,200 replay epochs), between
  users rather than during a session.
- The memory directory uses the TAF layout (`gp_states/`, `trajectories/`),
  so it can also be passed to TAF as `taf_run_dir`.
- Switches for the paper's ablations and open questions:
  - `use_user_gp=False` (ConBO without the user GP)
  - `replay_filter=False` (no variance filter)
  - `retrain="continue"` (keep training the BNN instead of resetting it)
  - `weight_clock="bo_steps"` (start the `α₁/α₂` clock after the random starts)
  - `refine_suggestion=False` (pure grid search, as in the paper)
- `λ` depends on the problem's output scale; the paper tuned it per problem
  (Branin 50, McCormick 10). For a new problem, set `replay_variance_fraction`
  instead and check `report.kept_fraction`.

### 8.3 Benchmark and plot

```bash
# One method over a sequence of 15 simulated users, several seeds
uv run python scripts/run_conbo_benchmark.py --method conbo --seeds 0 1 2 --test-id demo
uv run python scripts/run_conbo_benchmark.py --method bo --seeds 0 1 2 --test-id demo

# Regret curves over users 4-15, per-user panels and a summary table
uv run python scripts/plot_conbo_results.py --test-id demo --from-user 4
```

Methods (`benchmarks/conbo_runner.py`); all see the same users per seed:

| Method | What it runs |
| --- | --- |
| `conbo` | ConBO |
| `bo` | standard BO, each user from scratch with `r₀` random starts |
| `taf_m` / `taf_r` | OpenBO's TAF with all previous users as sources |
| `single_gp` | one GP on all previous users' data plus the current user's |

Only `--test branin` (the paper's Test 1) is defined so far. Results go to
`test_results/conbo/trajectories/<test_id>_<method>_<test>_s<seed>.json`. The
archived 5-seed comparison is described in Section 12.

### 8.4 Run through the server

The ConBO server runs one participant per WebSocket session and keeps the
population on disk between sessions:

```bash
# Terminal 1 (port 8769)
uv run python server_scripts/run_conbo_server.py

# Terminal 2: four simulated users in a row
uv run python server_scripts/run_conbo_fake_client.py --n-users 4 --n-evaluations 20 --prefix sim
```

The protocol is described in Section 5 ("ConBO server protocol").

## 9. TAF+: transfer across users with several objectives

TAF+ from Liao et al. (CHI 2024), built on the original TAF of Wistuba, Schilling and Schmidt-Thieme (Machine Learning, 2018); the website's Methods page has the overview. **Wherever the CHI 2024 paper and the original TAF paper disagree, OpenBO follows the original.** In particular:

- the source term is the improvement of each earlier user's posterior mean
  over the best value it predicts among the new user's evaluated designs,
  clipped at 0;
- the model weights are the original's TAF-PoE (inverse predictive
  variance), which the CHI paper calls TAF-M;
- the default kernel is SE-ARD (RBF), with Matérn-5/2 as an option.

This is different from OpenBO's older TAF (Section 4.1). The decisions,
deviations and open questions are in
[`docs/plans/taf_plus.md`](docs/plans/taf_plus.md).

### 9.1 The pieces

- **Acquisition** (`acquisition/taf_plus.py`): each objective `l` gets the
  original TAF terms. These are combined across objectives with the weights
  `α` (CHI eq. 4), which also combine the model weights (eq. 5). The
  population weights are then multiplied by a decay `δ(t)` (eq. 6): `δ = 1`
  for the first `d1` evaluations, then it falls by `d2` per evaluation until
  it reaches 0. From then on TAF+ is plain EI on the new user's GPs.
- **Population** (`optimizers/bo_taf_plus.py`): `TAFPlusPopulation` fits
  one GP per objective for each earlier user (`add_user(name, x, y)`, with
  `y` of shape `(n, L)`). `save`/`load` use a directory with `gp_states/`,
  `trajectories/` and `taf_plus_population.json`. `subset(exclude=name)`
  supports leave-one-out runs.
- **Optimizer**: `TAFPlusSequentialOptimizer` (ask/tell, `observe(x, y)`
  with `y` of shape `(1, L)`) and `run_bo_taf_plus`.
  - The first design maximizes the population's product-of-experts mean
    (`first_point="population"`, the default), or is the domain centre
    (`first_point="centre"`, also the fallback for an empty population).
  - After that, each step takes the TAF+ argmax over a fresh Sobol pool
    (`n_candidates`, default 1024) and polishes it with L-BFGS-B.
- **Workflow** (`optimizers/taf_plus_workflow.py`), the CHI paper's steps
  2 and 3:
  - `optimize_objective_weights` is Algorithm 1. It picks `α` from each
    population user's ratings of their own Pareto-optimal designs.
  - `search_decay` is the leave-one-out grid search for `(d1, d2)` over the
    paper's grid (`d1` in 1–9, `d2` in 0.1–0.3). Each population user in turn
    plays the new user. Pass `n_workers` to use several processes.

`TAFPlusConfig` switches (defaults first):

- `decay=True`: `False` keeps the population weights at full strength.
  That is the paper's "TAF (multi-objective)" baseline, and with one
  objective it is the original TAF.
- `d1=2`, `d2=0.3`: the CHI study's decay.
- `weight_mode="poe"`: `"taf_r"` uses the original's ranking-based TAF-R
  weights, with bandwidth `rho`.
- `kernel_type="rbf"`: `"matern52"` is the CHI paper's kernel.
- `resample_candidates=True`, `refine_suggestion=True`: `False` gives one
  fixed pool and the paper's pure pool search.

### 9.2 Run in Python

```python
import numpy as np

from openbo.optimizers.bo_taf_plus import TAFPlusConfig, TAFPlusPopulation, run_bo_taf_plus
from openbo.optimizers.taf_plus_workflow import search_decay
from openbo.test_functions.taf_plus_users import make_sphere_users

# 4 earlier users and 1 new one: 4D input, 3 maximized objectives
users = make_sphere_users(5, shift_range=0.1, scale_range=0.1, seed=0)
earlier, new_user = users[:4], users[4]
rng = np.random.default_rng(0)

# Step 1: one GP per objective for each earlier user, from their data
population = TAFPlusPopulation(input_dim=4, n_objectives=3)
for i, user in enumerate(earlier):
    x = rng.random((30, 4))
    population.add_user(f"user_{i:02d}", x, user.observe(x, rng))
population.save("test_results/taf_plus/demo_population")

# Step 3 (optional): leave-one-out search for the decay (d1, d2)
config = TAFPlusConfig(input_dim=4, objective_weights=(0.3, 0.5, 0.2), n_candidates=256)
decay = search_decay(population, config, d1_values=(2, 4), d2_values=(0.1, 0.3), n_iter=5)
d1, d2 = decay.best

# Step 4: adapt to the new user
config = TAFPlusConfig(input_dim=4, objective_weights=(0.3, 0.5, 0.2), d1=d1, d2=d2)
result = run_bo_taf_plus(lambda x: new_user.observe(x, rng), config, population, n_iter=10)
print(result.best_so_far)    # best weighted sum y @ alpha after each evaluation
print(result.decay_history)  # delta(t) per step
```

Most of the time goes to the decay search. For
real users, step 1 is usually a multi-objective BO session per person
(Section 7). Step 2 (`optimize_objective_weights`) needs each person's
ratings of their Pareto-optimal designs. Without ratings, set `α` by hand,
as the paper's simulations do.

### 9.3 Benchmark and plot

The runner (`benchmarks/taf_plus_runner.py`) reproduces the paper's
App. B simulations. The base function has three 2D spheres on a 4D
input, giving 3 objectives. Each simulated user shifts the inputs and scales
the outputs within a similarity range `r`. Each setting has 10 population
users, modelled with qLogNEHVI, and 10 new users. Settings (`--sim`):
`sim1` (`r = 0.01`, used with six `--alpha` vectors), and `sim2_r005`,
`sim2_r01`, `sim2_r02`, `sim2_r03` (`r` = 0.05–0.3, `α = (0.3, 0.5, 0.2)`).

```bash
# Population models (built on first use and cached), then the decay search
uv run python scripts/run_taf_plus_benchmark.py --method decay_search --sim sim2_r01 --workers 7 --test-id demo

# Each method on the new users (taf_plus reads the decay from the search above)
uv run python scripts/run_taf_plus_benchmark.py --method taf_plus --sim sim2_r01 --test-id demo
uv run python scripts/run_taf_plus_benchmark.py --method taf --sim sim2_r01 --test-id demo
uv run python scripts/run_taf_plus_benchmark.py --method bo --sim sim2_r01 --test-id demo
uv run python scripts/run_taf_plus_benchmark.py --method random --sim sim2_r01 --test-id demo

# Regret curves (each design, or --metric best), decay heatmaps and a table
uv run python scripts/plot_taf_plus_results.py --test-id demo
```

| Method | What it runs |
| --- | --- |
| `taf_plus` | TAF+ with the searched decay (or `--d1/--d2`) |
| `taf` | the same without decay (the paper's multi-objective TAF) |
| `bo` | scratch EI on the noisy weighted sum, 5 random starts |
| `random` | uniform random designs |

Building a population is the slowest step; it is cached on disk and reused. Results go to `test_results/taf_plus/` (Section 11). The archived
run and its findings are in Section 12: transfer gives an excellent first
design, but TAF+ rarely improves after it.

### 9.4 Run through the server

The TAF+ server runs one new user per WebSocket session, with a fixed
population loaded from `population_dir`:

```bash
# Terminal 1 (port 8770; the default config uses the archived range-0.1 population)
uv run python server_scripts/run_taf_plus_server.py

# Terminal 2: one simulated new user, 10 evaluations
uv run python server_scripts/run_taf_plus_fake_client.py --range 0.1 --user-seed 100
```

The protocol is described in Section 5 ("TAF+ server protocol").

## 10. NAF (MetaBO): a learned acquisition function

MetaBO from Volpp et al. (ICLR 2020): a neural acquisition function (NAF) trained with PPO on a family of related tasks; the website's Methods page has the overview.

**Where the paper and the authors' reference code
([boschresearch/MetaBO](https://github.com/boschresearch/MetaBO)) disagree,
OpenBO follows the code.** The code is AGPL-3.0, so OpenBO reimplements its
behaviour and copies none of it. The design record, with the paper and code
notes, decisions, measurements and open questions, is
[`docs/plans/naf.md`](docs/plans/naf.md). Read it before changing NAF code.

### 10.1 The pieces

- **Task families** (`test_functions/metabo_families.py`): MetaBO's three
  global benchmarks. `normalized_branin` is `-Branin` rescaled as in the
  reference code (`(y + 54.44) / 51.44`, maximum about 1.05);
  `normalized_goldstein_price` (2D, `-(log f − 8.693) / 2.427`) and
  `normalized_hartmann3` (3D, `-(f + 0.93) / 0.95`) use the reference's
  normalizations too, so all three have mean about 0 and standard deviation
  about 1. A task (`BraninTask`, `GoldsteinPriceTask`, `Hartmann3Task`) is
  `s · g(x − t)`, with translation `t` uniform in `[−0.1, 0.1]^D` and scaling
  `s` uniform in `[0.9, 1.1]`; `task.optimum` is tracked exactly.
  `sample_branin_tasks(n, seed)` (and `sample_goldstein_price_tasks`,
  `sample_hartmann3_tasks`) draw random tasks, and `branin_grid_tasks(50)` (and
  `goldstein_price_grid_tasks`, `hartmann3_grid_tasks`) give the 50 fixed
  training tasks of "MetaBO-50" (a Sobol grid over `(t, s)`).
  `NAF_FAMILY_SPECS` in `optimizers/naf.py` ties each family to its dimension,
  candidate set, GP hyperparameters and tasks, and `get_naf_family(name).config()`
  builds a matching `NAFConfig`.
- **GP** (`optimizers/naf.py`): every method uses a zero-mean RBF GP with
  **fixed** hyperparameters per family and almost no noise, as in the
  reference. For Branin these are the reference's values,
  `BRANIN_GP_HYPERPARAMETERS` (lengthscales 0.235 and 0.578, variance 2.0).
  `GOLDSTEIN_PRICE_GP_HYPERPARAMETERS` (lengthscales 0.109 and 0.087, variance
  0.652) and `HARTMANN3_GP_HYPERPARAMETERS` (lengthscales 0.709, 0.305 and 0.173,
  variance 0.694) are OpenBO's own type-2 ML fits, the median over 10 tasks. The
  reference's training scripts give the authors' values for these too
  (Goldstein-Price 0.130 and 0.07, variance 0.616, noise 1e-6; Hartmann-3 0.716,
  0.298 and 0.186, variance 0.83, noise 1.688e-11); they were noticed after the
  runs in Section 12 were made, and the fits are close to them.
  `FamilyGPHyperparameters.new_gp()` builds one; `fit_family_hyperparameters`
  fits values for a new family.
- **Neural AF** (`acquisition/naf.py`): `NeuralAF` is an MLP (4 hidden layers
  of 200 ReLU units) that gives one score (logit) per candidate point from
  that point's features `[μ(x), σ(x), x, t, T]` (`naf_features`): the GP's
  posterior mean and standard deviation, the location, the step `t` and the
  budget `T`. The features are raw, not standardized, as in the code.
- **Candidates** (`AdaptiveSet`): each step the AF scores a 31 × 31 grid,
  then a cloud of 1000 local points around each of its 5 best grid points,
  and chooses among the grid plus the 5 best local points (966 candidates).
  During training it samples from the softmax of the logits; at test time it
  takes the argmax.
- **Training** (`train_naf`, `NAFConfig`): one BO run on a task is one RL
  episode (`BOEpisode`) of `T = 30` steps. The reward at step `t` is
  `−log10(simple regret)`, so the policy is trained to minimize the summed
  log regret over the whole run. PPO with GAE, settings from the paper's
  Table 4: 1200 steps (40 episodes) per iteration, 4 epochs, minibatches of
  60, Adam with learning rate 1e-4, clip 0.15, γ = λ = 0.98, entropy
  coefficient 0.01, and a separate value network on `[t, T]`. Rollouts run in
  worker processes. The checkpoint kept for testing (`best_af.pt`) is the
  iteration with the best mean training return, as in the reference.
- **Optimizer**: `NAFSequentialOptimizer` (ask/tell; `from_directory(dir)`
  loads `best_af.pt` with the GP hyperparameters and budget it was trained
  with) and `run_naf(objective, optimizer)`. The budget `T` is one of the
  AF's inputs, so the optimizer refuses to run past the training budget.

### 10.2 Run a trained AF in Python

```python
import numpy as np

from openbo.optimizers.naf import NAFSequentialOptimizer, run_naf
from openbo.test_functions.metabo_families import sample_branin_tasks

# A new Branin-family task: translated by t in [-0.1, 0.1]^2, scaled by s in [0.9, 1.1]
task = sample_branin_tasks(1, seed=123)[0]

# Load the archived neural AF (best_af.pt) with its GP hyperparameters and budget (T = 30)
optimizer = NAFSequentialOptimizer.from_directory("benchmark_results/naf_branin/training/branin_metabo50_v1")

# Ask/tell by hand for the first evaluations ...
for _ in range(5):
    x = optimizer.suggest()  # shape (1, 2), in [0, 1]^2
    optimizer.observe(x, task(x))

# ... or let run_naf use the rest of the budget
x_obs, y_obs = run_naf(task, optimizer, n_iter=optimizer.budget - optimizer.t)
regret = task.optimum - np.maximum.accumulate(y_obs)
print(np.log10(np.maximum(regret[[0, 4, 9, 29]], 1e-20)))  # log10 regret after 1, 5, 10, 30 evaluations
```

This takes a few seconds. The objective takes `(n, 2)` points in `[0, 1]²`
and returns `(n,)` values to maximize, like every OpenBO objective. To use
your own objective, it has to belong to the family the AF was trained on.

### 10.3 Train a neural AF

```bash
# MetaBO-50: train on the 50 fixed Branin tasks (7 rollout workers, 8 threads for the update)
uv run python scripts/train_naf.py --tasks metabo50 --n-iterations 300 --run-id my_run

# Continue a run to more iterations (only --n-iterations may change on resume)
uv run python scripts/train_naf.py --tasks metabo50 --n-iterations 600 --run-id my_run --resume

# Another family (the default run id is <family>_metabo50); Hartmann-3 is 3D and
# takes about twice as long per iteration as Branin
uv run python scripts/train_naf.py --family goldstein_price --n-iterations 2000
uv run python scripts/train_naf.py --family hartmann3 --n-iterations 2000
```

`--tasks full` draws a fresh random task for every episode ("full MetaBO")
instead of the 50 fixed ones. A run writes `meta-bo-training/naf/<run-id>/`
after every iteration: `af.pt` (latest), `best_af.pt` (best mean training
return), `naf_config.json`, `training_history.json` (per-iteration return,
final log regret, entropy, losses and time) and `training_state.pt`
(everything needed to resume). The log prints one line per iteration; mean
final log10 regret falling and entropy falling mean the policy is learning.
Stopping the process between iterations is safe, since the checkpoint is
written before each log line.

**Cost** (no GPU): the time per iteration is dominated by the PPO update. The archived run
slowed down as training went on, because a trained policy creates denormal floats in
the PPO update; the update now flushes them (Section 10.6).

### 10.4 Benchmark and plot

The benchmark (`benchmarks/naf_runner.py`) follows the paper's Fig. 2: each
method runs on the same 100 unseen test tasks (`--test-seed 100`), 30
evaluations each. Every method uses the family's fixed-hyperparameter GP and
the same candidate set, so only the acquisition function differs.

```bash
AF=benchmark_results/naf_branin/training/branin_metabo50_v1   # or meta-bo-training/naf/<run-id>
for m in naf ei random taf_me_50 taf_r_50; do
  uv run python scripts/run_naf_benchmark.py --method $m --af-dir $AF --workers 8 --test-id demo
done
uv run python scripts/plot_naf_results.py --test-id demo --training-dir $AF
```

This block is for the Branin family, the default. **For Goldstein-Price or
Hartmann-3, pass `--family goldstein_price` or `--family hartmann3` to both
scripts**, with `--af-dir` pointing at a model of that family (for example
`benchmark_results/naf_goldstein_price/training/goldstein_price_metabo50_v1` or
`benchmark_results/naf_hartmann3/training/hartmann3_metabo50_v2_2000`); the
family sets the test tasks, the GP and the baselines' candidate set. If you
forget it, `run_naf_benchmark.py` checks the AF against the family: an AF of
another dimension (a Hartmann-3 AF with the default Branin) stops with an error that
says so, and an AF of the same dimension but another family (a Goldstein-Price AF
with Branin, both 2D) gets a warning that its GP hyperparameters belong to another
family, but still runs on the wrong tasks and saves `..._branin.json`, so heed the
warning. (Before this check existed, the Goldstein-Price AF ran silently on Branin
tasks and the Hartmann-3 one stopped with a shape error.)
```bash
uv run python scripts/run_naf_benchmark.py --method naf --family hartmann3 \
  --af-dir benchmark_results/naf_hartmann3/training/hartmann3_metabo50_v2_2000 --workers 8 --test-id demo
uv run python scripts/plot_naf_results.py --family hartmann3 --test-id demo \
  --training-dir benchmark_results/naf_hartmann3/training/hartmann3_metabo50_v2_2000
```
(The plot also needs the other methods' files for the same `--test-id` and
`--family`: run `ei`, `random`, `taf_me_50` and `taf_r_50` the same way.)

| Method | What it runs |
| --- | --- |
| `naf` | the trained AF (`best_af.pt` in `--af-dir`), greedy |
| `ei` | EI, starting from one random (scrambled Sobol) point. The paper (App. B.1) and the reference code start EI from the domain midpoint `[0.5, 0.5]`; OpenBO deviates. Starting from the midpoint changes EI's curve up to about evaluation 20 (median log10 regret at 15: −2.76 → −2.11) but not its value at 30 (−4.98 → −5.07) |
| `taf_me_50` | the original TAF (Wistuba et al.) with the 50 training tasks as sources (100 Sobol points each), product-of-experts (TAF-ME) weights |
| `taf_r_50` | the same with ranking-based TAF-R weights |
| `random` | uniform random points |

TAF is `taf_plus_acquisition` (Section 9) with one objective and no decay,
which is the reference's TAF-ME and TAF-R. One deviation: before any data,
the reference's TAF uses the task's own value at a corner as its incumbent,
which a real user could not provide; OpenBO's first TAF point maximizes the
sources' product-of-experts mean instead, as in TAF+.

Results go to `test_results/naf/trajectories/<test-id>_<method>_<family>.json`
(points, values and simple regret per step for every task) and
`test_results/naf/plots/` (median log10 regret with 30–70% bands, a table,
and with `--training-dir` the training curve). `naf`, `ei` and `random` take
under a minute for 100 tasks; `taf_me_50` and `taf_r_50` take about 40
minutes each on 8 workers, because every step scores 5000 local points
against 50 source GPs. Worker processes use one BLAS thread each (more
oversubscribes the CPU and makes TAF several times slower).

### 10.5 Run through the server

The NAF server runs one task per WebSocket session, with a trained AF loaded
from `af_dir` (by default the archived Branin model):

```bash
# Terminal 1 (port 8771)
uv run python server_scripts/run_naf_server.py

# Terminal 2: one simulated Branin-family task, 30 evaluations (about a second)
uv run python server_scripts/run_naf_fake_client.py --task-seed 100
```

A session can stop early (`n_iter` in the `start` message) but never run
past the AF's training budget. The objective must come from the family the
AF was trained on, on the same value scale. The protocol and a client loop
are in Section 5 ("NAF server protocol").

The server does not depend on the family: it only checks `input_dim` and
`y_range` in its config. Two more configs are shipped, each for its archived AF:
`naf_server_goldstein_price.yaml` (2D, `y_range` [-3.5, 4.5], the 600-iteration
model) and `naf_server_hartmann3.yaml` (3D, `y_range` [-1.5, 4.5], the
2000-iteration model). Start one with `--config-path` and pass `--family
goldstein_price` or `--family hartmann3` to the fake client:

```bash
uv run python server_scripts/run_naf_server.py --config-path configs/server_optimizers/naf_server_hartmann3.yaml
uv run python server_scripts/run_naf_fake_client.py --family hartmann3
```

A full session with each shipped model ran through the server (one task, so only
a smoke test: log10 regret about -3.4 for Goldstein-Price and -3.1 for
Hartmann-3). A 2D client is refused by a 3D server with a clear message. The
`y_range` values cover the observed values of 3050 tasks of each family with a
margin. For an AF you train yourself, set `input_dim` and a `y_range` that
covers the family's values.

### 10.6 Open problems

Results are in Section 12. Open questions 10–12 in [`docs/plans/naf.md`](docs/plans/naf.md):

- **The first point is a probe.** It is the same corner on every task, and its
  value tells the policy how the task is shifted and scaled. MetaBO's reward
  sums the log regret over all `T` steps, so the policy gives up a good first
  guess for better later steps, and TAF starts better. For human-in-the-loop
  use, note that this reward weights every step equally.
- **Repeated points.** A trained AF often re-evaluates points it has already
  seen once it has converged. The reference code allows repeats (no masking),
  so OpenBO keeps them. The training slowdown they seemed to cause was
  denormal floats in the PPO update, now fixed (about 63 s per iteration
  instead of 140 s).
- **Training length.** Longer training lowered the Branin plateau a lot
  (Section 12) and fixed Hartmann-3's stall at 600 iterations, but
  Goldstein-Price did not improve beyond 600 iterations (it was continued to
  2000). Each long run is **one training seed**, picked by best training
  return; repeat with other seeds before relying on the numbers. The Branin and
  Hartmann-3 training curves are unstable late in training. Resume a run with
  `--resume --n-iterations N`.

## 11. Results folder convention

By default, artifacts are organized under `test_results/`:

- `test_results/trajectories/`
  - single-function run JSONs:
    - `{test_id}_{method}_{function}.json`
  - family run folders:
    - `{test_id}_{method}_{base_function}_{subset}/`
    - containing one task file per task plus `summary.json`
- `test_results/plots/`
  - single-function and family plot PNGs
- `test_results/mobo/`
  - `trajectories/{test_id}_{method}_{problem}.json` (all seeds of one method)
    and `plots/` from the MOBO benchmark scripts, plus MOBO server smoke-test JSON
- `test_results/conbo/`
  - `trajectories/{test_id}_{method}_{test}_s{seed}.json` (one user sequence per file)
    and `plots/` from the ConBO benchmark scripts, plus ConBO server smoke-test JSON
    (`<prefix>_server_smoke.json`)
- `test_results/taf_plus/`
  - `populations/{test_id}_{sim}_s{seed}/` (cached population models),
    `decay/{test_id}_{sim}_{alpha}_s{seed}.json` (decay searches),
    `trajectories/{test_id}_{method}_{sim}_{alpha}_s{seed}.json` and `plots/`,
    plus TAF+ server smoke-test JSON (`taf_plus_server_smoke.json`)
- `test_results/naf/`
  - `trajectories/{test_id}_{method}_{family}.json` (all test tasks of one method)
    and `plots/` (`{test_id}_{family}.png`, `{test_id}_training.png`), plus NAF
    server smoke-test JSON (`naf_server_smoke.json`)
- `meta-bo-training/naf/{run_id}/` - NAF training runs (`best_af.pt`, `af.pt`,
  `naf_config.json`, `training_history.json`, `training_state.pt`)

Practical workflow convention:
- Keep using `test_results/` as the default for day-to-day experiments and quick iterations.
- Use `--results-dir benchmark_results` for large, milestone-style benchmark runs that you want to keep stable over time.
- Prefer unique `--test-id` values for archived runs in `benchmark_results/` to avoid accidental overwrite.

## 12. Archived benchmark results

The headline tables and plots are on the website's [Benchmarks](https://www.csie.ntu.edu.tw/~yichi/openbo-website/benchmarks.html) page. Each archive under `benchmark_results/` has a `run_all.sh` (or the commands) that reproduces it, and most have their own README with the full tables and findings:

| Archive | What it holds | Run your own |
| --- | --- | --- |
| `benchmark_scratch_botorch/` | Scratch BO (`bo_scratch_grid`, `bo_scratch_multistart`) vs BoTorch and random search, with (`benchmark_BO_BoTorch_with_noise/`) and without (`benchmark_BO_BoTorch_no_noise/`) output noise, plus search-behaviour plots (`visual_search_behaviors_with_noise/`) | Sections 2–3 |
| `benchmark_taf/` | `benchmark_taf_bo_{ackley,branin,hartmann6}/`: TAF-M / TAF-R family comparisons, plus 2D search-path plots for TAF and BoTorch | Section 4 |
| `conbo_branin_quick/` | ConBO vs standard BO, Test 1, one seed (README) | Section 8.3 |
| `conbo_branin_5seeds/` | ConBO vs standard BO, TAF-M, TAF-R and a single GP, Test 1, 5 seeds; about 47 minutes on 8 cores (README) | Section 8.3 |
| `mobo_2obj/` | OKA2 and DTLZ1a, 5 methods × 3 seeds × 40 iterations | Section 7.3 |
| `taf_plus_sims/` | CHI 2024 App. B Simulations 1 and 2, one seed, with cached populations and decay searches; about 3 hours on 8 cores (README) | Section 9.3 |
| `naf_branin/` | The trained Branin AF (`training/branin_metabo50_v1/`, 236 iterations) and 5 methods × 100 test tasks (README) | Sections 10.3–10.4 |
| `naf_goldstein_price/` | The same for Goldstein-Price, 600 iterations (README) | Sections 10.3–10.4 |
| `naf_hartmann3/` | The same for Hartmann-3, 600 and 2000 iterations (README) | Sections 10.3–10.4 |

Saved TAF source runs live under `meta-bo-training/taf-gps/` (`branin_train_v1`
from `train_taf.py`, `branin_server_end_to_end` from the Section 6 server
workflow). All 12 tasks of `branin_train_v1` are loaded (earlier, 11 of them
sat in a nested folder the loader did not read; see Section 4.1).

**Multi-objective BO** (`mobo_2obj/`, no README of its own). The budget is
reduced: 5 initial points plus 40 iterations, 3 seeds (the paper uses 100
iterations and 50 repeats). On OKA2, CEIPV-SLF, IEIPV and BoTorch end within
noise of each other (BoTorch rises fastest early); CEIPV-MT stalls, as its
single shared kernel suits OKA2 poorly. On DTLZ1a the NumPy methods reach a
final log10 hypervolume regret of about 1.3–1.7 while BoTorch plateaus near
3.2. With 3 seeds, the differences between CEIPV-SLF and IEIPV are not
significant. The strongly correlated SLF-GP problems, where the paper reports
the clearest gains from modelling correlation, and VLMOP3 have not been run yet.

**NAF on Branin, trained longer.** These runs are not in `naf_branin/`, whose
README covers only the archived 236-iteration model. Median log10 simple
regret on the same 100 test tasks:

| Method | @1 | @5 | @10 | @15 | @20 | @30 |
| --- | --- | --- | --- | --- | --- | --- |
| NAF, archived (236 iterations) | -0.66 | -1.60 | -3.31 | -3.35 | -3.36 | -3.36 |
| NAF, 313 iterations | -0.66 | -1.60 | -3.82 | -3.88 | -3.88 | -3.88 |
| NAF, 2000 iterations | -0.75 | -1.57 | **-4.97** | **-5.62** | **-5.62** | **-5.62** |
| EI | -0.15 | -0.87 | -1.53 | -2.76 | -3.37 | **-4.98** |

- With 313 iterations (`meta-bo-training/naf/branin_metabo50_v1_long`, best at
  307; `test_results/naf/trajectories/v2_long313_naf_branin.json`) the plateau
  is -3.88 instead of -3.36, and the mean over evaluations 1–30 (-2.95) beats
  EI (-2.77) and both TAFs.
- With 2000 iterations, the reference's length
  (`meta-bo-training/naf/branin_metabo50_v2_2000`, best checkpoint at iteration
  1585; `test_results/naf/trajectories/v3_2000_naf_branin.json`), the mean over
  evaluations 1–30 is -4.62. From 10 evaluations on it is below EI at every
  budget, and close to the paper's Figure 2(a) (about -5). It still repeats
  some points (on average 19.6 of the 30 evaluations are distinct, against
  24.4 for the archived model).
- **Read the 2000-iteration row with care.** It comes from **one training run
  (seed 0)** on one set of 100 test tasks, with no confidence interval, and
  from the checkpoint with the best training return (iteration 1585). The
  training curve has an abrupt setback right after that iteration: the mean
  final log10 regret was about -5.2 around iteration 1585, about -3.7 around
  1620 and about -4.4 at the end (`test_results/naf/plots/v3_2000_training.png`).
  The model of the **last** iteration (`af.pt`) reaches a median of -4.88 at 30
  evaluations (-4.49 at 10) and a mean over evaluations 1–30 of -3.51, so it is
  slightly behind EI (-4.98) at 30 evaluations. The table is therefore what the
  best checkpoint of this run reaches, an upper estimate of what a training
  gives, not an average. The regret curves against the baselines (the archived
  runs) are in `test_results/naf/plots/v3_2000_branin.png`.

## 13. Project structure

- `README.md` - project overview, workflows, and command examples.
- `docs/plans/` - design notes and implementation plans (`mobo.md`, `conbo.md`, `taf_plus.md`, `naf.md`).
- `relevant_papers/references.bib` - BibTeX for the papers OpenBO implements directly (PDFs are not redistributed; follow the DOIs).
- `pyproject.toml` - dependencies, build config, and project metadata.
- `configs/` - YAML configs and reusable artifacts.
  - `benchmark.yaml` - benchmark config placeholder (not read by any script).
  - `methods/*.yaml` - method-level config placeholders (not read by any script).
  - `family_splits/*.json` - persisted train/test task-family splits.
  - `server_optimizers/*.yaml` - runtime configs for BO servers (e.g. `input_dim`, `y_range`, TAF defaults).
- `src/openbo/` - main package (`import openbo`; distribution name is `open-bo`).
  - `test_functions/` - synthetic objectives, transforms, family/task variants and registry utilities; multi-objective problems (`multi_objective.py`), ConBO's simulated users (`families.py`), TAF+'s sphere users (`taf_plus_users.py`) and MetaBO's Branin, Goldstein-Price and Hartmann-3 families for NAF (`metabo_families.py`).
  - `models/` - GP implementations and kernels (the NumPy GP in `gp_scratch.py`), including the multi-output GP in `multi_output_gp.py`, a slice sampler in `slice_sampling.py`, and ConBO's MC-dropout population BNN (`bnn.py`) and generative replay (`conbo_replay.py`).
  - `acquisition/` - EI, TAF, EIPV/CEIPV for MOBO (`mobo.py`), ConBO's mixed population/user EI (`conbo.py`), TAF+ with objective weights and decay (`taf_plus.py`; see [`docs/plans/taf_plus.md`](docs/plans/taf_plus.md)), and the neural AF with its features and candidate set (`naf.py`).
  - `optimizers/` - random search (`random_search.py`), standard BO, TAF; MOBO in `mobo.py` and the qLogNEHVI baseline in `mobo_botorch.py`; continual BO across a sequence of users in `conbo.py`; TAF+ (transfer from earlier users with objective weights and decay) in `bo_taf_plus.py`, with its workflow steps (objective weights from ratings, decay search) in `taf_plus_workflow.py`; NAF / MetaBO (fixed family GPs, RL episodes, PPO training, ask/tell optimizer) in `naf.py`.
  - `server_optimizers/` - WebSocket session adapters for generic BO, TAF, MOBO, ConBO, TAF+ and NAF.
  - `benchmarks/` - `run_simple_benchmark` method dispatcher, Pareto/hypervolume utilities (`hypervolume.py`), the MOBO method dispatcher (`mobo_runner.py`), the ConBO user-sequence dispatcher (`conbo_runner.py`), the TAF+ simulation runner (`taf_plus_runner.py`), the MetaBO benchmark runner (`naf_runner.py`), and placeholder metrics/plotting/seeds modules.
- `scripts/` - local benchmark/training/plot entrypoints.
  - `run_benchmark.py` - run one method on one function; optional `--noisy` and 2D x-location plot.
  - `plot_results.py` - single-function multi-method log-regret plots (rerun or stored trajectories).
  - `create_family_split.py` - create and save train/test splits for task families.
  - `run_family_benchmark.py` - run one method across a task family and save per-task trajectories.
  - `plot_family_results.py` - family mean/std and best-so-far log-regret plots.
  - `plot_family_from_benchmark_runs.py` - merge any number of saved `run_family_benchmark.py` run folders into one comparison plot (custom legend labels per `--run`).
  - `train_taf.py` - produce TAF source artifacts (`trajectories/`, `gp_states/`) from training tasks.
  - `plot_taf_gp_predictions.py` - 2D GP mean/std heatmaps from saved TAF run artifacts.
  - `plot_taf_acquisition_heatmap.py` - visualize stored TAF acquisition query values by iteration.
  - `run_mobo_benchmark.py` - run one MOBO method on one multi-objective problem over several seeds.
  - `plot_mobo_results.py` - hypervolume and log-hypervolume-regret curves and Pareto fronts; merges seeds across test ids.
  - `run_conbo_benchmark.py` - run one ConBO-benchmark method over a sequence of simulated users, for several seeds.
  - `plot_conbo_results.py` - per-user log-regret curves and a summary table for the ConBO benchmark.
  - `run_taf_plus_benchmark.py` - one TAF+ simulation step: population modeling, decay search, or one method on the new users.
  - `train_naf.py` - train a MetaBO neural AF with PPO (Branin; checkpoints and resume).
  - `run_naf_benchmark.py` - run one method of the MetaBO benchmark (NAF, EI, TAF-ME-50, TAF-R-50, random) on unseen test tasks.
  - `plot_naf_results.py` - median log10 regret curves with 30–70% bands, and a NAF training curve.
  - `plot_taf_plus_results.py` - TAF+ regret curves (each design or best so far), decay-search heatmaps and a summary table.
  - `run_botorch_fake_client.py` - toy client loop for BoTorch-oriented external-objective testing.
  - `aggregate_results.py` - placeholder aggregation script.
- `server_scripts/` - server and WebSocket client helpers for ask/tell workflows.
  - `run_bo_server.py` - generic BO server (`bo_botorch` or `bo_scratch`) via config/start message.
  - `run_taf_server.py` - dedicated TAF server.
  - `run_mobo_server.py` - dedicated multi-objective BO server.
  - `run_mobo_fake_client.py` - multi-objective client (`--url`, `--function`, default OKA2) for MOBO server smoke tests.
  - `run_conbo_server.py` - continual BO server (one session per user, population kept in `memory_dir`).
  - `run_conbo_fake_client.py` - simulated user sequence for ConBO server smoke tests (`--url`, `--n-users`, `--n-evaluations`, `--prefix`).
  - `run_taf_plus_server.py` - TAF+ server (one session per new user, fixed population from `population_dir`).
  - `run_taf_plus_fake_client.py` - one simulated sphere user for TAF+ server smoke tests (`--url`, `--range`, `--user-seed`, `--alpha`, `--n-iter`).
  - `run_naf_server.py` - NAF (MetaBO) server (one session per task, trained AF from `af_dir`).
  - `run_naf_fake_client.py` - one simulated task of a MetaBO family for NAF server smoke tests (`--url`, `--family`, `--task-seed`, `--n-iter`).
  - `run_fake_client.py` - fake Branin client; optional 2D x-location plotting.
  - `run_manual_family_train_clients.py` - one session per train variant against scratch server.
  - `run_manual_family_test_clients.py` - one session per test variant against TAF server; optional saved JSON per session.
  - `plot_manual_family_trajectories.py` - manual-workflow regret plotting from saved train/test trajectory artifacts.
- `tests/` - test suite for objectives, GP/acquisition behavior, benchmark paths, and server logic (`*_test.py`; run with `uv run pytest tests`).
- `test_results/` - day-to-day outputs (Section 11).
- `benchmark_results/` - archived milestone runs (Section 12), including `naf_branin/` with the trained Branin NAF.
- `meta-bo-training/` - saved TAF source runs (`taf-gps/`), scratch-server artifacts (`server-scratch-artifacts/`) and NAF training runs (`naf/`).
- `notebooks/` - planned teaching notebooks (`01_bo_from_scratch` … `05_meta_bo_taf_conbo_naf`); currently empty placeholders.

## License and citation

OpenBO is licensed under the MIT License; see [LICENSE](LICENSE). To cite OpenBO, and the paper behind each method you use, see the website's [Publications](https://www.csie.ntu.edu.tw/~yichi/openbo-website/publications.html) page; the BibTeX entries are in [`relevant_papers/references.bib`](relevant_papers/references.bib). OpenBO is maintained by Yi-Chi Liao (yichi@csie.ntu.edu.tw).
