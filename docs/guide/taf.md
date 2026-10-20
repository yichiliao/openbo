# TAF: transfer from source tasks

[← Guide index](README.md) · [README](../../README.md)

## Meta-BO training and testing

Here we describe the **meta-BO loop** in-repo: define a task family and split, train on source tasks (e.g. scratch BO saving GPs and trajectories), then evaluate transfer methods such as TAF on held-out tasks using the same benchmark scripts as the [single-function](single-function.md) and [family](families.md) guides.

NAF (MetaBO) follows the same three steps with its own family, training script and benchmark; see [NAF](naf.md).

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

Note: this TAF differs from the original (Wistuba et al.) in how a source's improvement is measured. Each source is compared with a fixed 90% quantile of its own data, through a softplus, instead of with its best prediction over the points the target has already evaluated. So the sources keep pulling even after the target has explored their optima, and with many sources TAF can stall late in a run (see the ConBO benchmark in [Archived results](results.md#archived-benchmark-results)). `bo_taf_m` also weights sources by meta-feature similarity, not by predictive variance as in the original TAF-ME.

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
- The first `source_only_warmup_iters` (default 3) suggestions use the sources only. The warmup counts the target's observed points, however they arrived: with `--n-init 5` the random batch already has 5 points, so TAF uses the target GP from its first suggestion.
- `run_benchmark.py` and `plot_results.py` pass `--n-init` to TAF (since 2026-10-21; earlier it was ignored and TAF always started from the sources alone). Without `--n-init`, TAF still starts from the sources alone.
- `bo_taf_m` / `bo_taf_r` are recommended for cleaner benchmark tracking.
- TAF loads one source GP per `gp_states/<task>.json` file under `--taf-run-dir`, including subfolders (`gp_states/<sub>/<task>.json` is paired with `trajectories/<sub>/<task>.json`). A task without a matching trajectory is skipped with a warning. Two tasks with identical data (for example a backup copy of a run kept inside `gp_states/`) both load, and each repeat triggers a warning because it counts the same task twice. Only `gp_states/` is scanned. Check the number of sources (`n_sources` in the run metadata) if results look off.
- The committed `meta-bo-training/taf-gps/branin_train_v1` has all 12 tasks at the top level of `gp_states/` and `trajectories/`, the same layout `train_taf.py` writes, so re-running the training command overwrites the files instead of adding a second copy. (Earlier, 11 of the 12 tasks sat in a nested `train_task_0/` subfolder and the loader read only the top level, so TAF runs pointed at this directory used a **single** source. The output of the TAF fake client and server with their default config, which point at this run, came from that single-source version.)

## End-to-end TAF workflow: family split → scratch GPs → TAF

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
