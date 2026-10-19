# Single-function benchmarks

[← Guide index](README.md) · [README](../../README.md)

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

## Noise-free vs noisy objectives

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

## 2D heatmap + trajectory quick recipes

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
