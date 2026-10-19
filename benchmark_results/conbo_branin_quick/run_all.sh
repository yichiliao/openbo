#!/usr/bin/env bash
# Quick ConBO vs standard BO comparison on the paper's Branin Test 1 settings:
# 15 users (shift_range 0.3, scale_range 0.2), 30 evaluations each,
# r0 = 16 random starts decaying by 5 per user, lambda = 50, one seed.
# Three single-threaded jobs in parallel; ConBO takes about 9 minutes.
# Usage (from the repo root): bash benchmark_results/conbo_branin_quick/run_all.sh
set -u
D=benchmark_results/conbo_branin_quick
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$D/results" "$D/logs"
for m in bo conbo_eval conbo_bo_steps; do
  uv run python "$D/run_comparison.py" "$m" 0 "$D/results/$m.json" > "$D/logs/$m.log" 2>&1 &
done
wait
uv run python "$D/plot_comparison.py" "$D"
