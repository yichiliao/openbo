#!/usr/bin/env bash
# ConBO Test 1 (Branin) with 5 seeds: conbo vs bo, taf_m, taf_r, single_gp.
# Paper settings: 15 users (shift_range 0.3, scale_range 0.2), 30 evaluations
# each, r0 = 16 random starts decaying by 5 per user, lambda = 50.
# 25 single-threaded jobs, 7 at a time; about 25-30 minutes on 8 cores.
# Usage (from the repo root): bash benchmark_results/conbo_branin_5seeds/run_all.sh
set -u
D=benchmark_results/conbo_branin_5seeds
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$D/logs"
for m in conbo taf_r taf_m single_gp bo; do
  for s in 0 1 2 3 4; do echo "$m $s"; done
done | xargs -P 7 -L 1 sh -c \
  'uv run python scripts/run_conbo_benchmark.py --method "$0" --seeds "$1" --test-id t1 \
     --results-dir '"$D"' > '"$D"'/logs/"$0"_s"$1".log 2>&1'
uv run python scripts/plot_conbo_results.py --results-dir "$D" --test-id t1 --from-user 4
