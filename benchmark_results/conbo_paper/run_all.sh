#!/usr/bin/env bash
# Reproduces the ConBO (CHI 2025) simulation tests with the paper's settings, for $SEEDS (default 0-4).
# Test 2: McCormick (mccormick_paper), Test 1: Branin (branin_paper), Test 3: revisiting users (branin_revisit).
# Jobs run $JOBS at a time (default 4). Results: trajectories/, then plots/ per test.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=benchmark_results/conbo_paper
ID=v1
SEEDS=${SEEDS:-"0 1 2 3 4"}
JOBS=${JOBS:-4}
ALL="conbo conbo_no_gp conbo_no_filter bnn_no_replay bnn_direct_replay bo taf_paper single_gp"
TESTS=${TESTS:-"mccormick_paper branin_paper branin_revisit"}

for test in $TESTS; do
  methods=$ALL
  [ "$test" = branin_revisit ] && methods="conbo bnn_no_replay"
  for seed in $SEEDS; do for m in $methods; do echo "$m $test $seed"; done; done |
    xargs -P "$JOBS" -L 1 sh -c 'uv run python scripts/run_conbo_benchmark.py --method $0 --test $1 --seeds $2 --results-dir '"$R"' --test-id '"$ID"
  uv run python scripts/plot_conbo_paper_figures.py --results-dir "$R" --test-id "$ID" --test "$test"
done
