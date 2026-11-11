#!/usr/bin/env bash
# TAF+ (CHI 2024) App. B Simulations 1 and 2, and App. D (extended run), for the seeds in $SEEDS (default 0-4).
# Simulation 2: ranges 0.05-0.3, alpha = (0.3, 0.5, 0.2), decay searched per range, 20 evaluations.
# Simulation 1: range 0.01, six alpha vectors, no decay (TAF+ = taf), 20 evaluations.
# App. D: range 0.2, alpha = (0.3, 0.5, 0.2), 60 evaluations for TAF+, TAF and BO (test id v1ext).
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=benchmark_results/taf_plus_paper
W=${WORKERS:-7}
SEEDS=${SEEDS:-"0 1 2 3 4"}
run() { uv run python scripts/run_taf_plus_benchmark.py --results-dir "$R" "$@"; }

for seed in $SEEDS; do
  for sim in sim2_r005 sim2_r01 sim2_r02 sim2_r03; do
    run --test-id v1 --method population --sim "$sim" --seed "$seed" --workers "$W"
    run --test-id v1 --method decay_search --sim "$sim" --seed "$seed" --workers "$W"
    for m in taf_plus taf bo random; do run --test-id v1 --method "$m" --sim "$sim" --seed "$seed" & done
    wait
  done

  run --test-id v1 --method population --sim sim1 --seed "$seed" --workers "$W"
  for a in "1 0 0" "0 1 0" "0 0 1" "0.33 0.33 0.34" "0.5 0.3 0.2" "0.3 0.5 0.2"; do
    # shellcheck disable=SC2086
    for m in taf bo random; do run --test-id v1 --method "$m" --sim sim1 --seed "$seed" --alpha $a & done
    wait
  done

  run --test-id v1ext --method population --sim sim2_r02 --seed "$seed" --workers "$W"
  run --test-id v1ext --method decay_search --sim sim2_r02 --seed "$seed" --workers "$W"
  for m in taf_plus taf bo; do run --test-id v1ext --method "$m" --sim sim2_r02 --seed "$seed" --n-evals 60 & done
  wait
done
