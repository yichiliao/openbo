#!/usr/bin/env bash
# Reproduces TAF+ (CHI 2024) App. B Simulations 3 and 4, for the seeds in $SEEDS (default: 0).
# Simulation 3: three base functions (Tables 7-9), range 0.1, alpha = (0.3, 0.5, 0.2), decay searched.
# Simulation 4: 3-sphere base function, range 0.2, 2/4/6/8 population models, decay (4, 0.2).
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=benchmark_results/taf_plus_paper
ID=v1
W=${WORKERS:-7}
SEEDS=${SEEDS:-0}
run() { uv run python scripts/run_taf_plus_benchmark.py --results-dir "$R" --test-id "$ID" "$@"; }

for seed in $SEEDS; do
  for sim in sim3_relocated sim3_branin sim3_branin_hartmann; do
    run --method population --sim "$sim" --seed "$seed" --workers "$W"
    run --method decay_search --sim "$sim" --seed "$seed" --workers "$W"
    for m in taf_plus taf bo random; do run --method "$m" --sim "$sim" --seed "$seed" & done
    wait
  done

  run --method population --sim sim4 --seed "$seed" --workers "$W"
  for k in 2 4 6 8; do run --method taf_plus --sim sim4 --seed "$seed" --d1 4 --d2 0.2 --n-models "$k" & done
  wait
  for m in bo random; do run --method "$m" --sim sim4 --seed "$seed" & done
  wait
done
