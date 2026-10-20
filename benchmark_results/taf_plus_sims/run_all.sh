#!/usr/bin/env bash
# Reproduces benchmark_results/taf_plus_sims: TAF+ App. B Simulations 1 and 2 (seed 0).
# Simulation 2: four similarity ranges, alpha = (0.3, 0.5, 0.2), decay searched per range.
# Simulation 1: near-identical users (range 0.01), six objective weights, no decay (TAF+ = taf).
# Then TAF-EHVI and plain qLogNEHVI on the Simulation 2 users (added 2026-10-21).
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=benchmark_results/taf_plus_sims
ID=v1
W=7
run() { uv run python scripts/run_taf_plus_benchmark.py --results-dir "$R" --test-id "$ID" "$@"; }

for sim in sim2_r005 sim2_r01 sim2_r02 sim2_r03; do
  run --method population --sim "$sim" --workers "$W"
  run --method decay_search --sim "$sim" --workers "$W"
  for m in taf_plus taf bo random; do run --method "$m" --sim "$sim" & done
  wait
done

# TAF-EHVI and plain qLogNEHVI (added 2026-10-21): one process per run, about
# 100 and 75 minutes per range on one thread each.
for sim in sim2_r005 sim2_r01 sim2_r02 sim2_r03; do
  for m in taf_ehvi mobo; do run --method "$m" --sim "$sim" & done
done
wait

run --method population --sim sim1 --workers "$W"
for a in "1 0 0" "0 1 0" "0 0 1" "0.33 0.33 0.34" "0.5 0.3 0.2" "0.3 0.5 0.2"; do
  # shellcheck disable=SC2086
  for m in taf bo random; do run --method "$m" --sim sim1 --alpha $a & done
  wait
done
