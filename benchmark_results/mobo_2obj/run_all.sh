#!/usr/bin/env bash
# MOBO benchmark on OKA2 and DTLZ1a, 5 methods x 3 seeds.
# Reduced budget: n_init=5, n_iter=40 (the paper uses 100 iterations x 50 repeats).
# One single-threaded job per (method, problem, seed), 7 in parallel.
# Each job writes trajectories/mobo_2obj_s<seed>_<method>_<problem>.json;
# plot_mobo_results.py merges seeds with --test-id mobo_2obj_s0 mobo_2obj_s1 mobo_2obj_s2.
# Usage (from the repo root): bash benchmark_results/mobo_2obj/run_all.sh
set -u
D=benchmark_results/mobo_2obj
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
JOBS=$D/jobs.txt
: > "$JOBS"
# Slowest method (BoTorch) first to shorten the tail.
for m in mobo_botorch mobo_ceipv_slf mobo_ceipv_mt mobo_ieipv random; do
  for s in 0 1 2; do for p in oka2 dtlz1a; do
    echo "--method $m --problem $p --seeds $s --n-iter 40 --test-id mobo_2obj_s$s" >> "$JOBS"
  done; done
done
echo "jobs: $(wc -l < "$JOBS")"
xargs -P 7 -I{} bash -c '
  args="{}"; name=$(echo "$args" | tr -d " -" | tr "." "p")
  if uv run python scripts/run_mobo_benchmark.py $args --results-dir '"$D"' > '"$D"'/logs/$name.log 2>&1;
  then echo "DONE $args"; else echo "FAIL $args"; fi
' < "$JOBS"
echo "ALL FINISHED"
