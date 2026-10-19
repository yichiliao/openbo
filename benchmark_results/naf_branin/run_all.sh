#!/usr/bin/env bash
# Reproduces benchmark_results/naf_branin: MetaBO (Volpp et al., ICLR 2020) on the Branin family.
# 1. Train the neural AF with PPO on the 50 MetaBO-50 tasks. The archived run was started
#    with --n-iterations 300 and stopped by hand after iteration 235 (236 iterations); the
#    benchmark uses best_af.pt (best mean training return, iteration 193). Training slows from
#    about 52 s to 140 s per iteration on 8 cores (see docs/plans/naf.md, open question 10),
#    so this step takes about 5 hours.
# 2. Run every method on 100 unseen Branin tasks (test seed 100, 30 evaluations each).
#    naf, ei and random take about a minute in total; taf_me_50 and taf_r_50 about 40 minutes each.
# 3. Plot the regret curves, the training curve and a table.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=benchmark_results/naf_branin
ID=v1

uv run python scripts/train_naf.py --tasks metabo50 --n-iterations 236 --workers 7 --update-threads 8 \
  --run-id branin_metabo50_v1 --output-dir "$R/training"

for m in naf ei random taf_me_50 taf_r_50; do
  uv run python scripts/run_naf_benchmark.py --method "$m" --af-dir "$R/training/branin_metabo50_v1" \
    --n-tasks 100 --budget 30 --test-seed 100 --workers 8 --results-dir "$R" --test-id "$ID"
done

uv run python scripts/plot_naf_results.py --results-dir "$R" --test-id "$ID" --steps 1 3 5 10 15 20 30 \
  --training-dir "$R/training/branin_metabo50_v1"
