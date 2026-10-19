#!/usr/bin/env bash
# Reproduces benchmark_results/naf_goldstein_price: MetaBO (Volpp et al., ICLR 2020) on the
# Goldstein-Price family, trained for 600 iterations (the reference trains about 2000).
# 1. Train the neural AF with PPO on the 50 MetaBO-50 tasks: about 9 s per iteration on 8 cores,
#    about 1.5 hours. Training is deterministic for a seed on one machine, but the
#    result can differ in the last digits on another one (BLAS and float32).
#    The benchmark uses best_af.pt (best mean training return, iteration 499).
# 2. Run every method on 100 unseen Goldstein-Price tasks (test seed 100, 30 evaluations each).
#    naf, ei and random take seconds; taf_me_50 and taf_r_50 about 5 minutes each on 8 workers.
# 3. Plot the regret curves, the training curve and a table.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=benchmark_results/naf_goldstein_price
ID=v1

uv run python scripts/train_naf.py --family goldstein_price --tasks metabo50 --n-iterations 600 \
  --workers 7 --update-threads 8 --run-id goldstein_price_metabo50_v1 --output-dir "$R/training"

for m in naf ei random taf_me_50 taf_r_50; do
  uv run python scripts/run_naf_benchmark.py --method "$m" --family goldstein_price \
    --af-dir "$R/training/goldstein_price_metabo50_v1" \
    --n-tasks 100 --budget 30 --test-seed 100 --workers 8 --results-dir "$R" --test-id "$ID"
done

uv run python scripts/plot_naf_results.py --family goldstein_price --results-dir "$R" --test-id "$ID" \
  --steps 1 3 5 10 15 20 30 --training-dir "$R/training/goldstein_price_metabo50_v1"

# 4. v2: resume a copy of the run to 2000 iterations (the reference's length), about 3.3 hours
#    more (8.4 s per iteration). It did not improve the model: best_af.pt stays at iteration 499
#    and the last iteration is worse (see README, "2000 iterations without improvement").
#    Only af.pt, the config, the history, the log
#    and the training plot of that run are archived.
cp -r "$R/training/goldstein_price_metabo50_v1" "$R/training/goldstein_price_metabo50_v2_2000"
uv run python scripts/train_naf.py --family goldstein_price --n-iterations 2000 \
  --run-id goldstein_price_metabo50_v2_2000 --output-dir "$R/training" --resume

