#!/usr/bin/env bash
# Reproduces benchmark_results/naf_hartmann3: MetaBO (Volpp et al., ICLR 2020) on the
# Hartmann-3 family, trained for 600 iterations (the reference trains about 2000).
# 1. Train the neural AF with PPO on the 50 MetaBO-50 tasks: about 16 s per iteration on 8 cores,
#    about 2.7 hours. Training is deterministic for a seed on one machine, but the
#    result can differ in the last digits on another one (BLAS and float32).
#    The benchmark uses best_af.pt (best mean training return, iteration 591).
# 2. Run every method on 100 unseen Hartmann-3 tasks (test seed 100, 30 evaluations each).
#    naf, ei and random take seconds; taf_me_50 and taf_r_50 about 11 minutes each on 8 workers.
# 3. Plot the regret curves, the training curve and a table.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=benchmark_results/naf_hartmann3
ID=v1

uv run python scripts/train_naf.py --family hartmann3 --tasks metabo50 --n-iterations 600 \
  --workers 7 --update-threads 8 --run-id hartmann3_metabo50_v1 --output-dir "$R/training"

for m in naf ei random taf_me_50 taf_r_50; do
  uv run python scripts/run_naf_benchmark.py --method "$m" --family hartmann3 \
    --af-dir "$R/training/hartmann3_metabo50_v1" \
    --n-tasks 100 --budget 30 --test-seed 100 --workers 8 --results-dir "$R" --test-id "$ID"
done

uv run python scripts/plot_naf_results.py --family hartmann3 --results-dir "$R" --test-id "$ID" \
  --steps 1 3 5 10 15 20 30 --training-dir "$R/training/hartmann3_metabo50_v1"

# 4. v2: resume a copy of the run to 2000 iterations (the reference's length), about 6.3 hours
#    more (16 s per iteration; --resume continues from the saved state), then run NAF with the new
#    best_af.pt (iteration 1454). The baselines do not depend on the NAF model, so only naf is
#    rerun (test id v2); copy the v1 baseline trajectories to v2_<method>_hartmann3.json to plot.
cp -r "$R/training/hartmann3_metabo50_v1" "$R/training/hartmann3_metabo50_v2_2000"
uv run python scripts/train_naf.py --family hartmann3 --n-iterations 2000 \
  --run-id hartmann3_metabo50_v2_2000 --output-dir "$R/training" --resume
uv run python scripts/run_naf_benchmark.py --method naf --family hartmann3 \
  --af-dir "$R/training/hartmann3_metabo50_v2_2000" \
  --n-tasks 100 --budget 30 --test-seed 100 --workers 8 --results-dir "$R" --test-id v2
