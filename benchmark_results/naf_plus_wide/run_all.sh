#!/usr/bin/env bash
# Reproduces benchmark_results/naf_plus_wide: NAF+ (OpenBO's version, after Liao et al., CHI 2026)
# on the wide double-sphere family (training centres (0.2, 0.2)/(0.8, 0.8), novel users on the
# other diagonal), 2026-11-11 to 2026-11-12. One training seed per model. Plan:
# docs/plans/naf_plus.md, "Design stance" ("Wide-family results", "Items 2 and 3").
#
# Code: training and Test 1.1 of the original methods at commit 8960223; the novelty
# calibration, the naf_plus_mix/naf_plus_cal methods, Test 1.2 and the novel-user diagnostic at
# commit 2f7439c (branch feat/naf-plus-revision). The archived runs were made in
# meta-bo-training/naf_plus/ and test_results/naf_plus/ and copied here, so the run_dir
# fields inside the trajectories point there.
#
# 1. Train both AFs, 150 PPO iterations each, run in parallel with 4 workers and 4 update
#    threads each: about 185 s (with weights) and 240 s (without) per iteration, about
#    6.5 hours. The archived runs were paused once (with weights after iteration 116,
#    without after 105) and continued with --resume. The no-weights run copies the
#    detector of the first.
# 2. Calibrate the novelty test (500 users per policy, about 15 minutes on 6 workers).
# 3. Test 1.1 (double_sphere) and Test 1.2 (double_sphere_novel): 100 unseen users each, random
#    weights, 20 evaluations, test seed 100. About 1 minute per method on 8 workers; TAF+ 5-10.
#    naf_plus_no_w_norepeat was not run on Test 1.2.
# 4. The weight diagnostics: every Test 1.1 user (and every Test 1.2 user) under all 11
#    weights, about 15 minutes each on 8 workers.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=benchmark_results/naf_plus_wide
T=$R/training
W=$T/double_sphere_wide_v1
WN=$T/double_sphere_wide_no_w_v1
ID=wide_v1

uv run python scripts/train_naf_plus.py --family double_sphere_wide --run-id double_sphere_wide_v1 \
  --n-iterations 150 --workers 4 --update-threads 4 --output-dir "$T" > "$T/double_sphere_wide_v1.log" 2>&1 &
until [ -f "$W/detector.json" ]; do sleep 30; done
uv run python scripts/train_naf_plus.py --family double_sphere_wide --run-id double_sphere_wide_no_w_v1 \
  --no-weights --detector-from "$W" --n-iterations 150 --workers 4 --update-threads 4 \
  --output-dir "$T" > "$T/double_sphere_wide_no_w_v1.log" 2>&1
wait

uv run python scripts/calibrate_naf_plus_novelty.py --run-dir "$W" --workers 6

for test in double_sphere double_sphere_novel; do
  for m in naf_plus naf_plus_no_w naf_plus_no_ei naf_plus_sample naf_plus_no_w_sample naf_plus_no_ei_sample \
           naf_plus_norepeat naf_plus_no_w_norepeat naf_plus_no_ei_norepeat naf_plus_mix naf_plus_cal \
           naf_plus_cal_sample ei random taf_plus_15; do
    uv run python scripts/run_naf_plus_benchmark.py --method "$m" --test "$test" --run-dir "$W" \
      --run-dir-no-w "$WN" --workers 8 --results-dir "$R" --test-id "$ID"
  done
done
uv run python scripts/plot_naf_plus_results.py --results-dir "$R" --test-id "$ID"
uv run python scripts/plot_naf_plus_results.py --results-dir "$R" --test-id "$ID" --suffix _fallback \
  --methods naf_plus naf_plus_no_ei naf_plus_cal naf_plus_cal_sample ei
uv run python scripts/plot_naf_plus_results.py --results-dir "$R" --test-id "$ID" --suffix _core \
  --methods naf_plus naf_plus_no_ei naf_plus_no_w ei --colours "naf_plus_no_w=#eda100" --steps 1 3 5 10 15 20

uv run python scripts/diagnose_naf_plus_weights.py --run-dir "$W" --run-dir-no-w "$WN" --workers 8 \
  --out-dir "$R/weights" --test-id "$ID"
uv run python scripts/diagnose_naf_plus_weights.py --run-dir "$W" --run-dir-no-w "$WN" --workers 8 \
  --out-dir "$R/weights" --test-id "${ID}_novel" --novel --methods naf_plus ei naf_plus_cal
