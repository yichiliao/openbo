# NAF+: a neural AF with objective weights and a novelty fallback

[← Guide index](README.md) · [README](../../README.md)

NAF+ follows Liao et al., [*Efficient Human-in-the-Loop Optimization via Priors Learned from User Models*](https://dl.acm.org/doi/full/10.1145/3772318.3791976) (CHI 2026) in spirit, not in detail. It is [NAF (MetaBO)](naf.md), unchanged, plus the paper's two additions, and its numbers are **not a reproduction** of the paper's figures. Where the CHI 2026 paper and MetaBO disagree, OpenBO follows MetaBO; where the paper is silent or its mechanism is weak, OpenBO decides on merit. The design record, with every conflict, decision and measurement, is [`docs/plans/naf_plus.md`](../plans/naf_plus.md) (start from its "Handoff" and "Design stance" sections).

Status: the double-sphere users of the paper's App. B (Test 1, in an original and a wide family), the weight-conditioned AF, the novelty detector, training, the ask/tell optimizer and the benchmark work. There is no WebSocket server yet, and the paper's typing simulation (Test 2) is not implemented. The family's shift and scale ranges are not in the paper, so MetaBO's (±0.1, 0.9–1.1) are used.

## The pieces

- **Objective weights as AF inputs.** Each user has several objectives, combined as `sum_k w_k f_k(x)`. The AF reads the weights `w`, so one trained AF serves any trade-off (`NAFConfig.n_weights`; `0` is plain MetaBO NAF, see [NAF](naf.md)).
- **A novelty detector with an EI fallback.** A BNN trained on simulated users predicts each observation. The mean p-value `p_bar` of the user's observations so far sets an EI weight: `lambda_EI = 1` at `p_bar = 0`, `0` at `p_bar >= tau = 0.1`, linear in between. From iteration 4 on, the next point maximizes `lambda_EI * EI + (1 - lambda_EI) * naf`, both min-max normalized over the global grid.
- Code: `test_functions/naf_plus_families.py` (double-sphere users, `--family double_sphere` or `double_sphere_wide`), `acquisition/naf_plus.py` (p-values, `ei_weight`, the mixed score, and OpenBO's calibrated test), `optimizers/naf_plus.py` (`NAFPlusConfig`, `NoveltyDetector`, `NAFPlusSequentialOptimizer`, `train_naf_plus`, `run_naf_plus`) and `benchmarks/naf_plus_runner.py`.
- **The wide family.** On the original double sphere (centres (0.4, 0.4) and (0.6, 0.6)) the weights move the optimum by only 0.28, and the AF barely learned to use them. `double_sphere_wide` moves the training centres to (0.2, 0.2) and (0.8, 0.8), and its novel users (Test 1.2) sit on the other diagonal. On this family the weight conditioning is learned, and the EI fallback fires when it should and rescues novel users.

## Train

The detector is fitted first (about 16 minutes), then the AF is trained with PPO (about 90–120 s per iteration with two runs in parallel on 8 cores):

```bash
uv run python scripts/train_naf_plus.py --run-id double_sphere_v1 --n-iterations 150 --workers 4 --update-threads 4
# The wide family: add --family double_sphere_wide (and a new --run-id).
# The "without weights" ablation, reusing the detector:
uv run python scripts/train_naf_plus.py --run-id double_sphere_no_w_v1 --no-weights \
  --detector-from meta-bo-training/naf_plus/double_sphere_v1 --n-iterations 150 --workers 4 --update-threads 4
```

Runs go to `meta-bo-training/naf_plus/<run-id>/`: the NAF files (`af.pt`, `best_af.pt`, `naf_config.json`, `training_history.json`, `training_state.pt`) plus `naf_plus_config.json`, `detector.pt` and `detector.json`. Continue one with `--resume --n-iterations N`. Four trained runs are included: `double_sphere_v1`, `double_sphere_no_w_v1`, `double_sphere_wide_v1` and `double_sphere_wide_no_w_v1` (150 iterations, one seed each).

## Run in Python

```python
from openbo.optimizers.naf_plus import NAFPlusSequentialOptimizer, run_naf_plus
from openbo.benchmarks.naf_plus_runner import naf_plus_test_tasks

task = naf_plus_test_tasks("double_sphere", n_tasks=1)[0]  # a user with random weights
opt = NAFPlusSequentialOptimizer.from_run("meta-bo-training/naf_plus/double_sphere_v1", task.weights)
x, y = run_naf_plus(task, opt)  # y is the weighted sum of the objectives
print([(h.t, h.p_bar, h.lambda_ei) for h in opt.history])
```

`use_fallback=False` drops the detector (the "without EI" ablation). `rng=np.random.default_rng(0)` samples the AF's points from its softmax, as in training, instead of taking the argmax.

## Benchmark and plot

100 unseen users per test, 20 evaluations each. `--test double_sphere` is Test 1.1 (training-like users) and `double_sphere_novel` is Test 1.2 (novel centres); with a wide-family run the tests use the wide family.

```bash
uv run python scripts/run_naf_plus_benchmark.py --method naf_plus --test double_sphere --test-id v1 --workers 8 \
  --run-dir meta-bo-training/naf_plus/double_sphere_v1 --run-dir-no-w meta-bo-training/naf_plus/double_sphere_no_w_v1
uv run python scripts/plot_naf_plus_results.py --test-id v1
```

| Method | What it runs |
| --- | --- |
| `naf_plus` | the AF's argmax with the EI fallback (MetaBO's test protocol) |
| `naf_plus_no_ei` | the same without the fallback |
| `naf_plus_no_w` | the AF trained without weights (`--run-dir-no-w`) |
| `naf_plus_sample`, `naf_plus_no_w_sample`, `naf_plus_no_ei_sample` | the AF sampled from its softmax; a deviation from MetaBO's protocol, reported next to it |
| `naf_plus_norepeat`, `naf_plus_no_w_norepeat`, `naf_plus_no_ei_norepeat` | the argmax, never a point already evaluated (`avoid_repeats`) |
| `naf_plus_mix` | the same novelty test, but EI and the AF mixed as policies: with probability `lambda_EI` the point is EI's |
| `naf_plus_cal`, `naf_plus_cal_sample` | a calibrated novelty test (sum of squared standardized residuals against simulated training-like users, `tau = 0.05`) plus the policy mix, greedy and sampled |
| `ei`, `taf_plus_15`, `random` | EI on the same fixed GP, TAF+ with 15 earlier users, random search |

The calibrated methods need `novelty_calibration.json` in the run folder; write it once with `uv run python scripts/calibrate_naf_plus_novelty.py --run-dir <run> --workers 8` (500 users per policy, a few minutes). `scripts/diagnose_naf_plus_weights.py` runs the same users under all 11 weights `w1 = 0, 0.1, ..., 1` to check whether the AF uses its weight inputs (`--novel` for Test 1.2's users).

Results go to `test_results/naf_plus/` (`trajectories/<test-id>_<method>_<test>.json` with `p_bar` and `lambda_ei` per step for the NAF+ methods, and `plots/`).

## Results

The wide family's runs are archived in [`benchmark_results/naf_plus_wide/`](../../benchmark_results/naf_plus_wide/README.md) (models, calibration, Tests 1.1 and 1.2, weight diagnostics, `run_all.sh`), one training seed per model. In short: greedy NAF+ beats the AF trained without weights at every evaluation on Test 1.1, ties EI early and beats it at evaluation 5, while EI still leads at 15–20; on Test 1.2 the fallback fires for 65% of users and makes NAF+ the best method at evaluation 20 for users far from the training ones. TAF+ with 15 earlier users wins the first evaluations, because NAF+'s first point does not use the population.

The first runs on the original family (`double_sphere_v1`) are not archived; the plan records them: with the argmax NAF+ lost to EI (it repeats points, 37% of its evaluations), sampled it beat EI at evaluations 2–20, and the fallback rarely fired.
