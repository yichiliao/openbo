# NAF+ on the wide double-sphere family (2026-11-11 to 2026-11-12)

OpenBO's NAF+ (after Liao et al., CHI 2026, in spirit, not a reproduction) on the **wide** double-sphere family. Every number here comes from **one training seed per model**. Design record and full discussion: `docs/plans/naf_plus.md`, "Design stance" (sections "Wide-family results" and "Items 2 and 3 on the wide family"). `run_all.sh` reproduces everything.

## Setup

- **Users.** Each user has two objectives, `f_k(x) = 1 − 8‖x + δ − c_k‖²` on `[0, 1]²`, with a shift `δ ~ U(−0.1, 0.1)²` and an output scale `S ~ U(0.9, 1.1)`. The user's weights are `w ~ Dirichlet(1)`. The optimizer observes `S·(w1 f1 + w2 f2)` and is told `w`, but not `δ` or `S`.
- **Training users** have centres (0.2, 0.2) and (0.8, 0.8), so the weights move the optimum by up to 0.85.
  - Test 1.1 (`double_sphere`) draws new users of the same kind.
  - Test 1.2 (`double_sphere_novel`) puts the centres on the other diagonal, (0.2, 0.8) and (0.8, 0.2).
  - A novel user's optimum is `|0.6 − 1.2 w1|` from the optimum of a training user with the same weights. At `w1 = 0.5` it is the same function, and at the extremes it is 0.6 away. Test 1.2 is reported by this distance: near < 0.2 (38 users), mid 0.2–0.4 (34), far ≥ 0.4 (28).
- **Fixed GP:** isotropic lengthscale 0.55, variance 6.07.
- **Tests:** 100 users per test, 20 evaluations each, test seed 100.
- **Training:** PPO, 150 iterations, seed 0.
  - `training/double_sphere_wide_v1` (with weights): best return 63.7 at iteration 126.
  - `training/double_sphere_wide_no_w_v1` (`--no-weights`, the same detector): best return 64.8 at iteration 120.
  - The benchmark uses `best_af.pt`. `double_sphere_wide_v1/novelty_calibration.json` holds the calibrated test's null distribution.

## Methods

- `naf_plus`: the default. Greedy AF (its argmax); a mean-p-value novelty test with `τ = 0.1` from evaluation 4; EI blended in by eq. 4.
- `naf_plus_no_ei`: the AF alone.
- `naf_plus_no_w`: the AF trained without weights.
- Variants: `_sample` samples the AF from its softmax; `_norepeat` skips points already evaluated.
- `naf_plus_mix`: the same test, but EI and the AF chosen by a coin with probability `λ_EI`.
- `naf_plus_cal[_sample]`: the calibrated test (`S_n = Σ z²` against 500 simulated training users per policy, `τ = 0.05`) plus the coin.
- Baselines: `ei` (plain BO on the same GP), `taf_plus_15` (TAF+ with 15 earlier users), `random`.

## Results

**Test 1.1** (training-like users), median log10 simple regret:

| evaluation | 1 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| `naf_plus` | 0.21 | −0.01 | −0.43 | −2.33 | −4.30 |
| `naf_plus_no_w` | 0.67 | 0.08 | −0.06 | −0.22 | −0.29 |
| `naf_plus_sample` | 0.21 | −0.33 | −0.85 | −3.82 | −4.60 |
| `ei` | 0.24 | 0.18 | −0.10 | −2.43 | −5.09 |
| `taf_plus_15` | −1.24 | −1.29 | −1.30 | −1.33 | −1.37 |
| `random` | 0.16 | −0.25 | −0.51 | −0.86 | −1.09 |

**Test 1.2** (novel users), median log10 regret at evaluations 5 / 10 / 20. The last column is the share of users who got any EI.

| | all users | far (≥ 0.4) | near (< 0.2) | fired |
|---|---|---|---|---|
| `naf_plus_no_ei` | −0.23 / −2.13 / −4.21 | −0.31 / −0.38 / −0.68 | −0.00 / −2.92 / −4.33 | — |
| **`naf_plus`** | −0.19 / −2.77 / **−5.31** | −0.31 / −2.64 / **−5.49** | −0.00 / −2.85 / −4.42 | 65% |
| `naf_plus_mix` | −0.18 / −2.49 / −4.55 | −0.33 / −2.51 / −5.04 | 0.08 / −3.02 / −4.38 | 65% |
| `naf_plus_cal` | 0.02 / −2.66 / −5.04 | −0.31 / −2.59 / −5.07 | 0.14 / −2.96 / −4.46 | 82% |
| `naf_plus_no_ei_sample` | **−0.99 / −3.81** / −4.44 | **−1.11 / −3.99** / −4.71 | −0.83 / −3.69 / −4.32 | — |
| `naf_plus_sample` | −0.63 / −3.08 / −5.19 | −0.85 / −3.00 / −5.45 | −0.61 / −3.14 / −4.34 | 63% |
| `ei` | −0.04 / −2.43 / −5.21 | −0.20 / −2.41 / −5.07 | −0.00 / −2.58 / −5.36 | — |

On Test 1.1 the fallback fired for 5% of users (`naf_plus`), 8% (`naf_plus_cal`) and 2–10% (sampled variants). No method differs significantly from its no-EI counterpart there.

## Findings

1. **The weight conditioning is learned on this family.** It was not on the old one (centres (0.4, 0.4)/(0.6, 0.6)), where the weights moved the optimum by only 0.28.
   - Greedy NAF+ beats the AF without weights at every evaluation on Test 1.1.
   - It ties EI at evaluations 1–3, beats it at 5 and ties at 10. In the old run it lost early. EI still leads at 15–20.
   - In the weight diagnostic (`weights/wide_v1_*`), NAF+ solves 90–100% of users at every weight except `w1 = 0` (65%). The AF without weights solves users only near `w1 = 0.5` (0–65%), and that is the signature of ignoring the weights.
   - The first point moves with the weights, toward the correct corner. The `t = 0` prior is still close to uniform (at most about 1.6×), so the early lead is small.
2. **The novelty fallback works on this family.**
   - It fires for 5% of Test 1.1 users and, on Test 1.2, for 18% of near, 88% of mid and 100% of far users.
   - On far users it lifts greedy NAF+ from −0.68 to −5.49 at evaluation 20.
   - On Test 1.2 it beats `naf_plus_no_ei` at evaluations 5–20 and EI at evaluation 5, and ties EI at 10–20 (paired Wilcoxon, p < 0.05).
   - On the old family the same test almost never fired, because the novel users there were within the population's own spread.
3. **Mixing the policies (a coin) is worse than mixing the scores (eq. 4).** `naf_plus` beats `naf_plus_mix` at evaluations 15–20 on Test 1.2. On novel users the AF's own points are the weak ones, and the coin still takes them `1 − λ` of the time.
4. **The calibrated test fires too often.**
   - Per check it is calibrated (95% quantiles of `S_n` near χ²_n's), but over 17 checks 8–13% of typical users are flagged at some point.
   - It flags 55% of near novel users, which costs at evaluation 5, and 60–65% of users only 0.12 from a training user (weight diagnostic, `weights/wide_v1_novel_*`).
   - It does fix every far user (100% solved at every weight).
   - It stays an option, not the default.
5. **The sampled AF alone is the best method up to evaluation 10 on both tests**, even on far novel users. It then stalls around −4.4 to −4.7. With it, the fallback costs at evaluations 3–5 and helps at 10–20. The next gain is in this late stall.

## Files

- `training/`: both runs (`best_af.pt`, `af.pt`, configs, `training_history.json`, `training_state.pt`, the detector, the calibration) and their logs.
- `trajectories/wide_v1_<method>_<test>.json`: every method on both tests (except `naf_plus_no_w_norepeat` on Test 1.2).
- `plots/`:
  - `wide_v1_<test>_core.png`: the four core conditions. They are NAF+, NAF+ without EI, NAF+ without weights (yellow here, because orange fails the colour check against EI's magenta) and EI. On Test 1.1 the NAF+ and no-EI lines coincide, because the fallback does not fire there. `naf_plus_no_w` keeps the fallback, which is why it recovers on Test 1.2.
  - `wide_v1_<test>.png`: all methods.
  - `wide_v1_<test>_fallback.png`: the fallback variants against EI.
  - `wide_v1_<test>_lambda.png`: `λ_EI` over the evaluations.
- `weights/`: the weight diagnostics on Test 1.1's users (`wide_v1_*`) and on Test 1.2's (`wide_v1_novel_*`). Each has the JSON with every run, the prior heatmaps, regret and distance per evaluation, and regret against `w1`.
