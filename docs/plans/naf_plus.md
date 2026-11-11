# NAF+ implementation plan

**Design stance (2026-11-11): keep the paper's spirit, not its details; OpenBO decides on merit.** See "Design stance" below before changing anything.

Status: **Phases 1–4 done** (2026-10-12): the double-sphere family, its fixed GP hyperparameters, the weight-conditioned NAF, the novelty detector with EI mixing, and the optimizer, training, runner and scripts. Phase 5 first run (v1) done, with greedy and sampled AF variants (see Phase 5). **Paused 2026-10-12 and moved to the last roadmap item (6)**; start from "Handoff" below. Phase 0 done except the author questions.

Reference: Y.-C. Liao, J. Belo, H.-S. Moon, J. Steimle and A. M. Feit, [*Efficient Human-in-the-Loop Optimization via Priors Learned from User Models*](https://dl.acm.org/doi/full/10.1145/3772318.3791976), CHI 2026 (HOMI framework, NAF+ method). Read in full from arXiv 2510.07754v2 (CC BY 4.0; the PDF is not redistributed in this repo), including appendices A–C. The paper links code at [yichiliao/homi](https://github.com/yichiliao/homi), but on 2026-10-12 that repo held only a 90-byte README, so the paper is the only source. Its first author is this repo's owner, so questions the paper leaves open go to them (see "Questions for the author").

## Rules

- **Where the CHI 2026 paper and MetaBO (paper or reference code) disagree, follow MetaBO**, and where the MetaBO paper and its code disagree, follow the code (`docs/plans/naf.md`). Reimplement; never copy (AGPL-3.0). Every such case is listed in "Conflicts with MetaBO" below.
- So NAF+ in OpenBO is **OpenBO's existing NAF plus the paper's two additions**: objective-weight inputs, and the novelty detector with its EI fallback. Everything else (features, network, candidate set, GP, PPO, reward) stays as in `docs/plans/naf.md`.
- With no objective weights and the fallback turned off, NAF+ must reduce exactly to the current NAF. A test pins this against the archived Branin AF.
- Record every decision, deviation and measurement in this file.

## Paper summary

**HOMI (framework).** (1) Choose a parameterized user model, optionally fitted on a few real users. (2) Sample synthetic users from its parameter distribution. (3) Train a meta-BO method on them offline. (4) Deploy it with real users. NAF+ is one way to realize step 3.

**NAF+ has four parts (Sec. 4, Fig. 2):**
1. **GP** on the current user's observations of the *scalarized* objective `Σ w_k f_k(x)` (BoTorch `SingleTaskGP` in the paper).
2. **naf**, a neural AF trained with PPO, as in MetaBO. The state per candidate is `[μ_t(x), σ_t(x), t/T, w_1..w_N]` (eq. 2), over a **fixed** grid. Training samples from `Cat[logits]`; deployment takes the argmax. The objective weights are inputs. During training they are sampled per episode, `α ~ U[0,1]`, `w = (α, 1−α)`.
3. **EI** on the same GP and candidates.
4. **Novelty detector**: an MC-dropout BNN (2 × 256, dropout 0.2, 2000 epochs) trained on synthetic `(x, w) → y` pairs. For each new observation, `z = (y − μ_BNN(x)) / σ_BNN(x)` (eq. 3) and `p = 2(1 − Φ(|z|))`. `p̄` is the running mean of `p` over the user's observations. The score is
   - `A(x) = λ_EI · EI(x) + (1 − λ_EI) · naf(x)` (eq. 4),
   - `λ_EI = 1` if `p̄ = 0`, `0` if `p̄ ≥ τ`, `(τ − p̄)/τ` otherwise (eq. 5), with `τ = 0.1`.
   - naf has full control for the first `K = 3` iterations (`λ_EI = 0`), and the mixing starts at iteration 4.

**Training (Sec. 5.6, App. B.2).** naf has 6 layers of 256 tanh units. The critic has the same architecture. Actor lr 3e-4, critic lr 1e-3, 10 PPO epochs, clip 0.15, 80,000 steps, and synthetic users generated on the fly. Reward per step: `−(f_max − f(x_t))` (Table 1, plain regret).

**Novelty-detector data.** User study: 11 weights `w_speed ∈ {0, 0.1, …, 1}` × 20 synthetic users × a 100-point grid gives 2000 pairs per weight. Synthetic tests: "all `(x, y)` pairs collected during training", with the weights as a BNN input.

**Synthetic Test 1, double sphere (App. B).** The input is 2D and there are 2 objectives, `f_k(x) = 1 − γ Σ_i (x_i − c_k,i)²` with `γ = 8`, `c_1 = (0.4, 0.4)` and `c_2 = (0.6, 0.6)`. A user is an input shift `δ ~ U(−r/2, r/2)²` (`x' = x + δ`) plus an output scale `S ~ U(1 − s/2, 1 + s/2)` on the weighted sum. **`r` and `s` are not stated.** There are 20 test users with random weights and a budget of 20.
- Test 1.1 (in distribution) runs NAF+, NAF w/o weights, TAF and BO.
- Test 1.2 (novel users: centres moved to (0.3, 0.3) and (0.7, 0.7)) runs NAF+, NAF w/o EI, TAF and BO.

**Synthetic Test 2, touchscreen typing (App. C, Sec. 5.2).** Key width and height are the two inputs, with speed and accuracy as objectives.
- Fitts' law gives `MT = a + b log2(D/W + 1)`.
- The dual-Gaussian touch model has covariance `diag(α_x W² + σ_αx², α_y H² + σ_αy²)`. Keystrokes are Monte Carlo against the key boundaries, and timing noise is `σ = 0.15 s`.
- Sentences come from the Enron Mobile set, at least 26 characters each.
- Normalization maps 5–22 WPM to 0–1 and 30–0% error to 0–1.
- Each user samples 6 parameters from Gaussians. Training means are `(144.3, 75.636, 0.0075, 1.296, 0.0108, 1.153)` with stds `(15, 15, 0.001, 0.01, 0.001, 0.01)`; novel users have means `(180, 120, 0.0145, 1.6, 0.03, 2.0)`.

**Baselines.**
- BO: BoTorch EI, 6 random points plus 14 EI steps.
- TAF: TAF+ with 15 prior users, each multi-objective GPs from 10 random + 15 BO steps, with decay `α1 = 4`, `α2 = 0.2`.
- Ablations: NAF+ without weight inputs (trained on random weights but blind to them) and NAF+ without EI (no fallback).

**User study (Sec. 5).** A VR mid-air keyboard with width and height in [20, 40] mm and objective `0.7·Speed + 0.3·Accuracy`, run against TAF+ and ConBO. 12 participants did 10 iterations each. NAF+ was better at iterations 2–3. Out of scope here except as the motivation for the server.

## Conflicts with MetaBO, and what OpenBO does

| Topic | CHI 2026 | MetaBO (code) | OpenBO NAF+ |
| --- | --- | --- | --- |
| Candidate set | Fixed grid | Adaptive set: 31 × 31 grid plus local Sobol clouds around the 5 best points | **Adaptive set** (`AdaptiveSet`), scored by the mixed score `A(x)` at test time and by naf logits in training |
| Per-point features | `μ, σ, t/T, w` (no `x`) | `μ, σ, x, t, T` (raw) | **`μ, σ, x, t, T`, then `w_1..w_N`** |
| Policy net | 6 × 256 tanh | 4 × 200 ReLU, N(0, 0.01²) init | **4 × 200 ReLU** |
| Value net | Same architecture as the actor | Separate 4 × 200 on `[t, T]` | 4 × 200 on **`[t, T, w]`**. Returns depend on `w`, which is known per episode; MetaBO never has a known per-episode input. An addition, not a conflict. |
| PPO | lr 3e-4 / 1e-3, 10 epochs, 80k steps | Table 4: lr 1e-4, 4 epochs, minibatch 60, 1200 steps per iteration, about 2000 iterations | **MetaBO (`NAFConfig` defaults)** |
| Reward | `−(f_max − f(x_t))` | `−log10 max(1e-20, R_t)` | **`−log10` regret**, with `f_max` per (user, weights) |
| GP | BoTorch `SingleTaskGP`, fitted | Fixed per-family hyperparameters, zero mean, no standardization, near-zero noise | **Fixed per-family**. Type-2 ML is ill-posed on the double sphere, so the variance is matched to the family's `E[y²]` and the lengthscale chosen by held-out likelihood (Phase 1); fitted noise for noisy families (Test 2) |
| Training tasks | New synthetic user each episode | MetaBO-50 (released) or full MetaBO | **Full MetaBO** (`task_sampler`), with a fresh user and fresh `w` per episode, because the user model generates users on demand (the paper's whole point). A fixed pool is a switch. |
| EI | BoTorch EI | Analytic EI on the fixed GP; 1 Sobol initial point when EI runs alone | Analytic EI on the **same fixed GP**, inside NAF+. No initial design for NAF+. |

These follow the CHI paper, because MetaBO has no counterpart: weight inputs, the novelty detector, eqs. 3–5, `τ = 0.1`, `K = 3`, budget `T = 20`, the double-sphere and typing families, and the ablations.

## Decisions (proposed)

| Topic | Decision |
| --- | --- |
| Scope of first pass | **Test 1 (double sphere)**, Tests 1.1 and 1.2. It is cheap, noise-free and analytic, so the fixed noise-free GP setup carries over unchanged. Test 2 (typing) comes after Test 1 works and the author's simulator details are in (Phase 7). |
| Scalarization | The family returns `(n, L)` per-objective values, so TAF+, the server and plots can use them. NAF+, its GP and the detector see `y = S · Σ w_k f_k(x')`. Weights lie on the simplex. |
| Score mixing (eq. 4) | naf logits and EI have different scales. **Default:** min-max normalize each over the current candidate set, then mix. This is open question 1; the author's answer replaces the default. |
| Novelty detector | `PopulationBNN` from ConBO (`BNNConfig(input_dim=d+N, hidden_units=256, n_hidden_layers=2, dropout=0.2)`, `fit_nll` for 2000 epochs). The input is `[x, w]` and `σ` is the total predictive std (MC-dropout plus aleatoric head). **Data:** the user-study recipe (11 weights × 20 users × 100 Sobol points), because it is explicit and does not depend on the policy. The "pairs from PPO episodes" recipe is an option. |
| `p̄` | The running mean of `p` over **all** of the user's observations, including the first 3, recomputed after each `observe`. `λ_EI = 0` while `t < K`. Open question 3. |
| Optimum | Analytic for the double sphere: the weighted sum of isotropic quadratics with equal `γ` is a quadratic centred at `Σ w_k c_k − δ`, clipped to the unit box per coordinate. A test checks it against `estimate_optimum`. |
| Test users | **100** unseen users with random weights (MetaBO's protocol and the NAF benchmark), seed 100. The paper's 20 is a CLI flag. |
| Code layout | `test_functions/naf_plus_families.py`; weight support added to `acquisition/naf.py` and `optimizers/naf.py` (opt-in, default off); new `acquisition/naf_plus.py` (detector p-values, `ei_weight`, `mixed_scores`), `optimizers/naf_plus.py` (`NAFPlusConfig`, `NoveltyDetector`, `NAFPlusSequentialOptimizer`, `train_naf_plus`, `run_naf_plus`), `benchmarks/naf_plus_runner.py`, scripts `train_naf_plus.py`, `run_naf_plus_benchmark.py`, `plot_naf_plus_results.py`, and `server_optimizers/naf_plus_server.py` (port **8773**; 8772 went to the TAF-MOBO server). Trained runs go to `meta-bo-training/naf_plus/<run_id>/` (`af.pt`, `best_af.pt`, `naf_config.json`, `detector.pt`, `naf_plus_config.json`). |

## Mapping onto OpenBO

| Paper | OpenBO |
| --- | --- |
| naf, PPO, episodes | `NeuralAF`, `BOEpisode`, `train_naf` (`optimizers/naf.py`). Extend with optional weights; no fork. |
| Novelty BNN | `models/bnn.py` `PopulationBNN` (MC dropout, mean and log-variance heads, own generator, save/load). |
| EI | `acquisition/ei.py`, on `gp_mean_std` outputs. |
| Shift-and-scale users | Like `TaskVariantSpec` and `BraninTask`. The double sphere needs its own task class because of the two objectives and the analytic optimum. |
| TAF baseline (decay `α1 = 4`, `α2 = 0.2`) | `acquisition/taf_plus.py`: `decay_factor(t, d1=4, d2=0.2)` is the same schedule (`population_weight`). Multi-objective sources with the test user's weights, as in `TAFPlusPopulation`. Use `taf_plus_acquisition` on the fixed-hyperparameter GPs and the adaptive set, as the NAF runner does. |
| BO baseline | `ei` in the NAF runner style (fixed GP, one Sobol point), plus `bo_paper` (6 random + 14 EI) if the comparison needs the paper's exact baseline. |

## Implementation phases

### Phase 0: Prerequisites
- ✅ The paper PDF is in `relevant_papers/`, and `references.bib` notes it.
- ✅ **NAF open question 10 (repeated points)**, settled 2026-10-12: the reference allows repeats, so NAF+ does too, and the 52 → 140 s slowdown was denormal floats in the PPO update, now flushed (about 63 s per iteration). NAF+ shares `_ppo_update`, so it gets the fix. NAF items 2–3 (longer training; Goldstein-Price and Hartmann-3) do not block NAF+.
- Answers to the author questions below, at least questions 1 and 2.

### Phase 1: Double-sphere family
- `test_functions/naf_plus_families.py`:
  - `double_sphere_objectives(x, centres, gamma=8)` returns `(n, 2)`.
  - `DoubleSphereUser(shift, scale, centres)` has `objectives(x) -> (n, L)`, `__call__(x, weights) -> (n,)`, and `argmax(weights)`/`optimum(weights)`.
  - `sample_double_sphere_users(n, seed, shift_range, scale_range, centres)`.
  - `NOVEL_CENTRES = ((0.3, 0.3), (0.7, 0.7))`.
  - `sample_weights(n_objectives, rng)` gives `(α, 1−α)` for 2 objectives and Dirichlet(1) in general.
- Fixed GP hyperparameters from `fit_family_hyperparameters` on one user at `w = (0.5, 0.5)`. Check the lengthscales against a few other weights; if they differ a lot, pool or report it.
- Tests: the analytic optimum beats a dense grid and L-BFGS-B on 250 (user, w) pairs; shapes; ranges; seeding.
- ✅ *Done 2026-10-12:*
  - `test_functions/naf_plus_families.py` as listed above, plus `WeightedTask` (a user with fixed weights; the single-objective `__call__`/`optimum` form NAF episodes take), `sample_weighted_tasks`, `validate_weights` and `TRAINING_CENTRES`. Each objective is scaled by `S` (the same as scaling the weighted sum, since it is linear), so per-objective consumers see the user's own values. `sample_weights` is Dirichlet(1), which for two objectives is the paper's `α ~ U[0, 1]`.
  - **Shift and scale ranges:** defaults `r = 0.2`, `s = 0.2` (MetaBO's ±0.1 and 0.9–1.1), provisional until author question 2 is answered. Then no in-distribution maximizer is clipped (the weighted centre lies in [0.4, 0.6]², the shift within ±0.1).
  - `optimizers/naf_plus.py`: `DOUBLE_SPHERE_GP_HYPERPARAMETERS` and `select_family_hyperparameters`.
  - Tests: `tests/naf_plus_family_test.py` (10), including the optimum against a 101² grid plus L-BFGS-B on 250 (user, w) pairs for training and novel centres, and the hyperparameter constant recomputed from its selection (about 6 s).
- **GP hyperparameters: type-2 ML is ill-posed here** (decided with the user 2026-10-12: matched variance).
  - `fit_family_hyperparameters` at different weights gave unstable results: lengthscales (0.367, 0.397) with variance 4.48 at `w = (0.5, 0.5)`, but (0.01, 0.74) at `w = (0.25, 0.75)` and (54, 0.003) at `w = (1, 0)`.
  - Cause: the noise-free double sphere is a quadratic, the limit of an RBF GP as lengthscale and variance grow together. The exact NLML on the `w = 0.5` fitting data keeps falling: −471 at the fitted values, −552 at (0.6, var 4.5), −654 at (1.0, var 20). The fit stops early only because `_robust_cholesky` silently escalates jitter on the ill-conditioned matrices. Held-out likelihood runs away the same way (best at the grid edge, lengthscale 1.0 with variance 80).
  - **Choice:** the variance is the family's `E[y²]` over the unit square (1.41, prior std 1.19, close to Branin's 1.41), and the isotropic lengthscale minimizes held-out predictive NLL averaged over 3, 10 and 20 observations at that variance: **0.35** (60 users with random weights; 0.30–0.40 within 0.2 nats). Settings with larger variances predict better at 20 points (median held-out RMSE 0.044 for lengthscale 0.6 with variance 4.5, against 0.153 for the `w = 0.5` fit), but only because the larger the prior, the closer the RBF GP gets to a quadratic; there is no stopping point, and a prior std well above the family's spread would put the AF's raw `σ` input on the wrong scale.
  - Novel-centre users select (0.45, var 2.77): their outputs spread about twice as much. Open question 4 is answered: novel users are out of distribution for the fixed GP too, by design, since the hyperparameters come from the training family only.

### Phase 2: Weight-conditioned NAF
- `naf_features(..., weights=None)` appends `w` to every row. `n_naf_features(dim, use_x, n_weights=0)`.
- `NAFConfig` gains `n_objectives: int = 0` (0 = current NAF). `BOEpisode` takes `weights` and computes regret against `optimum(weights)`. `task_sampler` returns `(task, weights)` when `n_objectives > 0`.
- The value net input becomes `[t, T, w]`; `value_features(t, budget, weights=None)`.
- `use_weights=False` with `n_objectives > 0` gives the "NAF+ w/o weights" ablation: the weights are sampled but hidden from the AF.
- `save_naf`/`load_naf` store `n_objectives` and `use_weights`. Old `naf_config.json` files load with 0.
- Tests: the archived Branin AF gives bit-identical suggestions; a 1D, two-objective toy family whose optimum moves with `w` is learned in a short run (the first greedy pick tracks `w`); a reproducibility test.
- ✅ *Done 2026-10-12, simpler than planned:*
  - **One switch, `NAFConfig.n_weights`** (default 0), instead of `n_objectives` plus `use_weights`. The weights belong to the task: a `WeightedTask` (Phase 1) already includes them in its values and optimum, and carries them as `task.weights`. The AF reads them if and only if `af.n_weights > 0`. So no `task_sampler` change was needed, and `BOEpisode` reads `task.weights` only for a weight-conditioned AF.
  - **The "NAF+ w/o weights" ablation is plain NAF (`n_weights = 0`) trained on weighted tasks**: the weights vary per episode, but neither the AF nor the value net sees them. The value net reads `[t, T, w]` exactly when the AF reads `w`.
  - `acquisition/naf.py`: `naf_features(..., weights=None)`, `n_naf_features(dim, use_x, n_weights=0)`, `NeuralAF(..., n_weights=0)`, `af_weights(af, weights)` (the weights an AF reads: `None` for a plain AF, which ignores given weights; an error if a weighted AF gets none or the wrong number), `naf_score_fn(..., weights=None)`.
  - `optimizers/naf.py`: `NAFConfig.n_weights`, `value_features(t, budget, weights=None)`, `Rollout.weights` (per step), the value net sized `2 + n_weights`, and `NAFSequentialOptimizer(..., weights=None)` / `from_directory(dir, name, weights=None)`, which checks the weights lie on the simplex. Old `naf_config.json` files and training states load with `n_weights = 0`, so the running Branin training can be resumed unchanged.
  - Tests: `tests/naf_plus_weights_test.py` (7, about 80 s): the archived Branin AF reproduces its archived benchmark trajectories bit for bit on 3 tasks; features, weight checks and value inputs; a hand-set AF whose suggestion flips with the weights (a random AF picks the same corner for every weight, so it cannot show this); save/load; seeded and parallel training; and a 1D toy (peaks at 0.2 and 0.8, `w = (1, 0)` or `(0, 1)`) where 300 short iterations teach the greedy first pick to go to the selected peak (within 0.05).

### Phase 3: Novelty detector and mixing
- `acquisition/naf_plus.py`:
  - `novelty_p_values(mean, std, y)` (eq. 3 and the two-tailed p).
  - `ei_weight(p_bar, tau)` (eq. 5).
  - `mixed_scores(naf_logits, ei, lam, normalization="minmax")` (eq. 4).
- `optimizers/naf_plus.py`:
  - `NoveltyDetector` wraps `PopulationBNN` with `p_value(x, w, y)`, `save`/`load`.
  - `build_detector_dataset(user_sampler, weight_grid, n_users, n_points, seed)`.
  - `fit_detector(...)`.
- Tests:
  - eq. 5 at its three branches and its continuity at `τ`;
  - `p` matches `scipy.stats.norm`;
  - in-distribution users give `p̄` well above `τ` on average, and novel-centre users give `p̄` below it. This sanity check ties the detector to Test 1.2.
- ✅ *Done 2026-10-12:*
  - `acquisition/naf_plus.py`: `novelty_p_values` (eq. 3, `2·Φ(−|z|)`), `ei_weight(p_bar, tau=0.1)` (eq. 5), `ei_score_fn(gp, best_y)` (analytic EI on the fixed GP), and `mixed_score_fn(naf_score, ei_score, lam, reference)` (eq. 4) in place of the planned `mixed_scores` on arrays. `NOVELTY_THRESHOLD = 0.1`, `NAF_ONLY_ITERATIONS = 3`.
  - **Score normalization (author question 1, default):** min-max over a **fixed reference set** (the adaptive set's global grid) each step, not per call. `AdaptiveSet.candidates` scores the grid and the local clouds in separate calls and then compares them, so per-call normalization would make those scores incomparable. A fixed affine map keeps the ranking of points outside the reference. `lam = 0` returns the pure neural AF score function itself (NAF+ then picks exactly what NAF picks, and EI is not computed); `lam = 1` returns EI.
  - `optimizers/naf_plus.py`: `weight_grid(n_objectives, step)` (the simplex lattice with zeros, unlike TAF+'s `objective_weight_grid`; 11 settings for two objectives), `build_detector_dataset(user_sampler, weights, dim, n_users=20, points_per_dim=10, seed)` (fresh users per weight, seed `seed + 1009·k`; 22,000 rows for the paper's recipe), `NoveltyDetector(bnn, dim, n_weights)` (`predict`, `p_values`, `save`/`load` as `detector.pt` plus `detector.json`), `detector_config` (2 × 256, dropout 0.2) and `fit_detector(..., epochs=2000)`. The detector is ConBO's `PopulationBNN`, unchanged.
  - **σ (author question 3, default):** the BNN's total predictive std. The spread between users at the same `(x, w)` lands in the variance head, so the test asks whether a value is plausible for a typical user; the epistemic part alone would shrink with data and flag every user.
  - Tests: `tests/naf_plus_detector_test.py` (8, about 30 s): p-values against `scipy.stats.norm`; eq. 5 at its branches and at `τ`; EI against `expected_improvement_maximization`; eq. 4 on a reference including a point outside it; the weight grid; the dataset layout; save/load; and a small detector (64 units, 400 epochs) that keeps typical users above `τ` and flags users with centres at (0.05, 0.05)/(0.95, 0.95).
  - **Measured: the paper's detector barely flags Test 1.2's novel users at the provisional ranges.** Full recipe (22,000 points, 2 × 256, 2000 full-batch epochs, Adam 1e-3): 1353 s with 4 threads beside the Branin training (300 epochs: 210 s). On 50 users with random weights and random designs, median `p̄` and the share below `τ = 0.1`: training-like users 0.53 / 2% after 3 points, 0.48 / 2% after 10; novel-centre users 0.26 / 18% after 3, 0.28 / 6% after 10. 300 epochs gave nearly the same numbers, so it is not undertraining. Moving the centres out by 0.1 stays within the population's own variation at shift ±0.1 and scale ±10%; at `w = (0.5, 0.5)` the novel optimum is even at the same place, (0.5, 0.5). So with these ranges, `λ_EI` stays 0 for most novel users and Test 1.2 cannot show the fallback. Open question 5.

### Phase 4: Optimizer, training entry point, runner
- `NAFPlusConfig` (frozen): `tau = 0.1`, `naf_only_iterations = 3`, `use_fallback = True`, `score_normalization`, and the detector settings.
- `NAFPlusSequentialOptimizer(af, detector, gp_hyperparameters, weights, ...)`:
  - `suggest()` maximizes `A(x)` over the adaptive set; `observe(x, y)` takes the scalarized `y`.
  - `history` records `p`, `p̄` and `λ_EI` per step (Fig. 7).
  - `from_directory(dir)`. It refuses to run past `T`, like NAF.
- `train_naf_plus(config, family, run_dir)` = `train_naf` with weights, then `fit_detector`. `run_naf_plus(...)`.
- `benchmarks/naf_plus_runner.py`:
  - `NAF_PLUS_METHODS = (naf_plus, naf_plus_no_w, naf_plus_no_ei, ei, taf_plus_15, random)`; Phase 5 added `naf_plus_sample` and `naf_plus_no_w_sample`.
  - `NAF_PLUS_TESTS = (double_sphere, double_sphere_novel)`.
  - `run_naf_plus_method(method, test, ...)` with worker processes under `threadpool_limits(1)`.
- `scripts/train_naf_plus.py` (with `--resume`), `run_naf_plus_benchmark.py`, `plot_naf_plus_results.py` (median log10 regret with 30–70% bands, and a `λ_EI` box plot per iteration).
- Tests: a session reproduces `run_naf_plus`; `use_fallback=False` equals pure argmax of the naf logits; the weights change the suggestion.
- ✅ *Done 2026-10-12:*
  - `optimizers/naf_plus.py`: `NAFPlusConfig` (frozen; `family`, `shift_range`, `scale_range`, `tau`, `naf_only_iterations`, detector settings; `users`/`tasks(n, seed, novel)`; `naf_plus_config.json`). No `use_fallback` or `score_normalization` fields: the fallback is switched off by loading without the detector (`from_run(..., use_fallback=False)`), and min-max over the global grid is the only normalization so far.
  - `NAFPlusSequentialOptimizer` subclasses `NAFSequentialOptimizer`: every `observe` adds the detector's p-values; `suggest` uses `λ_EI = 0` before iteration `naf_only_iterations + 1` (or without a detector) and eq. 5 from `p̄` after, mixing by `mixed_score_fn` over the global grid; `history` holds `(t, p̄, λ_EI)` per suggestion. `from_run(dir, weights, name="best_af", use_fallback=True)` (not `from_directory`, whose signature it would clash with).
  - `train_naf_plus(naf_config, plus_config, run_dir, log, resume, detector_from)`: fits the detector **first** (once; a killed AF run keeps it), or copies it from another run with the same settings (the ablation run reuses NAF+'s), then `train_naf` on fresh users with fresh weights each episode. Refuses a run directory with different NAF+ settings. `naf_config.n_weights` must be 0 (ablation) or the family's objective count.
  - `benchmarks/naf_plus_runner.py` as planned. `naf_runner._grid_bo`/`_ei_scores` became public (`grid_bo`, `ei_score_factory`) for reuse. `taf_plus_15`: 15 users from the training distribution (seed 7) with one fixed-hyperparameter GP per objective on 25 Sobol points; per-objective target GPs; `taf_plus_acquisition` with PoE weights, the user's weights and `decay_factor(t, 4, 0.2)`; first point = sources' PoE mean. The trajectory files record the user's weights and, for the NAF+ methods, the fallback history.
  - Scripts: `train_naf_plus.py` (defaults: 500 iterations, budget 20, 7 workers, output `meta-bo-training/naf_plus/<run-id>`; `--no-weights`, `--detector-from`, `--detector-epochs`, `--shift-range`, `--scale-range`, `--resume`), `run_naf_plus_benchmark.py` (`--method`, `--test`, `--run-dir`, `--run-dir-no-w`, output `test_results/naf_plus/`), `plot_naf_plus_results.py` (regret per test with slot-6 green added to the NAF palette, validated: all checks pass, contrast warning relieved by line styles, legend and table; `λ_EI` box plots per iteration).
  - Tests: `tests/naf_plus_optimizer_test.py` (10, about 25 s, two tiny trained runs in a fixture): without a detector, and with a typical user (stub p = 0.5), NAF+ suggests exactly what NAF does; a novel user (stub p = 0) gets EI's argmax from the iteration after `naf_only_iterations`; p = 0.05 gives the eq. 4 mix at λ = 0.5; the real detector scores every observation; training files, resume and settings checks; every method runs on novel users; parallel equals serial for `naf_plus` and `taf_plus_15`.
  - Smoke run of the three scripts (1 iteration, 5 detector epochs, 3 users): all work. One full-size PPO iteration with 2 workers beside the Branin training: 136 s. `taf_plus_15` costs about 17 s per user on one worker (15 sources × 2 objectives on about 6000 points per step), so use `--workers`.

### Phase 5: Test 1 benchmark
- Train two AFs, with and without weight inputs, on full-MetaBO double-sphere users, `T = 20`. **Measure** the time per iteration first. Branin is about 55–65 s after the denormal fix, so 300 iterations take about 5 h each, and the paper's 80k steps are only about 67 iterations of 1200 steps. Choose the iteration count from the training curve and record it here.
- Test 1.1 (in distribution) and Test 1.2 (novel centres), 100 users each, every method. Archive to `benchmark_results/naf_plus_double_sphere/` with `run_all.sh`. (2026-11-12: the wide family's runs are archived in `benchmark_results/naf_plus_wide/`, with `run_all.sh` and a README of the findings; v1 on the old family is not archived yet.)
- Report what the paper reports (median regret per iteration, NAF+ vs w/o-weights significance per iteration) plus mean regret over iterations 1–5, since early iterations are the HCI target (NAF open question 11).

**First run (v1, 2026-10-12).**
- **Training.** `double_sphere_v1` (with weights) and `double_sphere_no_w_v1` (`--no-weights`, detector copied) were trained in parallel, each with 4 workers and 4 update threads, for 150 iterations each (the user's choice, to finish today). The detector took 974 s. Each iteration took 92–155 s, depending on load.
  - With weights: mean return per 25-iteration block was 24 → 47 → 57 → 65 → 69, with the best of 73.2 at iteration 115 and 67.7 at iteration 149. It was flattening, and the entropy stayed high at about 2.6.
  - Without weights: the best was 66.8 at iteration 92. **The policy collapsed at iteration 128:** entropy rose from about 3.5 to 6 and the return fell to about 33, with no recovery by iteration 149. The benchmark uses `best_af` (iteration 92).
- **Benchmark** (`test_results/naf_plus/trajectories/v1_*`, 100 users per test, greedy AF as in MetaBO and the paper). Median log10 regret:

  | method | 1.1 @1 | @5 | @10 | @20 | 1.2 @1 | @5 | @10 | @20 |
  |---|---|---|---|---|---|---|---|---|
  | NAF+ | 0.56 | 0.10 | −1.53 | −3.70 | 0.56 | −0.15 | −1.75 | −4.57 |
  | NAF+ w/o weights | 0.47 | 0.29 | 0.17 | 0.17 | 0.49 | 0.19 | 0.08 | 0.07 |
  | NAF+ w/o EI | 0.56 | 0.10 | −1.53 | −3.70 | 0.56 | −0.14 | −1.71 | −3.56 |
  | TAF+ (15 users) | −1.26 | −1.57 | −1.76 | −1.88 | −1.08 | −1.44 | −1.58 | −1.63 |
  | EI | 0.13 | 0.02 | −2.41 | −4.80 | 0.20 | −0.00 | −2.37 | −4.75 |
  | Random | 0.13 | −0.41 | −0.72 | −0.96 | 0.13 | −0.40 | −0.79 | −1.06 |

  - The greedy AF repeats points: 37% of NAF+'s evaluations (Test 1.1) and 49% of the no-weights AF's are exact repeats. EI repeats 1%.
  - **The fallback barely fires.** On Test 1.1, λ_EI > 0 for 1–2% of users and changed the points of 2 of 100 users. On Test 1.2 it fires for 4–12% of users per iteration (mean λ_EI 0.02–0.06) and changed 20 of 100 users. Where it fires it helps: the median @20 improves from −3.56 to −4.57, by breaking the AF's repeats. This is consistent with question 5: the detector sees few novel users at the provisional ranges.
- **Greedy versus sampled.** Training samples from `Cat[logits]`, and deployment takes the argmax (MetaBO's code, and the paper's Sec. 4). Here the two diverge. On 100 users, sampling from the same `best_af`, median log10 regret was:

  | AF, sampled | 1.1 @1 | @5 | @10 | @20 | mean 1–20 | 1.2 @1 | @5 | @10 | @20 | mean 1–20 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | NAF+ | 0.21 | −0.87 | −4.23 | −5.58 | −3.49 | 0.26 | −0.90 | −3.65 | −5.52 | −3.36 |
  | w/o weights | 0.08 | −1.34 | −3.73 | −5.35 | −3.26 | 0.12 | −0.78 | −3.05 | −4.97 | −2.98 |

  Sampled, the AF beats EI from about evaluation 5 onwards on both tests. Greedy, it is worse than EI throughout. With the policy's entropy still about 2.6, much of the learned behaviour is in the spread of the distribution, and the argmax gets stuck repeating points. The paper's NAF+ results presumably used the argmax, so this gap may also explain why NAF+ needed the EI fallback.
- **Decision (user, 2026-10-12): add the sampled policy as extra benchmark methods; do not train longer.** Greedy stays the default (MetaBO's test protocol). `naf_plus_sample` and `naf_plus_no_w_sample` draw each AF point from `softmax` over `xi_t` (`AdaptiveSet.sample`, an `rng` on the optimizers, seeded per user with `seed + 1009 * task_index`). Once `lambda_EI > 0` the point is still the argmax of eq. 4, because the min-max-normalized mix is not a policy. This deviates from MetaBO's evaluation and is reported next to it, never in place of it.
- **Sampled benchmark** (`v1_naf_plus_sample_*`, `v1_naf_plus_no_w_sample_*`; 100 users, about 40 s each on 8 workers). Median log10 regret:

  | method | 1.1 @1 | @2 | @3 | @5 | @10 | @20 | 1.2 @1 | @2 | @3 | @5 | @10 | @20 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | NAF+ (sampled) | 0.22 | −0.06 | −0.38 | −0.90 | −4.15 | −5.57 | 0.24 | −0.10 | −0.50 | −0.93 | −3.62 | −5.56 |
  | w/o weights (sampled) | 0.08 | −0.12 | −0.31 | −1.03 | −3.48 | −5.42 | 0.10 | −0.12 | −0.26 | −0.88 | −3.40 | −5.17 |

  Mean of the median over 1–5 / 1–20 (Test 1.1; Test 1.2):
  - NAF+ greedy: 0.28 / −1.65; 0.22 / −1.96.
  - NAF+ sampled: −0.36 / −3.43; −0.39 / −3.34.
  - w/o weights sampled: −0.41 / −3.22; −0.33 / −3.03.
  - EI: 0.11 / −2.51; 0.15 / −2.45.
  - TAF+: −1.37 / −1.71; −1.28 / −1.52.

  Exact repeats: 23% sampled versus 37% greedy (Test 1.1). Paired Wilcoxon tests per step (p < 0.05):
  - NAF+ sampled beats EI at evaluations 2–5 and 7–20 on both tests. It is worse only at evaluation 1 on Test 1.1, where the first point is the AF's softmax draw, not a Sobol point.
  - NAF+ sampled beats its no-weights counterpart at evaluations 10–19 (Test 1.1) and 3–4 and 12–20 (Test 1.2). Sampled, the weights matter mainly late. Greedy, the no-weights AF never converges.
  - The fallback fired at some step for 2 of 100 users on Test 1.1 and 35 of 100 on Test 1.2. There is no sampled no-EI method, so its effect under sampling is not isolated.
- **Still open:** TAF+ is the best method for evaluations 1–5 on both tests. NAF's first draw ignores the population (NAF open question 11).
- No method beats TAF+ in evaluations 1–3. TAF+'s first point comes from 15 earlier users' product-of-experts mean, whereas NAF's first point is a single fixed grid point for every user.

### Phase 6: Server and docs
- `server_optimizers/naf_plus_server.py`: `NAFPlusServerRuntimeConfig`, `NAFPlusServerSession`, `serve_naf_plus_websocket`, port 8773. One session is one user.
  - `start` may set `weights` (on the simplex, `N` entries) and `n_iter ≤ T`.
  - `observe.y` is **one value per objective**, checked against per-objective `y_range` like the TAF+ and MOBO servers. The server scalarizes with the session's weights.
  - Each suggestion carries `phase` (`naf` for `t < K`, then `mixed`), `lambda_ei` and `p_bar`.
- `configs/server_optimizers/naf_plus_server.yaml` (`run_dir`, `af_name`, `y_range`, `weights_default`, `tau_default`), `server_scripts/run_naf_plus_server.py`, `run_naf_plus_fake_client.py` (`--url`, a double-sphere user, `--novel`), and `tests/naf_plus_server_test.py`.
- README: a new section (§12, renumbering the later ones), §0 scope and roadmap, §5 protocol, §13 results and §14 structure. CLAUDE.md: layout, ports and gotchas.

### Phase 7: Test 2, typing simulation (after the author's input)
- `test_functions/typing_model.py`: a QWERTY geometry parameterized by key width and height, Fitts plus dual-Gaussian Monte Carlo typing, WPM and error normalization, and per-user parameter sampling (training and novel means from App. C).
- A noisy objective: the fixed GP needs a fitted noise term, a deliberate change from NAF's near-zero noise. `f_max` comes from the noise-free expectation, using many Monte Carlo sentences on a fine grid.
- The sentence source is open: Enron Mobile is external, so use a small committed corpus or a bigram model.

## Design stance (decided 2026-11-11)

The owner (the paper's first author) judges OpenBO's details more correct than the paper's. **OpenBO's NAF+ keeps the paper's spirit, not its details**, and is not a reproduction of the CHI 2026 figures. The spirit is three ideas:

1. Train offline on synthetic users drawn from a user model (HOMI).
2. Condition the acquisition function on each user's objective weights.
3. Notice when a user falls outside the model, and fall back to plain BO.

Everything else follows OpenBO's NAF (MetaBO's features including `x`, network, PPO settings, `−log10` reward, fixed GPs, adaptive set), as before. Where the paper is silent or its mechanism is weak, OpenBO decides on merit; the author questions below are now design choices, not blockers. Report results as "NAF+ (OpenBO)" next to v1, never in place of it, and keep the v1 behaviour reachable so the archived results stay reproducible.

**Proposed changes** (none implemented yet; numbers in the order to do them):

1. **No repeats with the argmax** (no retraining). Take the argmax over candidates excluding points already evaluated; on a noise-free objective a repeat gives no information. Keeps a deterministic test protocol and targets v1's main loss (37% repeats). The sampled methods stay as comparisons. A deviation from MetaBO's evaluation, which allows repeats.
2. **Mix policies, not scores** (no retraining; replaces eq. 4 and settles author question 1). With probability `λ_EI` take EI's argmax, otherwise the AF's choice (seeded). Scale-free, no min-max normalization. `λ_EI = 0` is still exactly NAF.
3. **A calibrated novelty test** (no retraining; replaces the mean of p-values against `τ = 0.1`; settles author question 3). Statistic: the joint log-likelihood of the user's observations under the detector (or Fisher's combination of the per-observation p-values). Threshold: set by simulation on training-distribution users at a chosen false-alarm rate (for example 5%), per number of observations. Keep `σ` as the total predictive std.
4. **Novelty as a sweep** (no retraining). Move the centres outward in steps (0.4/0.6 training, then 0.3/0.7, 0.2/0.8, 0.1/0.9) and report regret and fallback rate against distance from the training users. Keep the current shift and scale ranges (±0.1, 0.9–1.1) and state them; this settles author question 2 for OpenBO's purposes.
5. **The user model as a prior** (needs training). The detector already learns the population mean `μ_BNN(x, w)`. Either feed `μ_BNN` (and maybe `σ_BNN`) to the AF as extra per-point inputs, or take the first point from its argmax and let the AF take over from step 2 (cheaper; no retraining). Targets the gap where TAF+ wins evaluations 1–5. An OpenBO extension, in neither paper.
6. **Early iterations first in reporting.** The main measure becomes mean median log10 regret over evaluations 1–5 (and 1–10), with @20 secondary, over at least 3 training seeds with per-step paired tests.

**Item 1 measured (2026-11-11): excluding exact repeats is not enough.** `avoid_repeats` on `NAFSequentialOptimizer`/`NAFPlusSequentialOptimizer` (`exclude` on `AdaptiveSet.maximize`/`sample`, tolerance 1e-9 in the max norm, off by default) and the methods `naf_plus_norepeat`, `naf_plus_no_w_norepeat`, `naf_plus_no_ei_norepeat` (`v1_*_norepeat_*`, same v1 models, 100 users, about 40 s each on 8 workers). Median log10 regret, and the share of evaluations that are exact repeats / within 1e-3 of an earlier point (max norm):

| Test 1.1 | @1 | @2 | @3 | @5 | @10 | @20 | mean 1–5 | mean 1–20 | exact | < 1e-3 |
|---|---|---|---|---|---|---|---|---|---|---|
| NAF+ (argmax, v1) | 0.56 | 0.32 | 0.30 | 0.10 | −1.53 | −3.70 | 0.28 | −1.65 | 39% | 46% |
| NAF+ no repeats | 0.56 | 0.32 | 0.30 | 0.08 | −2.10 | −5.37 | 0.28 | −2.47 | 0% | 27% |
| NAF+ sampled | 0.22 | −0.06 | −0.38 | −0.90 | −4.15 | −5.57 | −0.36 | −3.43 | 24% | 36% |
| w/o weights, no repeats | 0.47 | 0.46 | 0.43 | 0.29 | 0.12 | 0.12 | 0.41 | 0.20 | 0% | 45% |
| EI | 0.13 | 0.13 | 0.13 | 0.02 | −2.41 | −4.80 | 0.11 | −2.51 | 1% | 1% |

Test 1.2 is the same picture (NAF+ no repeats: mean 1–5 0.21, 1–20 −2.48, @20 −5.51; sampled −0.39, −3.34, −5.56; EI 0.15, −2.45, −4.75). The exact-repeat share counted here (39%) differs slightly from Phase 5's 37%, which used another tolerance.

- It helps late: @20 improves from −3.70 to −5.37 (Test 1.1) and −4.57 to −5.51 (1.2), but the first 5 evaluations are unchanged, since repeats start later.
- Paired Wilcoxon (p < 0.05): still worse than EI at most evaluations up to 16 (Test 1.1) and 11 (1.2), and worse than the sampled AF at **every** evaluation on both tests. It beats its no-weights counterpart at evaluations 2–20 on both tests (the no-weights greedy AF still never converges).
- **Cause:** the argmax moves to a local-cloud point next to the excluded one; 27% of evaluations are within 1e-3 of an earlier point. The greedy policy keeps returning to one spot; exact repeats were only the visible part.
- The fallback makes no difference on Test 1.1 (identical to `no_ei_norepeat`) and helps slightly at evaluations 16–20 on Test 1.2.
- **Next:** excluding a neighbourhood (a radius) would be a heuristic on top of a heuristic. Item 2 (mixing policies) and a test-time temperature are the principled follow-ups; the sampled AF stays the best NAF+ variant.

**Greedy stays the default (owner, 2026-11-11).** NAF+ briefly sampled by default (commit `5f53c16`, with the methods renamed); that was reverted the same day, so `naf_plus` is the argmax (MetaBO's protocol) and the `_sample` methods sample. The one new result of that commit is kept as `naf_plus_no_ei_sample` (rerunning it reproduces the file exactly).

**The fallback under sampling** (`v1_naf_plus_no_ei_sample_*`, which v1 lacked). Mean median log10 regret 1–5 / 1–20:

| | Test 1.1 | Test 1.2 |
|---|---|---|
| `naf_plus_sample` (fallback) | −0.36 / −3.43 | −0.39 / −3.34 |
| `naf_plus_no_ei_sample` | −0.36 / −3.44 | −0.41 / −3.36 |

The fallback fired for 2 of 100 users on Test 1.1 and 35 of 100 on Test 1.2. It is never significantly better (paired Wilcoxon, p < 0.05), and on Test 1.2 it is **worse at evaluations 5 and 10** (@5: −0.93 against −1.03): the AF copes with these novel users, and the eq. 4 mix turns a sampled step into a greedy one. Stance items 2–4 are the way to give the fallback something to do.

**Does the weight conditioning work? (2026-11-11, `scripts/diagnose_naf_plus_weights.py`, outputs in `test_results/naf_plus/weights/v1_*`).** The 100 Test 1.1 users, each under all 11 weights `w = (w1, 1 − w1)`, `w1 = 0, 0.1, …, 1` (`w1` weights the objective centred at (0.4, 0.4)); 1,100 runs per method, greedy, no EI fallback: `naf_plus` (true weights), `naf_plus_no_w`, and plain EI (fixed GP, one Sobol point). Randomness seeded per user, the same for every weight. 905 s on 8 workers.

- **The learned prior is nearly flat and barely moves.** At `t = 0` (no data, so the AF sees only `x` and `w`) the highest grid probability is 1.09–1.22× uniform. The mass sits in the upper-right half (`x1 + x2` above about 1); its centre moves from (0.53, 0.54) at `w1 = 0` to (0.51, 0.52) at `w1 = 1`, while the average user's optimum moves from (0.6, 0.6) to (0.4, 0.4). The greedy first point is on the top edge, moving from (0.3, 1) to (0, 1) as `w1` grows: a probe, not a guess at the optimum (MetaBO's reward gives early steps little weight; NAF open question 11).
- **Behaviour does depend on the weights, but unevenly.** Per `w1`, median final log10 regret / share of users below −3:

  | `w1` | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | NAF+ | 0.15 / 28% | 0.08 / 37% | −2.67 / 49% | −2.56 / 46% | −4.43 / 59% | −4.49 / 63% | −4.31 / 60% | −4.82 / 69% | −5.08 / 80% | −4.95 / 82% | −4.95 / 76% |
  | w/o weights | 0.10 / 2% | 0.09 / 5% | 0.11 / 8% | 0.13 / 12% | 0.19 / 11% | 0.21 / 14% | 0.21 / 16% | 0.22 / 8% | 0.20 / 4% | 0.19 / 6% | 0.22 / 1% |
  | EI | −4.74 / 100% | −4.83 / 100% | −4.79 / 100% | −4.77 / 100% | −4.88 / 100% | −4.74 / 97% | −4.83 / 99% | −4.86 / 100% | −4.89 / 100% | −4.79 / 100% | −4.91 / 100% |

  NAF+ is bimodal: a user is either solved (median below EI at `w1 ≥ 0.7`) or stuck. At `w1 = 0` it spends 690 of 2000 evaluations at one top-edge point, (0.27, 1.0). The two objectives are symmetric, so this asymmetry is the policy's, not the family's (one training seed).
- **Against EI:** NAF+ has higher regret at iterations 1–3 and 6–20 (paired Wilcoxon over users, p < 0.05; lower only at 4 on the extreme weights). Mean median log10 regret over 1–5 is 0.05–0.27 for NAF+ against 0.02–0.13 for EI at every weight. Per point, NAF+ lands closer to the optimum than EI at iterations 2–5 and 10–20 (EI keeps exploring; NAF+ stays near what it found), but its greedy repeats stop it from refining.
- **Against no weights:** NAF+ is better at iterations 2–20 in every weight group. **Caveat:** the greedy no-weights AF fails at every weight (1–16% solved; sampled, the same AF reaches −5.42 at @20), so this comparison mostly shows that greedy no-weights is broken, not how much the weights add. The clean evidence that the weights are used is that NAF+'s behaviour changes with them; the wrong-weights control (AF told `1 − w`) was proposed and dropped by the owner.
- **Conclusion:** the AF reads the weights, but it has not learned a weight-dependent prior worth using early: the first steps are near-uniform probes, and EI is better at every weight in iterations 1–5. Item 5 (the user model as a prior) targets exactly this.

**Why the weights barely matter on this family.** With equal steepness, `Σ w_k f_k = S·(1 − 8‖x + δ − c̄‖² − 0.64·w1·w2)` with `c̄ = Σ w_k c_k`: every task is one round bowl whose peak lies on the diagonal between (0.4, 0.4) and (0.6, 0.6), moved by the user's shift. The weights move the peak by at most 0.28, about as much as a user's own shift (±0.1 per coordinate), so they tell the AF little that one or two observations don't.

**Decision (owner, 2026-11-11): a wide family, `double_sphere_wide`.** Training centres (0.2, 0.2) and (0.8, 0.8), so the weights move the optimum 0.85 along the diagonal; with shifts of ±0.1 every optimum stays in [0.1, 0.9]² (never clipped). Novel users (Test 1.2) use the other diagonal, (0.2, 0.8) and (0.8, 0.2). Shift and scale ranges unchanged. Fixed GP from the same selection procedure (`select_family_hyperparameters(sample_weighted_tasks(60, seed=11, centres=WIDE_TRAINING_CENTRES), 2, seed=0)`, which reproduces v1's 0.35/1.41 on the old centres): **lengthscale 0.55, variance 6.07** (values range from about −11.3 to 1). Code: `DOUBLE_SPHERE_FAMILIES` (name → training and novel centres) in `naf_plus_families.py`, `FAMILY_GP_HYPERPARAMETERS` and `NAFPlusConfig.centres(novel)` in `optimizers/naf_plus.py`; `--family double_sphere_wide` in `train_naf_plus.py`. The runner takes the family from the run's `naf_plus_config.json`.

Training `double_sphere_wide_v1` (with weights) and `double_sphere_wide_no_w_v1` (`--no-weights`, detector copied) exactly as v1: 150 iterations, seed 0, 4 workers and 4 update threads each, in parallel. Only the family changes, so differences come from the family. Then Test 1.1 and the weight diagnostic on the new runs.

**Wide-family results (2026-11-11, test id `wide_v1`).** Training finished in about 6.5 h with one pause, resumed with `--resume`. Best-return checkpoints: with weights 63.7 at iteration 126 (training final log10 regret −4.69); without weights 64.8 at iteration 120 (−5.61). The no-weights run learned this time; in v1 it collapsed. Under training's sampled rollouts it matches or beats the weighted run, so training returns did not show the weights helping.

Test 1.1: 100 users with random weights, median log10 simple regret. The wide family's values span about −11 to 1 (old family: about −2 to 1), so compare methods within a family, not across.

| evaluation | 1 | 2 | 3 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|
| `naf_plus` (greedy) | 0.21 | 0.14 | −0.01 | −0.43 | −2.33 | −4.25 | −4.30 |
| `naf_plus_no_w` | 0.67 | 0.40 | 0.08 | −0.06 | −0.22 | −0.25 | −0.29 |
| `naf_plus_norepeat` | 0.21 | 0.14 | −0.04 | −0.53 | −2.52 | −4.32 | −4.35 |
| `naf_plus_sample` | 0.21 | −0.22 | −0.33 | −0.85 | −3.82 | −4.59 | −4.60 |
| `naf_plus_no_w_sample` | 0.25 | −0.06 | −0.16 | −0.89 | −3.28 | −4.91 | −5.34 |
| `ei` | 0.24 | 0.21 | 0.18 | −0.10 | −2.43 | −4.66 | −5.09 |
| `taf_plus_15` | −1.24 | −1.24 | −1.29 | −1.30 | −1.33 | −1.37 | −1.37 |
| `random` | 0.16 | −0.13 | −0.25 | −0.51 | −0.86 | −1.02 | −1.09 |

Paired Wilcoxon, p < 0.05:
- **Greedy NAF+ against EI:** better at evaluation 5; tied at 1–3 and 10; worse at 15–20. In v1 it was worse at 1–3 and 10–20.
- **Greedy NAF+ against `naf_plus_no_w`:** better at every evaluation.
- **Sampled NAF+ against EI:** better at evaluations 2–10, worse at 20.
- **Sampled NAF+ against `naf_plus_no_w_sample`:** better at 2 and 10, worse at 15 and 20.

The fallback is identical to no-EI again, so it does not fire. TAF+ wins evaluations 1–5 and then stalls, as in v1.

Weight diagnostic (`test_results/naf_plus/weights/wide_v1_*`, 1,027 s). Per `w1`: median final log10 regret / share of users below −3.

| `w1` | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NAF+ | −3.69 / 65% | −4.08 / 97% | −4.50 / 100% | −4.70 / 100% | −4.28 / 100% | −4.29 / 100% | −4.21 / 100% | −4.35 / 100% | −4.48 / 100% | −4.08 / 90% | −4.08 / 100% |
| w/o weights | −0.12 / 0% | 0.09 / 0% | 0.07 / 16% | −0.14 / 25% | −2.71 / 43% | −4.46 / 65% | −3.79 / 53% | −0.11 / 27% | −0.14 / 6% | −0.30 / 1% | −0.54 / 0% |
| EI | −5.11 / 100% | … | | | | −5.17 / 100% | | | | … | −5.09 / 100% |

(EI is between −5.09 and −5.21 at every weight.)

- **The AF now uses the weights.** The no-weights AF solves users only near `w1 = 0.5`, where the optimum is near the average one, and fails at the extremes: the expected signature of ignoring the weights. NAF+ solves 90–100% at every weight except `w1 = 0` (65%), so the v1 bimodality is mostly gone.
- **The greedy first point follows the weights:** (1, 1) at `w1 = 0`, where the optimum is near (0.8, 0.8); (0.43, 0) at 0.5; (0, 0) at `w1 = 1`, where the optimum is near (0.2, 0.2). It points in the right direction but lands in the corner, not at the optimum: still a probe.
- **The prior is still flat:** at most about 1.1–1.6× uniform. Its centre of mass moves only from (0.56, 0.52) to (0.47, 0.45), while the optimum moves from (0.8, 0.8) to (0.2, 0.2).
- **Repeats:** 5.3 repeated evaluations per 20-step greedy run (v1: 6.8). EI still ends lower at every weight, by about 0.5–1.4.

**Conclusion.** With weights that matter, the conditioning is learned and pays off: greedy NAF+ moves from losing to EI early to tying or beating it up to evaluation 10, and it clearly beats the AF without weights. The early advantage is still small, because the `t = 0` prior is nearly uniform and the first point is a corner probe; that is item 5's target (and NAF open question 11's: the reward gives early steps little credit). EI keeps the late lead. One training seed per run.

**Items 2 and 3 on the wide family (2026-11-12, test id `wide_v1`).** Decided with the owner before running:
- Test 1.2 keeps the full weight range and is reported against the novelty distance `|0.6 − 1.2 w1|`: 0 at `w1 = 0.5`, where a novel user is the same function as a training user, and up to 0.6 at the extremes. Bins: near < 0.2 (38 users), mid 0.2–0.4 (34), far ≥ 0.4 (28).
- `λ_EI` is graded (eq. 5's shape), with `τ = 0.05` on the calibrated p-value, the first check before evaluation 4, and EI's plain argmax when the coin picks EI.

Code:
- `novelty_z_scores`, `novelty_statistic` and `calibrated_p_value` in `acquisition/naf_plus.py`.
- `NoveltyCalibration`, `calibrate_novelty` and `CALIBRATION_SEED = 5003` in `optimizers/naf_plus.py`; `scripts/calibrate_naf_plus_novelty.py` writes `novelty_calibration.json`.
- The optimizer options `novelty_test` (`mean_p`/`calibrated`) and `mix` (`score`/`policy`, with `mix_rng`); `NAFPlusStep` gains `novelty_p` and `source`.
- The methods `naf_plus_mix` (old test, policy mix), `naf_plus_cal` and `naf_plus_cal_sample` (calibrated test, policy mix).
- The diagnostic's `--novel` option.

The defaults keep v1 bit for bit.

Calibration: 500 training-like users per policy, 909 s on 6 workers. The 95% quantile of `S_n` (greedy) is 2.7 at n = 1, 6.5 at 3, 10.2 at 5, 20.0 at 10 and 36.3 at 20. For a calibrated Gaussian these would be χ²_n quantiles: 3.8, 7.8, 11.1, 18.3 and 31.4. So the detector is roughly calibrated, slightly wide early and narrow late.

**The existing fallback already works on this family.** Test 1.2 baseline, median log10 regret at evaluations 3 / 5 / 10 / 20, with the share of users for whom `λ_EI > 0` at some step:

| Test 1.2 | all users | far (≥ 0.4) | near (< 0.2) |
|---|---|---|---|
| `naf_plus_no_ei` | 0.24 / −0.23 / −2.13 / −4.21 | −0.12 / −0.31 / −0.38 / −0.68 | 0.26 / −0.00 / −2.92 / −4.33 |
| `naf_plus` (mean p, eq. 4) | 0.24 / −0.19 / −2.77 / **−5.31**, fires 65% | −0.12 / −0.31 / −2.64 / **−5.49**, 100% | 0.26 / −0.00 / −2.85 / −4.42, 18% |
| `naf_plus_mix` (mean p, policy) | 0.24 / −0.18 / −2.49 / −4.55, 65% | −0.12 / −0.33 / −2.51 / −5.04, 100% | 0.26 / 0.08 / −3.02 / −4.38, 18% |
| `naf_plus_cal` (calibrated, policy) | 0.24 / 0.02 / −2.66 / −5.04, 82% | −0.12 / −0.31 / −2.59 / −5.07, 100% | 0.26 / 0.14 / −2.96 / −4.46, 55% |
| `naf_plus_no_ei_sample` | **−0.31 / −0.99 / −3.81** / −4.44 | **−0.45 / −1.11 / −3.99** / −4.71 | −0.29 / −0.83 / −3.69 / −4.32 |
| `naf_plus_sample` | −0.31 / −0.63 / −3.08 / −5.19, 63% | −0.45 / −0.85 / −3.00 / −5.45, 96% | −0.29 / −0.61 / −3.14 / −4.34, 16% |
| `naf_plus_cal_sample` | −0.31 / −0.56 / −2.56 / −4.96, 80% | −0.45 / −0.64 / −2.39 / −5.07, 100% | −0.29 / −0.57 / −2.99 / −4.42, 50% |
| `ei` | 0.02 / −0.04 / −2.43 / −5.21 | 0.05 / −0.20 / −2.41 / −5.07 | 0.00 / −0.00 / −2.58 / −5.36 |

Test 1.1 (false alarms), share fired: `naf_plus` 5%, `naf_plus_mix` 5%, `naf_plus_cal` 8%, `naf_plus_sample` 2%, `naf_plus_cal_sample` 10%. No method there differs significantly from its no-EI counterpart at any evaluation (paired Wilcoxon, p < 0.05).

Paired Wilcoxon on Test 1.2 (p < 0.05):
- **`naf_plus` beats `naf_plus_no_ei`** at evaluations 5–20. Greedy, the AF alone is lost on far users (−0.68 at @20).
- **`naf_plus` beats EI** at 5. EI is better at 3; they tie at 10–20.
- **`naf_plus` beats `naf_plus_mix`** at 15–20 and **beats `naf_plus_cal`** at 5, 15 and 20.
- `naf_plus_sample` is worse than `naf_plus_no_ei_sample` at 3–5 and better at 10–20.

Conclusions:
- **Item 2 (policy mix) is refuted here.** The blended score of eq. 4, a compromise point every step, beats a coin between the two policies at the same `λ_EI`. On novel users the AF's own points are the weak ones, and the coin still takes them `1 − λ` of the time. The policy mix stays an option, not the default.
- **Item 3 (calibrated test) is not needed here, and as built it fires too often.** At the same `τ` per check it flags more users (82% against 65% on Test 1.2), including 55% of near users, who need no EI and pay for it at @5 (0.14 against −0.00). On Test 1.1, repeated checks raise the share of users ever flagged above `τ` (8–10%). The old mean-p test with `τ = 0.1` happens to sit at a good operating point on this family: 5% on Test 1.1, 18% near, 88% mid and 100% far. The calibrated test stays an option. Its value is a known per-check false-alarm rate on families where the old test is off, as on the old family, where it almost never fired.
- **The default stays `naf_plus`** (mean p, eq. 4). On the wide family it is the best greedy fallback, and on Test 1.2 it does the job item 3 was meant for.
- **The diagnostic on novel users** (`--novel --methods naf_plus ei naf_plus_cal`, outputs `test_results/naf_plus/weights/wide_v1_novel_*`, 860 s) covers every Test 1.2 user under all 11 weights. Distance runs 0.6, 0.48, …, 0 at `w1 = 0.5`, … 0.6.
  - **The AF alone** (greedy, no EI) solves 94–100% of users at `w1 = 0.2–0.8`. It fails at the extremes: 1% and 15% at `w1 = 0` and 0.1, 33% at 0.9, 71% at 1.
  - **`naf_plus_cal`** solves 100% at every weight; median final log10 regret is −4.3 to −5.2 against EI's −5.1 to −5.2.
  - **How often `naf_plus_cal` fires** by `w1` from 0 to 1: 100, 100, 100, 100, 65, **13**, 60, 100, 100, 100, 100%. At `w1 = 0.5` these users are training users, so 13% is the share of typical users ever flagged over 17 checks (5% per check). At distance 0.12 (`w1 = 0.4`, 0.6) it already fires for 60–65%.
  - **Paired over all weights:** `naf_plus_cal` beats the AF alone at evaluations 7–20 and loses at 4–6, the steps right after the first check. It beats EI only at 8–10, and loses at 1–4, 6–7 and 16–20.
- **Open: sampling against the fallback.** The sampled AF alone is the best method on Test 1.2 up to evaluation 10, even on far users, and the fallback costs it early. Stopping the AF from sampling forever (it stalls at about −4.4) matters more than when EI comes in.

**Double Branin considered (2026-11-11), decision deferred until the wide run is in.** The owner proposed Branin plus a flipped Branin. Measured on a 301² grid with `normalized_branin`: the regret of a weight-blind choice (always the `w = (0.5, 0.5)` argmax), median / worst over the 11 weights, is 0.01 / 0.05 for the flip in both axes, 0.01 / 0.04 left-right, 0.04 / 0.20 upside-down and 0.01 / 0.21 for a 90° rotation, against **0.51 / 1.44** for the wide double sphere. Branin's flat valleys overlap, so every pairing has a compromise region nearly optimal for all weights. The flip in both axes also lands two flipped optima almost on Branin's own ((0.96, 0.17) against (0.88, 0.18); (0.12, 0.82) against (0.04, 0.84)), so the optimum sits in one of two corners and jumps between them around `w1 = 0.55`. Good later as a harder, multimodal test (no analytic optimum; NAF on Branin needed about 1500 iterations to beat EI clearly), with a pairing chosen so that the weights matter.

Unchanged: the weight-conditioned AF and value net, the double-sphere family, the detector network (2 × 256, dropout 0.2, grid data; PPO-pair data stays an option), `K = 3` as the default (sweepable), and the Phase 6 server plan. Phase 7 (typing) still needs a simulator, which OpenBO may now build on its own terms.

## Handoff (2026-10-12)

NAF+ is paused and is now the last roadmap item. The owner chose to push on with the rest of OpenBO, and a student will run the thorough benchmark. State at handoff:

**What exists**
- Code:
  - family: `test_functions/naf_plus_families.py`;
  - acquisition: `acquisition/naf_plus.py`;
  - weight-conditioned AF: `NAFConfig.n_weights` in `optimizers/naf.py`;
  - detector, optimizer and training: `optimizers/naf_plus.py`;
  - runner: `benchmarks/naf_plus_runner.py`;
  - scripts: `scripts/{train,run,plot}_naf_plus*.py`;
  - tests: `tests/naf_plus_*_test.py`.
- Trained runs, 150 PPO iterations each, `best_af` is the checkpoint used:
  - `meta-bo-training/naf_plus/double_sphere_v1` (with weights; best return 73.2 at iteration 115);
  - `double_sphere_no_w_v1` (best return 66.8 at iteration 92; collapsed at 128).
- Results: `test_results/naf_plus/trajectories/v1_*` (8 methods × 2 tests × 100 users) and `plots/v1_*`. The numbers are in Phase 5 above.

**Reproduce v1**, from the repo root:
```bash
uv run python scripts/train_naf_plus.py --run-id double_sphere_v1 --n-iterations 150 --workers 4 --update-threads 4
uv run python scripts/train_naf_plus.py --run-id double_sphere_no_w_v1 --no-weights \
  --detector-from meta-bo-training/naf_plus/double_sphere_v1 --n-iterations 150 --workers 4 --update-threads 4
for t in double_sphere double_sphere_novel; do
  for m in naf_plus naf_plus_no_w naf_plus_no_ei naf_plus_sample naf_plus_no_w_sample ei taf_plus_15 random; do
    uv run python scripts/run_naf_plus_benchmark.py --method $m --test $t --n-tasks 100 --test-id v1 --workers 8 \
      --run-dir meta-bo-training/naf_plus/double_sphere_v1 --run-dir-no-w meta-bo-training/naf_plus/double_sphere_no_w_v1
  done
done
uv run python scripts/plot_naf_plus_results.py --test-id v1
```
The detector takes about 16 minutes. Training takes about 90–120 s per iteration per run with the two runs in parallel on 8 cores, so about 4–5 h for 150. Each NAF+ benchmark method takes under a minute; `taf_plus_15` takes about 13 minutes.

**Main findings so far** (one seed; provisional family ranges)
1. With the argmax (MetaBO's and the paper's protocol), the AF loses to EI at every budget. It repeats points (37%), and its policy entropy is still about 2.6.
2. Sampled from its softmax, the same AF beats EI at evaluations 2–20 (Wilcoxon p < 0.05), with a median regret of about −5.6 at @20 against −4.8.
3. The weights help significantly but mostly late (evaluations 10–20, sampled). The greedy no-weights AF never converges.
4. The fallback rarely fires. p̄ separates the two groups: a median of about 0.47 for training-like users against 0.19–0.28 for novel users. But only 15–25% of novel users go below τ = 0.1, and the mean λ_EI is 0.05. The novel users barely hurt the sampled AF, so there is little to rescue.
5. TAF+ (15 users) wins evaluations 1–5 on both tests.

**Next work, in suggested order**
1. Phase 6: the server (`naf_plus_server.py`, port 8773, see above) and the README section.
2. Several training seeds (at least 3), with significance tests per step against EI and against the no-weights ablation.
3. A sampled method with the fallback off (`naf_plus_no_ei_sample`), to isolate the fallback's effect once repeats are not the issue.
4. A harder novel test (for example centres 0.2/0.8), to check whether the fallback fires and helps when the AF really struggles. It needs no retraining.
5. The paper's double-sphere ranges (author question 2) and `σ_BNN` (question 3), then refit the detector and retrain.
6. Longer training (300+ iterations), to see whether the argmax policy sharpens. On Branin, going from 193 to 313 iterations still left 26% repeats.
7. Early iterations (NAF open question 11): the first point ignores the population, and MetaBO's reward gives early steps little weight.
8. Phase 7, the typing simulation (needs author question 6).

## Questions for the author

Since 2026-11-11 (Design stance) these no longer block anything: questions 1–3 are settled by OpenBO's own choices (stance items 2–4), 4 stays an option, and 5 is moot (OpenBO keeps `x`). The answers are still worth recording for comparison with the paper.

1. **Eq. 4 scales.** Is `naf(x)` the softmax probability, the raw logit, or something normalized? Was EI normalized before mixing? Raw EI and logits differ by orders of magnitude, which makes `λ_EI` meaningless without a common scale.
2. **Double-sphere ranges.** What were `shift_range` and `scale_range` in App. A.2/B.1? Was the scale applied to the weighted sum only (as B.1 reads)?
3. **`p̄` window.** Does the running mean include iterations 1–3 (before the mixing starts)? Is `σ_BNN` the epistemic std only, or epistemic plus noise?
4. **Detector data in the synthetic tests.** Did they use all PPO training pairs (App. A.3) or a grid, as in the user study (Sec. 5.6)?
5. **naf's `x` input.** Eq. 2 omits `x`; MetaBO includes it. Was that a simplification in the text, or did naf really not see `x`? (OpenBO follows MetaBO either way, but this matters for comparing results.)
6. **Typing simulator.** Is the simulator code (keyboard geometry, sentence set) available to reimplement from? What units were `W` and `H` in Test 2 (touchscreen) as opposed to the VR study?

## Out of scope (first pass)
- The VR user study and Unity client (roadmap item 5).
- More than 2 objectives in the benchmarks (the code supports `N` weights).
- Continual updating of naf with real users (Discussion, "Development of other HOMI methods").

## Open questions
1. Score normalization in eq. 4 (author question 1). The default is min-max per candidate set.
2. Training length. MetaBO ran about 2000 iterations and the CHI paper 80k steps; decide from the Phase 5 curve.
3. Does the fixed-hyperparameter GP fitted at `w = (0.5, 0.5)` suit `w` near 0 or 1? Phase 1 checks this.
4. ✅ *Answered 2026-10-12 (Phase 1):* yes. Novel-centre users would select lengthscale 0.45 with variance 2.77 (training family: 0.35, 1.41), so Test 1.2 users are out of distribution for the fixed GP as well as for the policy.
5. **Can the detector see Test 1.2's novel users?** Not at the provisional ranges (Phase 3 measurement: 6–18% flagged). It depends on the paper's real `shift_range`/`scale_range` (author question 2); a narrower population makes novel users stand out. Decide before Phase 5: use the author's ranges, narrow the defaults, or report Test 1.2 as it comes out.
