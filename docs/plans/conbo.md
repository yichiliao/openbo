# Plan: Continual BO across users (ConBO)

Status: Phases 1–5 done; Phase 6 has Test 1 with 5 seeds; Tests 2–3 and the ablations are pending. Last updated 2026-10-10.

Reference: Y.-C. Liao, P. Streli, Z. Li, C. Gebhardt and C. Holz, [*Continual Human-in-the-Loop Optimization*](https://dl.acm.org/doi/10.1145/3706598.3713603), CHI 2025 ([arXiv:2503.05405](https://arxiv.org/abs/2503.05405), [project page](https://siplab.org/projects/Continual_Human-in-the-Loop_Optimization)). The main paper and appendix have been read. No reference code is available, so the paper is the only reference (see Open question 1).

## Decisions (proposed, to confirm before Phase 1)

| Topic | Decision |
| --- | --- |
| Scope of first pass | Reproduce the paper's simulated tests (Branin, McCormick, revisiting users) in 2D. Single-objective, sequential (q = 1), synthetic users only. |
| Population model | A torch MC-dropout BNN with a mean head and a log-variance head, as in the paper. Torch is already a dependency (BoTorch). |
| User model | Reuse `GPScratch` (Matérn-5/2) and the analytic EI in `acquisition/ei.py`. No BoTorch inside ConBO. |
| Memory format | Store each finished user in the TAF layout: `gp_states/<user>.json` + `trajectories/<user>.json`. Replay refits the GPs from these files, like `bo_taf._load_source_surrogates`. A ConBO memory directory can then also serve as a TAF source directory, which gives a fair TAF baseline for free. |
| State across users | Split into a long-lived `ConBOPopulation` (BNN, stored users, user counter; save/load to a directory) and a per-user ask/tell `ConBOSequentialOptimizer`. This is the first OpenBO method whose state outlives one task. |
| Candidate search | Grid in 2D, as in the paper (40 × 40). A Sobol pool for d > 2, reusing the pool helpers in `bo_scratch`. |
| Output scale for λ | *Revised in Phase 3.* λ is in raw output units, as in the paper, because no scale-free form reproduces both of the paper's settings. `standardized_variance_threshold` converts a threshold given in standardized units for problems without a tuned λ, and `ReplayTargets.kept_fraction` reports how much each user keeps so λ can be tuned. |
| Benchmark runner | A separate `scripts/run_conbo_benchmark.py`, the same way MOBO has its own. The existing family runner treats tasks as independent, but here the order of users matters. |

## Paper summary

**Setting (CHiLO)**
- Users arrive one after another. Each user has their own objective `f_u(x)` over the same design space, and their own small evaluation budget.
- The optimizer should get faster for later users by learning from earlier ones, without storing their raw data.
- The challenges it names are catastrophic forgetting, the stability–plasticity trade-off, and uneven data across users.

**Population model (BNN)**
- 3 fully connected layers of 100 units with ReLU, then dropout 0.1 after the last hidden layer. Dropout stays on at prediction time.
- Two linear heads: mean `μ̂` and log-variance `log σ̂²`, trained with Gaussian negative log-likelihood.
- Predictive distribution from `M` stochastic forward passes (Kendall & Gal, 2017): the mean is `mean_m μ̂_m`, and the variance is `mean_m σ̂²_m + var_m μ̂_m`.

**User model**
- A fresh single-task GP per user with a Matérn-5/2 kernel and homoscedastic noise, fit only to that user's observations.
- When a user has no random initial points (`r_u = 0`), their GP starts with a zero mean and the population model guides the first iterations alone.

**Acquisition**
- EI from each model, mixed by a weight that moves from the population model to the user model:
  - `a_t(x) = w_t · EI_pop(x) + (1 − w_t) · EI_user(x)`
  - `w_t = 1` for `t ≤ α₁`, `w_t = 1 − (t − α₁)·α₂` for `α₁ < t ≤ α₁ + 1/α₂`, and `w_t = 0` after that.
  - The user study uses `α₁ = 5`, `α₂ = 0.2`, so the weight reaches 0 at `t = 10`.
- Argmax over a 40 × 40 grid of the normalized 2D space.

**Fewer random starts for later users**
- `r_u = max(0, r₀ − (u − 1)·d_r)` random initial points for user `u` (counting from 1).
- Study: `r₀ = 6`, `d_r = 2`. Branin test: `r₀ = 16`, `d_r = 5`. McCormick test: `r₀ = 3`, `d_r = 3`.

**Generative replay**
- After user `u` finishes, store only their fitted GP.
- Sample replay inputs: a grid plus random points (study: 20 × 20 + 100; simulations: 30 × 30 + 50).
- Query every stored GP `i ≤ u` for `μ_{i,s}` and `σ²_{i,s}` at each replay input `s`.
- **Variance filter:** keep the pair `(i, s)` only if `σ²_{i,s} < λ`. A user whose variance is above `λ` everywhere drops out of that round.
- Retrain the BNN on the filtered targets (eq. 7):
  `ℒ = (1/|S|) Σ_s [ (μ̂_s − mean_{i∈U_s} μ_{i,s})² + (σ̂²_s − mean_{i∈U_s} σ²_{i,s})² ]`
- **Online:** during a session, the BNN is also updated by Gaussian NLL on each new `(x, y)` for a few epochs.

**Training budgets**
- Study: 800 epochs for population training, 20 epochs per new observation.
- Simulations: 1,200 epochs and 15 epochs.
- λ = 5 (study), 50 (Branin), 10 (McCormick).

**Simulated benchmark (Appendix A)**
- Each user is a shifted and scaled copy of a base function, flipped to maximization with inputs in `[0, 1]²`.
  - Input shift: uniform in `±shift_range/2` per input. Output scale: uniform in `1 ± scale_range/2`. No input scaling, and no added noise.
- **Test 1:** Branin, 15 users, `shift_range = 0.3`, `scale_range = 0.2`, 30 iterations per user.
- **Test 2:** McCormick, 15 users, `shift_range = 0.5`, `scale_range = 0.2`, 7 iterations per user.
- **Test 3 (forgetting):** Branin, 10 users, `shift_range = 0.4`, `scale_range = 0.4`. Earlier users are revisited after later ones.
- **Metric:** per-iteration regret against each user's optimum found by grid search, plus computation time per iteration. Error bars are ±1 std. The number of repeats is not stated.
- **Compared methods:** ConBO, ConBO without the GP, ConBO without the variance filter, BNN without replay, BNN with direct replay (stored raw data), single GP over all data, TAF, and standard BO.

## Mapping onto OpenBO

| ConBO piece | Already in OpenBO | To build |
| --- | --- | --- |
| Synthetic users | `generate_variants(base, n, max_input_shift=shift_range/2, max_input_scale_delta=0, max_output_scale_delta=scale_range/2)` and `make_variant_function_spec` | `mccormick` in `synthetic.py`, `registry.py` and `TASK_BOUNDS`. A grid-search optimum per user (the shift can move the optimum out of the domain). |
| User GP and EI | `GPScratch(kernel_type="matern52")`, `expected_improvement_maximization` | A zero-mean start when the user has no data yet. |
| Memory of past users | TAF artifact layout and its loader | Reuse as-is. |
| Population BNN | — | `models/bnn.py` |
| Replay | — | Replay sampling, variance filter and eq. 7 loss |
| Mixed acquisition | — | `acquisition/conbo.py` |
| Optimizer | Ask/tell pattern | `optimizers/conbo.py` |
| Benchmarks | Plot helpers in the existing scripts | `scripts/run_conbo_benchmark.py`, `scripts/plot_conbo_results.py` |

## Implementation phases

Each phase ends with a `tests/<topic>_test.py` file and a full `uv run pytest` run.

### Phase 1: Users and the McCormick function ✅ done

**As built**
- `mccormick` in `synthetic.py`, `registry.py` and `TASK_BOUNDS`. Inputs map from `[0, 1]²` to `x1 ∈ [−1.5, 4]`, `x2 ∈ [−3, 4]`. The maximum has a closed form: `MCCORMICK_MAXIMUM = √3/2 + π/3 ≈ 1.91322`, at `x1 − x2 = 1`, `x1 + x2 = −2π/3` (about `(0.17, 0.21)` in unit coordinates).
- In `families.py`:
  - `generate_user_variants(base, n_users, shift_range, scale_range, seed)`: calls `generate_variants` with `max_input_shift = shift_range/2`, no input scaling and `max_output_scale_delta = scale_range/2`, so the variants can be saved with a benchmark run.
  - `make_user_sequence(base, variants, prefix="user")`: builds specs named `user_000`, ….
  - `estimate_optimum(objective, dim)`: a 201-per-axis grid for d ≤ 3 (a Sobol set above), then L-BFGS-B polish of the best 5 points. It recovers the known optima to about 1e-7 (Hartmann6 to 1e-5).
- **Why each user's optimum is estimated:** variants clip `x + shift` to `[0, 1]`. When a shift pushes the base optimizer outside the domain, the user's optimum is lower than `output_scale · base optimum`, which is what `FunctionSpec.optimum` would otherwise claim. This happens for McCormick, whose optimizer sits near the domain corner. Building 15 users takes about 0.25 s.
- Tests: `tests/conbo_users_test.py` (14 tests).

**Original plan**
- Add `mccormick` (2D, maximized, inputs mapped from `[0, 1]²` to its native domain) with its known optimum.
- A `make_user_sequence(base, n_users, shift_range, scale_range, seed)` helper on top of `generate_variants`, returning specs plus a grid-search optimum per user.
- Tests: the McCormick optimum, the variant ranges, reproducibility from the seed.

### Phase 2: BNN population model (`models/bnn.py`) ✅ done

**As built**
- `BNNConfig` (frozen): `input_dim`, `hidden_units=100`, `n_hidden_layers=3`, `dropout=0.1`, `learning_rate=1e-3` (Adam), `batch_size=None` (full batch), `n_predict_samples=50`, `seed`. The paper gives the architecture; the rest are our defaults (open question 2).
- `PopulationBNN`:
  - `fit_nll(x, y, epochs)`: Gaussian NLL.
  - `fit_replay(x, mean_target, var_target, epochs, renormalize=True)`: eq. 7.
  - `predict(x) -> (mean, var)`: mean of the sampled means, and mean head variance plus variance of the sampled means.
  - `reset_parameters()`, `save`/`load` (one `.pt` file holding the config, weights, Adam state, normalization and dropout RNG state).
  - `fit_*` return the loss per epoch.
- **Determinism:** torch `float64`. Weights are initialized inside `fork_rng` from `config.seed`. Dropout masks and minibatch order come from the model's own `torch.Generator`, so the global torch RNG is untouched. `predict` reseeds its dropout masks on every call, so the same inputs always give the same output, which keeps acquisition values stable within a step.
- **Output standardization:** set from the first data the model sees, then frozen, so online NLL updates and replay share one scale. `fit_replay` resets it by default, since replay is a full retrain on all users. This is where the proposed "λ relative to the standardized output" will hook in (Phase 3).
- The log-variance is clamped to `[log 1e-6, log 1e3]` in standardized units.

**Findings**
- **NLL fits work.** A noisy sine is matched to about 0.1 where there is data. With input-dependent noise of 0.1 and 0.6 std, the learned variances are about 0.009 and 0.38–0.43 (true values 0.01 and 0.36), across 5 seeds.
- **MC dropout is mildly overconfident away from the data.** The variance grows only 3–8× (seeds 1–5) between the data region and far outside it, because dropout acts on one layer at rate 0.1. This is the paper's model, so we keep it; it means the user GP has to supply most of the exploration once `w_t` falls.
- **Eq. 7 matches large variances well and small ones loosely.** In standardized units the replay variances are about 0.005–0.1, so their squared error is tiny next to the mean term. Large target variances are matched to within about 10%; the smallest come out several times too large.
  - A log-variance loss was tried and dropped: the variance errors were no better, and the mean error grew from 1.6 to 8.2 (on a ±100 target).
  - If small replayed variances turn out to matter in Phase 6, the next things to try are weighting the variance term or more epochs.
- **Cost on CPU:** a replay fit at the paper's size (950 points, 1,200 epochs, 2D) takes about 8 s. Predicting a 40 × 40 grid with 50 dropout samples takes about 0.2 s.
- Tests: `tests/bnn_test.py` (12 tests, about 13 s).

**Original plan**
- `PopulationBNN`: an MC-dropout MLP with mean and log-variance heads. `fit_nll(x, y, epochs)`, `fit_replay(x, mu_target, var_target, epochs)` (eq. 7), `predict(x, n_samples) -> (mean, var)`, and `state_dict`-based save/load.
- Torch only inside the class. NumPy `float64` at its boundaries, like the rest of OpenBO.
- Seeded through `torch.Generator` so runs repeat exactly on CPU.
- Tests: the variance is positive; a fit on a 1-D toy problem is calibrated (wider away from the data); replay targets are matched; save/load gives the same predictions.

### Phase 3: Replay (`models/conbo_replay.py`) ✅ done

**As built**
- `replay_inputs(dim, grid_points_per_dim=30, n_random=50, seed)`: a full grid plus uniform points, which is the paper's 30 × 30 + 50 by default. Above 2D the grid becomes a scrambled Sobol set of `grid_points_per_dim²` points.
- `replay_targets(user_gps, x, max_variance) -> ReplayTargets`:
  - Queries every `GPScratch.posterior` at `x`. This is the variance of the latent function, without observation noise.
  - Keeps (user, input) pairs with `var < λ`, averages mean and variance over the qualifying users, and drops inputs with none.
  - Also returns `n_users` per input, `kept_fraction` per user and `excluded_users`.
  - `max_variance=None` turns the filter off, for the "without variance filter" ablation.
- `standardized_variance_threshold(user_gps, x, variance_fraction)`: returns `variance_fraction · Var(all users' replayed means)`, i.e. λ in the units the BNN trains in.
- Loading stored GPs from disk is left to Phase 5. It will reuse the TAF loader logic in `bo_taf._load_source_surrogates`, which is private today.

**λ cannot be made scale-free** (measured on users with BO trajectories at the paper's budgets: Branin 16 random + 14 EI, McCormick 3 + 4):
- With the paper's λ, each Branin user keeps 66–91% of the replay inputs (λ = 50), and each McCormick user only 13–49% (λ = 10), since a 7-point GP is uncertain almost everywhere.
- **Relative to each GP's prior variance:** useless. The fitted Branin prior variance ranges from 9k to 530k across users, so any fraction keeps everything.
- **Relative to the spread of the replayed means or the output range:** the paper's two λ values map to about 0.017 vs 0.16 in standardized variance (≈0.03 vs 0.11 of the range in standard deviation). These differ by about 10×, so the paper tuned λ per problem and per budget.
- **Decision:** keep λ in raw units for reproduction, provide the standardized conversion for new problems, and expose `kept_fraction` for tuning.

**End-to-end check** (5 Branin users, 30 BO evaluations each, λ = 50, 1,200 epochs):
- All 950 replay inputs are kept, since at least one user is confident at each. No user is excluded.
- The BNN's mean correlates at 0.975 with the true average of the users' functions (RMSE 13 against a std of 48). The fit takes 7.5 s.
- The BNN's own argmax scores −2 to −10 per user, against optima near −0.4. Branin's optima lie in narrow valleys that averaging smooths over, which is why ConBO hands over to the user GP after `α₁` iterations.
- Tests: `tests/conbo_replay_test.py` (10 tests). Stub GPs check the filter and the averages against hand computation, and real GPs check locality and the BNN fit.

**Original plan**
- `replay_targets(user_gps, inputs, lam) -> (inputs, mu_mean, var_mean, mask)` implementing the variance filter and the per-input averages.
- Tests: the filter drops high-variance pairs and fully excluded users, and the averages match a hand computation.

### Phase 4: Mixed acquisition (`acquisition/conbo.py`) ✅ done

**As built**
- `population_weight(t, alpha1=5, alpha2=0.2)`: `clip(1 − (t − α₁)·α₂, 0, 1)` for a 1-based `t`. This equals the paper's piecewise form.
- `mixed_expected_improvement(weight, pop_prediction, user_prediction, best_pop, best_user)`: computes `w·EI_pop + (1 − w)·EI_user` with the existing analytic EI. A zero-weight term is skipped, so its prediction may be `None`.
- `default_incumbents(y_user, pop_mean)` settles **open question 5**:
  - Both EIs use the best value the current user has observed, so both measure improvement over the same reference in the user's own output units.
  - Before the user has any data, both use the best population mean over the candidates.
  - Using the population model's own best predicted mean for EI_pop was rejected: once the user beats it, EI_pop would keep proposing points the user has already improved on.
- Tests: `tests/conbo_acquisition_test.py` (8 tests), covering the schedule breakpoints (1, 1, 1, 1, 1, 0.8, 0.6, 0.4, 0.2, 0), exact interpolation, the endpoints, the handover of the argmax, validation and incumbents.

**Left for Phase 5: what `t` counts.** The paper does not say whether `t` counts the user's random initial points. We will count every evaluation of the user, starting at 1, so in the study setting (T = 10) the schedule spans the whole session regardless of `r_u`. The alternative, counting only BO iterations, would give early users (many random points) almost no population guidance, which matches the intent anyway since their population model is weak. This will be a config option if it matters.

**Original plan**
- `population_weight(t, alpha1, alpha2)` and `conbo_acquisition(x, pop_pred, user_pred, best_y, w)`.
- Decide which incumbent EI uses for each model (see Open questions).
- Tests: the weight schedule at its breakpoints; `w = 1` and `w = 0` reduce to the single-model EIs.

### Phase 5: Optimizers (`optimizers/conbo.py`) ✅ done

**As built**
- `ConBOConfig` (frozen): the paper's values by default (`α₁ = 5`, `α₂ = 0.2`, `r₀ = 6`, `d_r = 2`, 40 × 40 candidates, 30 × 30 + 50 replay inputs, 1,200 replay epochs, 15 online epochs), plus switches for the open questions and the ablations:
  - `weight_clock`: `"evaluations"` (default) or `"bo_steps"`.
  - `retrain`: `"reset"` (default) or `"continue"`.
  - `online_update`, `use_user_gp` (False gives "ConBO without GP"), and `replay_filter` (False gives "without variance filter").
  - `replay_max_variance` (raw λ) or else `replay_variance_fraction = 0.05` (standardized), between the paper's Branin (0.017) and McCormick (0.16) equivalents.
- `ConBOPopulation(config, memory_dir=None)`:
  - `new_user(name)` returns user `n_users + 1`, seeded with `seed + 1009·u`.
  - `finish_user(opt)` refits the user's GP with optimized hyperparameters, stores it, retrains the BNN by replay and saves if a directory was given. It returns a `ReplayReport` (threshold, kept inputs, per-user kept fractions, excluded users, seconds).
  - `user_gps()`, `save`/`load`.
- **Memory layout:** `gp_states/<user>.json` and `trajectories/<user>.json` (the TAF layout; a test checks `bo_taf._load_source_surrogates` reads it), `population_bnn.pt`, and `conbo_state.json` (config, user order, replay reports).
- `ConBOSequentialOptimizer`:
  - Makes `r_u` random suggestions, then takes the argmax of the mixed EI over the candidate grid.
  - **Adapts a private copy of the BNN online** (`fit_nll` on all of the user's points, `online_epochs` each time), so an abandoned session never changes the population model. This settles **open question 3**: online updates use the current user's data only, and are discarded at `finish_user`, where replay retrains from memory.
  - The user GP is refit lazily, only when a suggestion needs it. Skipping the fits during random starts saves up to 16 per user.
  - **Edge cases:**
    - The first user (no population model yet) is plain BO with at least one random start.
    - A user with no data yet uses the population model only.
    - If replay keeps nothing (λ too strict for tiny users), no population model is trained and the next user is plain BO; the `ReplayReport` shows `n_inputs_kept = 0`.
- **Open question 4:** the default is `retrain="reset"`, so the population model is a function of the memory alone and online drift from the last user does not carry over. `"continue"` is available for the benchmark.
- `run_conbo_sequence(objectives, config, n_iter_per_user, names=, optima=, memory_dir=)` returns a `ConBORunResult` with per-user trajectories, the number of random starts, the weight trace, seconds and the replay report.

**Finding: what `t` counts matters a lot.** With the Branin settings (`r₀ = 16`, `d_r = 5`), users 2 and 3 make 11 and 6 random evaluations. Counting `t` over all evaluations, the weight has already fallen to 0 (or 0.6) before the first BO step, so the population model barely acts until user 4. With `weight_clock="bo_steps"`, every user after the first gets 5 population-guided steps. The first comparison should run both.

**Cost:** 3 Branin users × 30 evaluations take about 70 s (about 8 s of replay per user, and roughly 0.5 s per GP hyperparameter fit).

- Tests: `tests/conbo_optimizer_test.py` (17 tests after the Phase 6 refinement tests).

**Original plan**
- `ConBOConfig` (dataclass): `α₁`, `α₂`, `r₀`, `d_r`, `λ`, epochs, dropout samples, grid size, seed.
- `ConBOPopulation`: holds the BNN and the memory directory; `new_user() -> ConBOSequentialOptimizer`; `finish_user(optimizer)` stores the GP and trajectory and runs replay; `save` and `load`.
- `ConBOSequentialOptimizer`: `suggest()` and `observe(x, y)`, with the `r_u` random starts, then the mixed acquisition, plus online BNN updates.
- `run_conbo_sequence(user_specs, config, n_iter_per_user)`: the convenience loop over users.
- Tests: the random-start count per user, memory files written in the TAF layout, and a short end-to-end sequence that runs and is reproducible.

### First look ✅ done (2026-10-10)
One seed of the paper's Branin Test 1, ConBO against standard BO with the same machinery: `benchmark_results/conbo_branin_quick` (README there).
- ConBO leads clearly in the first 5–20 evaluations of users 4–15: mean log10 regret 0.35 vs 1.00 after 5 evaluations.
- At 30 evaluations both methods are capped by the 40 × 40 candidate grid (9 of 15 users finish at the best grid point). Phase 6 should polish the final suggestion or refine the grid.
- Users 11 and 14 ended poorly under ConBO (about −0.1), not because of the wrong Branin basin. The user considers this run-to-run randomness; recheck it across seeds in Phase 6.

### Phase 6: Benchmarks and plots (Test 1 done; Tests 2–3 and ablations pending)

**As built**
- **Grid refinement:** `ConBOConfig.refine_suggestion = True` (default) polishes the best grid point of the mixed EI with L-BFGS-B (`argmax_on_grid`), because the 40 × 40 grid capped the final regret in the first look. `refine_suggestion=False` gives the paper's pure grid search.
- `benchmarks/conbo_runner.py`: `CONBO_METHODS = (conbo, bo, taf_m, taf_r, single_gp)`, `CONBO_TESTS` (Test 1 only: `branin`, 15 users, shift 0.3, scale 0.2, 30 evaluations, `r₀ = 16`, `d_r = 5`, λ = 50) and `run_conbo_method`. Every method sees the same users per seed.
  - `bo`: every user fresh, with `r₀` random starts, then EI on the user GP (the same machinery as ConBO's first user).
  - `taf_m` / `taf_r`: all previous users as sources, stored in the TAF layout; ConBO's decaying random starts; the first user runs as `bo`.
  - `single_gp`: one GP on all previous users' data plus the current user's, EI against the current user's best, same random starts. Hyperparameters are fitted once per user: refitting every step on up to about 450 pooled points took about 3 hours per seed (65 s per fit at 420 points).
- `scripts/run_conbo_benchmark.py` (`--method --test --seeds --results-dir --test-id`) writes `trajectories/<test_id>_<method>_<test>_s<seed>.json`. `scripts/plot_conbo_results.py` plots mean regret per evaluation over users from `--from-user` on, regret per user after 5 and after all evaluations, and prints a table at `--evals`.
- Tests: `tests/conbo_runner_test.py` (8 tests, on a 3-user test with tiny training budgets).
- **Cost per seed** (five jobs in parallel): conbo about 10 min, taf_r 8, taf_m 6, bo 2.
- **TAF stalls on later users.** Once the target GP is confident (EI ≈ 0 near its data), the source-improvement terms dominate the TAF acquisition, and TAF keeps re-suggesting one source user's optimum. In the pilot, 5 of 12 later TAF-M users repeated the same point for their last 10 evaluations. This is expected TAF behavior with many sources and no random starts, not a bug, so the baseline is kept as is.

**Test 1 results (2026-10-10, `benchmark_results/conbo_branin_5seeds`, 5 seeds)**, mean log10 regret over users 4–15:

| Method | k = 1 | 5 | 10 | 20 | 30 |
| --- | --- | --- | --- | --- | --- |
| ConBO | **0.69** | **0.42** | 0.17 | **−0.54** | −1.91 |
| Standard BO | 1.46 | 0.93 | 0.63 | −0.15 | **−2.15** |
| TAF-M | 0.94 | 0.65 | 0.25 | −0.34 | −0.63 |
| TAF-R | 0.85 | 0.47 | **0.10** | −0.12 | −0.16 |
| Single GP | 1.09 | 0.51 | 0.28 | −0.03 | −0.19 |

- ConBO reproduces the paper's main claim: it is best early in a session, and beats standard BO in nearly every seed up to 20 evaluations.
- TAF-R and the single GP are close to ConBO at 5–10 evaluations.
- Standard BO ends slightly ahead at 30 evaluations: ConBO is better in only 2 of 5 seeds.
- The grid refinement lifts the grid floor from the first look: on the same seed 0, ConBO at k = 30 improves from −1.29 to −1.69 (one-seed pilot).
- The users with poor ConBO results in the first look don't stand out across seeds. Each seed draws different users, so positions aren't comparable.

**Original plan**
- `scripts/run_conbo_benchmark.py`: Tests 1–3 with the paper's settings, several seeds, JSON trajectories per method under `test_results/conbo/`.
- **Methods:**
  - `conbo`, plus the ablations `conbo_no_gp`, `conbo_no_filter`, `bnn_no_replay`, `bnn_direct_replay`.
  - `bo_scratch` per user (standard BO).
  - `bo_taf_m` / `bo_taf_r` using all previous users as sources, read from the same memory directory.
- **Single GP over all users:** optional, since its cost grows cubically.
- `scripts/plot_conbo_results.py`: regret per iteration for selected users, mean regret per user index (does it improve with more users?), and time per iteration.
- Archive a first run under `benchmark_results/conbo_*` with a `run_all.sh`, like `mobo_2obj`.

### Phase 7: Docs and server (README section and server done 2026-10-11)
- ✅ README §9 "Continual BO (ConBO)": pieces, Python usage for a whole sequence and for one user per session, notes and switches, benchmark commands. Later sections renumbered (results folders §10, benchmark results §11, structure §12).
- README: move ConBO from the roadmap into its own section, plus the §0 scope list, §10 results and §11 structure.
- ✅ Server (2026-10-11): `server_optimizers/conbo_server.py`, port 8769. One WebSocket session per user, one user at a time, population in `memory_dir`; `stop` discards unless `store: true`. See `docs/guide/servers.md` and `docs/guide/conbo.md`.

## Out of scope (first pass)
- Real users and the VR typing study.
- Preference feedback.
- More than 2 dimensions in the paper reproduction, although the code should not assume d = 2.
- GPU training.

## Open questions
1. ✅ *Settled 2026-10-10: no reference code.* **Original code.** The paper says ConBO is on the project page, but the page links only the PDF and arXiv. We treat the paper as the only reference: questions 2–5 get documented defaults, made configurable where they affect forgetting. Success means reproducing the paper's qualitative behavior (regret falls as more users are seen, earlier users are not forgotten in Test 3, the ablations rank roughly as reported), not its exact numbers.
2. **BNN training settings.** The optimizer, learning rate, batch size and number of dropout samples at prediction are not given. Start with Adam (lr 1e-3), full-batch training and 50 samples, then check sensitivity.
3. **What the online update trains on.** It could use the current user's points only, or mix them with replay data. This changes how much the model forgets earlier users during a session.
4. **Replay schedule.** Retrain from scratch after each user, or continue from the current weights? And does eq. 7 replace the NLL loss, or add to it?
5. **EI incumbents.** EI for the population model and for the user model may need different incumbents. Using the best observation so far for both is the simplest choice, but the scales can differ.
6. **λ in standardized units.** Check that the proposed standardized λ reproduces the paper's behavior with its raw values (50 for Branin, 10 for McCormick).
7. **Number of repeats** for the benchmark. The paper doesn't say. Use at least 5 seeds, or 10 if the runs are cheap enough.
