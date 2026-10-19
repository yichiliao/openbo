# Plan: TAF variant for human-in-the-loop optimization (TAF+)

Status: **Phases 0–5 done** (2026-10-11): plan, acquisition (`acquisition/taf_plus.py`), optimizer with population store (`optimizers/bo_taf_plus.py`), workflow steps 2–3 (`optimizers/taf_plus_workflow.py`), the App. B simulations (`benchmark_results/taf_plus_sims/`) and the WebSocket server (`server_optimizers/taf_plus_server.py`). Results are reported in `docs/guide/results.md` and the archive's README; `docs/guide/taf-plus.md` is the user guide. Deferred: open questions 5 and 7, which the simulations showed matter. Any fix for those would be a deliberate deviation from the papers. The user confirmed (2026-10-11) that the stall after the first design matches their own earlier TAF+ results and is structural; the answer is a learned acquisition (NAF, next on the roadmap), not patches to TAF+.

References:

- **Variant (TAF+):** Y.-C. Liao, R. Desai, A. M. Pierce, K. E. Taylor, H. Benko, T. R. Jonker and A. Gupta, [*A Meta-Bayesian Approach for Rapid Online Parametric Optimization for Wrist-based Interactions*](https://dl.acm.org/doi/full/10.1145/3613904.3642071), CHI 2024 ([open-access PDF](https://acris.aalto.fi/ws/portalfiles/portal/146008506/3613904.3642071.pdf)). The main paper and appendices A, B and D have been read.
- **Original TAF:** M. Wistuba, N. Schilling and L. Schmidt-Thieme, [*Scalable Gaussian process-based transfer surrogates for hyperparameter optimization*](https://doi.org/10.1007/s10994-017-5684-y), Machine Learning 107, 2018 (open access). §6 (SGPT and its weights), §7 (TAF) and §8.2/8.5.3 (variants and experiments) have been read.
- No reference code is available for either paper (the CHI 2024 code was not released by Meta). The papers are the only references.

## The rule

> **Whenever the CHI 2024 paper and the original TAF paper (Wistuba et al., 2018) disagree, follow the original TAF paper.**

TAF+ adds two things on top of TAF: multiple objectives and a decay on the population weights. Those come only from the CHI 2024 paper, so there is nothing to disagree with. Everything TAF+ inherits from TAF (the source improvement term, the model weights, the normalization of the combination) must follow the original paper, even where the CHI 2024 paper describes it differently or more loosely. Every place where this rule decided something is listed in "Disagreements" below.

## Paper summary (CHI 2024)

Terminology: *population models* are the GPs of earlier users (TAF's source tasks); the *adaptation model* is the GP of the current user (TAF's target).

**TAF (as described in §3.3)**
- Eq. 2: `TAF(x) = (Σ_j w_j·PEI_j(x) + w_{M+1}·EI(x)) / Σ_{j=1}^{M+1} w_j`, where `EI` is the adaptation model's expected improvement and `PEI_j` is the "population expected improvement" of population model `j`, "calculated similarly to EI" (referring to §7 of the original).
- §3.3.1: the weights are variance-based. A model's weight at `x` is higher when its predictive variance at `x` is lower. The paper says this is the method the original calls TAF-M (§6.1 of the original), and that Volpp et al. call it TAF-ME.

**Extension 1: multiple objectives (§4.1.1)**
- Every model (population and adaptation) predicts `L` objectives, so it gives `L` acquisition values and `L` weights per `x`.
- *Within-model combination* with objective weights `α ∈ Δ^L` (eq. 4): `PEI_j⁺(x) = Σ_l α_l·PEI_j^l(x)` and `EI⁺(x) = Σ_l α_l·EI^l(x)`.
- The weights are combined the same way (eq. 5): `w_j⁺ = δ(t)·Σ_l α_l·w_j^l(x)` for population models and `w_{M+1}⁺ = Σ_l α_l·w_{M+1}^l(x)` for the adaptation model.
- Then the *between-model combination* is TAF's eq. 2 with the `+` quantities (eq. 3).
- `α` can be changed at any time, even after the population models are built.

**Extension 2: decay (§4.1.2, eq. 6)**
- `δ(t) = 1` for `t ≤ d1`, `δ(t) = 1 − (t − d1)·d2` for `d1 < t ≤ d1 + 1/d2`, and `δ(t) = 0` after that. `t` is the iteration count, `d1 ≥ 0` is an integer and `d2 ∈ (0, 1]`.
- It multiplies only the population weights. Once it reaches 0, TAF+ is plain EI on the adaptation model.
- This is the same schedule as ConBO's `population_weight` (α₁/α₂), but applied to TAF weights rather than to a two-model mix.

**Workflow (§4.2, Fig. 5)**
1. *Population modeling.* Each population user runs multi-objective BO (qEHVI in BoTorch). Their data gives one GP per user that predicts all `L` objectives.
2. *Objective weight optimization* (Algorithm 1, App. A.2). Each population user rates their own Pareto-optimal designs. Ratings are rescaled per user to [1, 100]. For every weight vector on a 0.1 grid with all entries non-zero, take each user's Pareto design with the highest weighted sum and add up its rating. Pick the weight vector with the highest total.
3. *Decay hyperparameter optimization* (§4.5). Leave-one-out simulation: each population model in turn plays the new user, and the remaining ones are the population. Grid search over `d1 ∈ {1, …, 9}` and `d2 ∈ {0.1, 0.2, 0.3}`, 10 iterations each, picking the pair with the best average weighted-sum performance.
4. *Adaptation.* Run TAF+ on the new user with the population models, `α` and `(d1, d2)`.

**Settings**
- GPs: BoTorch `SingleTaskGP`, Matérn-5/2, inferred homoscedastic noise. The same settings for population and adaptation models.
- Acquisition search: argmax over 1024 Sobol candidates per iteration (study), 256 random candidates (simulations).
- Study: 14 population users, 10 adaptation iterations, `α` = [0.4, 0.3, 0.3] and [0.5, 0.3, 0.2], best `(d1, d2) = (2, 0.3)` for both interactions. The BO baseline uses 5 random starts plus 5 iterations on the same weighted sum.
- It is not stated whether TAF+ uses any random initial points. Read literally, it starts from the population models alone.

**Simulations (App. B)**
- Base function: 4D input, 3 objectives. Each objective is a 2D "Sphere" `y = 1 − 8·Σ_{i=1}^{2}(x_i − c_i)²` on a pair of inputs: (x1, x2) with centre (0.55, 0.4), (x2, x3) with centre (0.6, 0.45), (x3, x4) with centre (0.65, 0.35). Gaussian noise with σ = 0.05.
- Users: shift every input by `U(−r_s/2, r_s/2)` and scale the output by `U(1 − r_c/2, 1 + r_c/2)`. 20 users per group: 10 population, 10 new.
- Population modeling: MOBO with 20 random starts plus 20 iterations.
- Simulation 1: `r_s = r_c = 0.01`, six objective weight vectors, no decay.
- Simulation 2: ranges 0.05, 0.1, 0.2 and 0.3, `α` = [0.3, 0.5, 0.2]. Best decay settings were (7, 0.1), (4, 0.1), (4, 0.2) and (3, 0.2).
- Simulation 3: other base functions (relocated spheres, spheres plus Branin, sphere plus Branin plus Hartmann-3; App. Tables 7–9). Simulation 4: population sizes.
- Baselines: standard BO (20 random starts plus 20 iterations) and random sampling.

## Original TAF summary (Wistuba et al., 2018)

The original is written for minimization. OpenBO maximizes, so every `min` below becomes a `max` with the sign flipped.

- **Surrogate.** One GP per source data set `i = 1..M`, fitted on that data set only, plus one target GP `M+1` on the new data set's observations (§6, eq. 30).
- **TAF (§7, eqs. 46–47).** `a(x) = (w_{M+1}·EI_{M+1}(x) + Σ_{i=1}^{M} w_i·I_i(x)) / Σ_{i=1}^{M+1} w_i`, with `I_i(x) = max(y_{D_i}^min − Ŷ_i(x), 0)`, `Ŷ_i(x) ~ N(μ_i(x), σ_i²(x))`.
  - `y_{D_i}^min` is "the best value achieved so far on `D_i`". §8.5.3 explains that "the improvement on the meta-data shrinks over time as better x have been found", so it is the best value on source `i` **among the configurations tried so far on the target**, not the best of source `i`'s own data.
  - The paper calls `I_i` the "predicted improvement" on source `i`, as opposed to the target's "expected improvement".
- **Weights** are "set analogously to" the surrogate framework (§7, last sentence), giving three variants (§8.2, §8.5.3):
  - **TAF-PoE** (§6.1, generalized product of experts): `w_i = β_i·σ_i⁻²(x)` with `β_i = 1/(M+1)` for every model, the target included (eqs. 38–39). Per point, variance-based.
  - **TAF-M** (§6.2, kernel regression with meta-features): `w_i = k_ρ(χ_i, χ_{M+1})` with the Epanechnikov kernel `γ(‖χ_i − χ_j‖/ρ)`, `γ(t) = ¾(1 − t²)` for `t ≤ 1` and 0 otherwise (eqs. 41–43). Constant over a run.
  - **TAF-R** (§6.2, kernel regression with pairwise rankings): the same kernel on descriptors built from pairwise comparisons of the target's evaluated points under each source's mean `μ_k` (eq. 45), so the distance is the normalized number of discordant pairs (a Kendall-tau distance). Updated after every iteration. TAF-R did best in the experiments.
  - With kernel regression the target's own weight is `k_ρ(χ_{M+1}, χ_{M+1}) = γ(0) = ¾`. The paper does not write this out for TAF; it follows from "analogously".
- **Scales.** TAF was introduced to avoid normalizing across data sets: each `I_i` is measured on source `i`'s own scale (§7, first paragraph).
- **GPs** in the experiments: SE-ARD kernel, kernel parameters learned by maximizing the marginal likelihood on the meta-training set (§8.2).

## Disagreements and how the rule resolves them

| # | Topic | CHI 2024 | Original TAF | Resolution |
| --- | --- | --- | --- | --- |
| D1 | Source term | `PEI_j`, "population expected improvement", "calculated similarly to EI" (eq. 1). | `I_i(x) = max(y_{D_i}^min − Ŷ_i(x), 0)`, called the "predicted improvement". | **Original**, read as the improvement of the posterior mean (Open question 1). |
| D2 | Reference value in the source term | Eq. 1 uses `f⁺`, the best value observed so far, and does not say which model it is measured on. | The best value on source `i` among the configurations tried so far on the target. | **Original.** In OpenBO, the sources have no data at the target's points, so source `i`'s value there is its posterior mean `μ_i`: the reference is `max_{x ∈ X_target} μ_i(x)` (the same reading as MetaBO's code). |
| D3 | Name of the variance-based weights | Calls them TAF-M, citing §6.1 of the original. | §6.1 is TAF-**PoE**. TAF-M is the meta-feature kernel regression of §6.2. | **Original names.** TAF+ uses TAF-PoE weights (variance-based, as the CHI paper intends). OpenBO's existing `taf_m` (meta-feature distance) already matches the original's meaning of TAF-M. |
| D4 | Weight formula | "Lower variance gives a higher weight", no formula. | `w_i = σ_i⁻²(x)/(M+1)` for every model including the target. | **Original.** The `1/(M+1)` factor cancels in eq. 46 but is kept for fidelity. |
| D5 | GP kernel | Matérn-5/2 with inferred noise (BoTorch `SingleTaskGP`). | SE-ARD, hyperparameters by marginal likelihood (an experimental setting, not part of the TAF definition). | **Original** (decided): `kernel_type="rbf"` by default, Matérn-5/2 as an option. |

TAF+'s own additions (objective weights α, the decay δ(t) and the workflow steps) exist only in the CHI 2024 paper and are taken from it unchanged.

## How it differs from OpenBO's current TAF

OpenBO's TAF (`acquisition/taf.py`, `optimizers/bo_taf.py`) departs from the original in three ways (see `naf.md` open question 5):

- Its source reference is a fixed 90% quantile of each source's own data, not the best source mean over the target's points.
- It uses a softplus instead of `max(·, 0)`.
- It has no TAF-PoE (variance-based) weights. Its `taf_m` weights (meta-feature distances) do match the original's TAF-M, though they are normalized to sum to one and the target weight is a free parameter rather than ¾.

TAF+ must use the original forms. The archived TAF results depend on the current defaults, so the proposal is to **leave `bo_taf` and its defaults as they are** and build TAF+ as a separate method that follows the original. With one objective (`α = [1]`) and no decay (`δ ≡ 1`), TAF+ reduces to the original TAF. That gives OpenBO a faithful TAF baseline as a by-product, which NAF needs anyway.

## Mapping onto OpenBO (proposed)

| Topic | Decision |
| --- | --- |
| Scope of first pass | Synthetic only: reproduce App. B Simulations 1 and 2 (4D, 3 objectives). Sequential (q = 1). |
| Acquisition | New `acquisition/taf_plus.py`: the original source improvement and weights (per objective), the within-model combination (eqs. 4–5), the decay `δ(t)` (eq. 6) and the between-model combination (eq. 3). Vectorized over candidates. |
| Population models | One `GPScratch` per objective per population user (independent objectives, as with BoTorch's model list in the paper). Stored in the TAF layout, extended to `L` objectives: `gp_states/<user>.json` holds one state per objective, `trajectories/<user>.json` holds `x_values` and an `(n, L)` `y_values`. |
| Adaptation model | One `GPScratch` per objective, with fitted noise. Kernel per D5. |
| Population modeling (Step 1) | Reuse OpenBO's MOBO (`MOBOSequentialOptimizer` or the qLogNEHVI baseline in `mobo_botorch.py`) to collect each population user's data, then fit and save the per-objective GPs. |
| Objective weights (Step 2) | `optimize_objective_weights(pareto_y_per_user, ratings_per_user, step=0.1)`, a direct port of Algorithm 1. Synthetic users have no ratings, so the simulations set `α` by hand, as the paper does. |
| Decay search (Step 3) | `search_decay(...)`: the paper's leave-one-out grid search over `(d1, d2)`. |
| Optimizer | `optimizers/bo_taf_plus.py`: `TAFPlusConfig` (frozen dataclass), `TAFPlusSequentialOptimizer` (`suggest()` / `observe(x, y)` with `y` of shape `(L,)`) and `run_bo_taf_plus`. Iteration count `t` is the number of observations so far. |
| Candidate search | Sobol pool (1024 by default) plus an optional L-BFGS-B polish, like ConBO's `argmax_on_grid`. The paper's pure pool search is `refine_suggestion=False`. |
| Benchmarks | `benchmarks/taf_plus_runner.py` and `scripts/run_taf_plus_benchmark.py` / `plot_taf_plus_results.py`, kept out of `runner.py` like MOBO and ConBO, since the target is a weighted sum of several objectives. Methods: `taf_plus`, `taf` (multi-objective, no decay: the paper's TAF baseline), `bo` (weighted-sum EI), `random`. |
| Test functions | Add the App. B sphere base function and its shifted/scaled users to `test_functions` (`multi_objective.py` or a new module), reusing `TaskVariantSpec`-style shifts. |
| Server | Later phase: a TAF+ WebSocket server on the next free port (8770), `observe.y` with one value per objective, as in the MOBO server. |

## Implementation phases

### Phase 0: Read both papers ✅ done
- Both papers read and the Disagreements table filled in (2026-10-11).
- Open questions 1–4 answered the same day.
- Confirm the decisions above.

### Phase 1: Acquisition (`acquisition/taf_plus.py`) ✅ done
- The original source term (D1, D2) and TAF-PoE weights (D3, D4), per objective, then eqs. 3–6. TAF-R weights as an option, since they did best in the original.
- Tests: with `L = 1`, `α = [1]` and `δ ≡ 1`, the value matches a hand-written original TAF; `δ = 0` gives plain EI on the adaptation model; the weights follow eq. 5; the decay schedule matches eq. 6 at its corners.
- Done 2026-10-11 (`tests/taf_plus_acquisition_test.py`, 14 tests). The module works on predictive means and variances, not GPs, like `acquisition/conbo.py`; the optimizer does the GP calls.
  - `decay_factor(t, d1, d2)` is eq. 6. It reuses ConBO's `population_weight`, which is the same schedule.
  - `source_references` takes each source's mean at the target's evaluated points and returns its maximum (D2). `predicted_improvement` is `max(μ − ref, 0)` (D1).
  - `poe_weights` is `σ⁻²/K` over all `K = M + 1` models (D4).
  - `ranking_distance` is the fraction of ordered pairs of target points on which `1[f(x_i) > f(x_j)]` differs between the target's observations and a source's means (eq. 45, as the original's text describes it). With fewer than two points it is 0, so every source gets the full ¾. `taf_r_weights` applies the Epanechnikov kernel per source and objective; the target gets `γ(0) = ¾`.
  - `taf_plus_acquisition` takes `(n, L)` target predictions and `(M, n, L)` source predictions and applies eqs. 4, 5 and 3. Objective weights must be non-negative and sum to 1.

### Phase 2: Optimizer and population store (`optimizers/bo_taf_plus.py`) ✅ done
- Ask/tell optimizer, `run_bo_taf_plus`, save/load of multi-objective population models.
- Tests: the loop runs, the population is read back identically, and objective values never leak between users.
- Done 2026-10-11 (`tests/taf_plus_optimizer_test.py`, 11 tests).
  - `TAFPlusConfig` (frozen): `objective_weights`, `decay` (False gives the paper's "TAF (multi-objective)" baseline), `d1 = 2`, `d2 = 0.3` (the study's values), `weight_mode` (`poe` or `taf_r`), `rho`, `kernel_type = "rbf"`, `n_candidates = 1024`, `refine_suggestion`.
  - `TAFPlusPopulation`: `add_user(name, x, y)` fits one GP per objective (hyperparameters and noise by marginal likelihood) and stores their states; `subset(exclude)` for leave-one-out; `save`/`load`. Layout: `gp_states/<user>.json` with key `gp_states` (a list, one per objective), `trajectories/<user>.json` with `x_values` and `(n, L)` `y_values`, and `taf_plus_population.json`. The single-objective TAF loader (`bo_taf`) cannot read it.
  - `TAFPlusSequentialOptimizer`: the first point is the domain centre; after that, the argmax of TAF+ over a Sobol pool, polished by L-BFGS-B (`argmax_on_grid` from ConBO). The decay's `t` counts evaluations from 1, so the centre is `t = 1` and the first TAF+ step is `t = 2`. The target GPs refit (with fitted noise) at every step.
  - `result()` reports the weighted sum `y @ α` and its best-so-far.
- Two fixes needed for the target GP with **one** observation:
  - `GPScratch` standardizes a single point (or all-equal values) with a std of `1e-12`, so its predictive variance drops to about `1e-24`. Under TAF-PoE the target's weight then swamps every source. The optimizer resets `y_std` to 1, which is exact because the normalized targets are all 0. `GPScratch` itself is unchanged, because ConBO's first user goes through the same path and archived results depend on it.
  - Fitting hyperparameters to one point collapses the signal variance (to about `5e-5`), with the same effect. The target keeps its prior hyperparameters until it has two points.
- Cost: the population's predictions on the candidate pool are cached once per run. A step then takes 3–7 s with 5 population users, 3 objectives and 1024 candidates, mostly refitting the target's hyperparameters. Fitting a population user (3 objectives, 40 points) takes about 13 s.
- Observed: because eq. 4 sums **per-objective** clipped improvements, TAF+ heads for points where one objective improves a lot rather than for the weighted-sum optimum. See Open question 5.

### Phase 3: Workflow steps 2 and 3 ✅ done
- `optimize_objective_weights` (Algorithm 1) and `search_decay` (leave-one-out grid search).
- Tests: the Table 1 example in the paper picks [0.2, 0.3, 0.5] among its three weight vectors. (Table 1 prints design A's weighted sum under [0.2, 0.3, 0.5] as 5.5; it is 4.5. The winner, B at 6.2, is unaffected.)
- Done 2026-10-11 in `optimizers/taf_plus_workflow.py` (`tests/taf_plus_workflow_test.py`, 7 tests).
  - `objective_weight_grid(L, step=0.1)`: every vector of non-zero tenths summing to 1 (36 for 3 objectives). `normalize_ratings` rescales a user's ratings to [1, 100]; if they are all equal, all become 100 (the algorithm divides by zero there, and any constant leaves the argmax unchanged). `optimize_objective_weights(pareto_y, ratings, candidate_weights=None)` returns the best vector and every score. Ties go to the first vector and the first design.
  - `search_decay(population, config, d1_values, d2_values, n_iter=10, n_workers=1)`: each population user in turn is the new user, simulated by the **posterior mean of their own GPs** (the paper: "we singled out a population model and treated it as the new test user"), with the others as the population. A pair's score is the mean weighted best-so-far over users and iterations (the area under the paper's Fig. 7 curves; the paper does not define "best performance" more precisely). Also returns the no-decay baseline. `n_workers > 1` runs pairs in separate processes, with identical results.
  - Cost: about 1.4 s per step with 10 population users, 3 objectives, a 256-point pool and no polish, almost all of it refitting the target GPs. The paper's grid (27 pairs plus the baseline) × 10 users × 10 steps is about an hour serially.

### Phase 4: Simulations ✅ done
- App. B Simulation 1 (six `α` vectors, no decay) and Simulation 2 (four similarity ranges with the decay search). Archive in `benchmark_results/` with a `run_all.sh`.
- Code (2026-10-11): `test_functions/taf_plus_users.py` (`three_spheres`, `SphereUser`, `make_sphere_users`, `SphereUser.weighted_optimum`), `benchmarks/taf_plus_runner.py` (`TAF_PLUS_METHODS = (taf_plus, taf, bo, random)`, `TAF_PLUS_SIMULATIONS`, `build_population` with an on-disk cache, `run_taf_plus_method`), `scripts/run_taf_plus_benchmark.py` and `scripts/plot_taf_plus_results.py`. Tests: `tests/taf_plus_runner_test.py`.
- Choices and deviations, also listed in the runner's docstring:
  - Noise (std 0.05) is added after the user's output scale, as observation noise; the paper does not say which.
  - Population modeling uses qLogNEHVI (`mobo_botorch`) with 256 raw samples and 4 restarts, not qEHVI with 1024 and 10. One user took 149 s instead of 343 s and reached about the same hypervolume. Reference point (−1, −1, −1).
  - TAF+ and TAF search a **fresh** 256-point Sobol pool at every step (`resample_candidates=True`, the study's "at each iteration, we generated ... candidates") and **polish** the best candidate with L-BFGS-B (the paper: pure search over 256 random points). See "Second attempt" below for why.
  - Every method gets 20 evaluations per new user. BO is scratch EI (RBF) on the noisy weighted sum with 5 random starts, as in the paper's user study; the simulations' 20 random starts would make BO identical to random search within 20 evaluations.
  - Scored by the regret of the **noise-free** weighted sum at the best point so far (`optimum − max_{s≤t} f(x_s)·α`), with the optimum from `SphereUser.weighted_optimum`. The paper plots the weighted value against a grid-search maximum.
  - One seed (10 new users) per setting, as the paper.
- Run: `benchmark_results/taf_plus_sims/run_all.sh` (test id `v1`).
- **First attempt (2026-10-11), stopped.** The `sim2_r005` decay search gave every `(d1, d2)` pair and the no-decay baseline the same score (0.8598). Two causes: the centre start (see Open question 3), and the score, the mean **best-so-far**, which saturated by step 2 in every leave-one-out run, before the decay (which acts only after step `d1`) could change anything. Ties went to the first pair, (1, 0.1); the paper reports (7, 0.1). Fixes, decided the same day: the population-mean first point, and scoring the decay search by the mean weighted value of **each evaluated design** (what a user experiences per iteration; the paper's Fig. 7 does not say which it plots). The runner records both `instant_regret` and best-so-far `regret`; the plot script defaults to the former (`--metric instant`). The populations were unaffected and kept.
- **Second attempt (2026-10-11), stopped after two ranges.** With one fixed 256-point pool per run, every new user's final TAF+ regret equalled the regret of the best point in their pool (e.g. 0.113, 0.162): TAF+ found it at step 1 and could never do better, while BO's continuous search reached 0.03. A fresh pool each step alone barely helped (final regret 0.085). Diagnostic on the `sim2_r005` users, decay (9, 0.1), 10 new users, mean log10 regret of the best so far:

  | Search | step 1 | step 20 |
  | --- | --- | --- |
  | 256 fresh pool, no polish | −0.99 | −1.07 |
  | 1024 pool, no polish | −1.35 | −1.37 |
  | 256 pool + L-BFGS-B polish | **−2.47** | −2.47 |
  | BO (5 random starts, continuous) | 0.01 | −1.49 |

  Decided the same day: a fresh pool each step plus polish. Transfer itself works very well; the coarse pool hid it.
- **Results (2026-10-11, third attempt):** `benchmark_results/taf_plus_sims/README.md` has the tables and plots. In short:
  - The first point (population product-of-experts mean) reaches a regret of 0.003 / 0.013 / 0.055 / 0.13 for ranges 0.05 / 0.1 / 0.2 / 0.3, and about 0.0003 in Simulation 1 for all six `α`. That beats 20 evaluations of BO up to range 0.2; at 0.3, BO overtakes TAF+ by evaluation 20.
  - TAF+ barely improves after step 1, and the designs it shows after step 1 are no better than random (open question 7).
  - With one-hot `α`, TAF keeps showing fairly good designs after step 1; with any mixed `α`, it jumps to random-search level from step 2. This is direct evidence for open question 5 (eq. 4).
  - The decay search picks late, slow decay in every range ((6, 0.1), then (8, 0.1) three times), within about 0.03 of no decay; the paper's earlier, faster choices for diverse users score worse here. TAF+ and TAF without decay are nearly identical.
- **Observed in every variant: after step 1, TAF+ explores.** Once the first point sits near the population's optimum, each source's predicted improvement over its reference (D2) is close to zero everywhere, so the target's EI takes over and explores. The best-so-far stays excellent, but the regret of each evaluated design after step 1 is large (around 10^0.2). For human-in-the-loop use, where the user experiences every design, this is a property of TAF's structure (original eqs. 46–47), not of this implementation. It is also why the per-iteration decay score favours little decay. See Open question 7.

### Phase 5: Server and docs ✅ done
- Done 2026-10-11: `server_optimizers/taf_plus_server.py` (`TAFPlusServerRuntimeConfig`, `TAFPlusServerSession`, `serve_taf_plus_websocket`), `server_scripts/run_taf_plus_server.py` (port 8770), `server_scripts/run_taf_plus_fake_client.py` (one simulated sphere user), `configs/server_optimizers/taf_plus_server.yaml` and `tests/taf_plus_server_test.py` (7 tests).
  - One session is one new user, modelled on the MOBO server: `observe.y` has one value per objective, checked against `y_range`. The `start` message may override any `TAFPlusConfig` field except the kernel, which comes from the population.
  - The population is loaded from `population_dir` at each `start` and never changed. The CHI paper builds it offline before adaptation, so finished users are not added (unlike ConBO's server).
  - No random starts: the first suggestion (`phase: population`) is the population's product-of-experts optimum. Each `suggest` reports `decay`, `δ(t)`.
  - Smoke test with the shipped config (archived range-0.1 population, 1024 candidates plus polish): about 6 s per step; one simulated user reached a noise-free regret of 0.011 in 10 evaluations, with the same exploration after step 1 as the simulations.
- README §0, §5 (server list, port table, config, protocol, client loop, fake client), §10 (user guide, including 10.4 on the server), §11 results folders, §12 results and §13 structure.

## Open questions

1. ~~**Reading `Ŷ_i` in the source term (D1).**~~ *Decided 2026-10-11:* the improvement of the posterior mean, `max(μ_i(x) − ref_i, 0)` in maximization form. The original calls it the "predicted improvement" and contrasts it with the target's "expected improvement"; this is also MetaBO's reading. The other reading, EI under each source's posterior, is closer to the CHI paper's "PEI" and is not implemented.
2. ~~**GP kernel (D5).**~~ *Decided 2026-10-11:* the rule extends to the kernel. `kernel_type="rbf"` (SE-ARD) is the default, `"matern52"` (CHI 2024) is an option.
3. ~~**No target data yet.**~~ *Decided 2026-10-11, revised the same day:* the first evaluation maximizes the population's **product-of-experts mean** of the weighted objective, `Σ_j w_j⁺ μ_j⁺ / Σ_j w_j⁺` over the sources only (`population_mean`; the original's SGPT-PoE mean, eq. 37, combined over objectives as in CHI eqs. 4–5; a plain average under TAF-R). TAF+ proper starts from the second step, when the reference in D2 is defined. The first choice was the domain centre, but App. B's sphere optima sit close to the centre (mean centre regret 0.05 of an optimum near 0.9 for ranges up to 0.1), which handed TAF and TAF+ a head start that BO's random starts do not get. The centre stays available as `first_point="centre"`, and is the fallback for an empty population.
4. ~~**Random initial points.**~~ Settled by question 3: none.
5. **Eq. 4 is not the improvement of the weighted sum.** `Σ_l α_l·max(μ_j^l − ref_j^l, 0)` rewards a point that improves one objective a lot even if it worsens another, while `max(Σ_l α_l·μ_j^l − ref_j, 0)` would measure the improvement of the weighted-sum objective that the run is scored on. Likewise each `EI^l` uses objective `l`'s own best value, which can come from a different point for each objective. This is what the CHI paper writes, and the original (single-objective) has nothing to say, so it stays as written. In a 2-objective test (`test_population_guides_the_second_point`), the second point went to one objective's optimum rather than the weighted optimum. **It matters (Phase 4):** in Simulation 1, with one-hot `α`, TAF's designs after step 1 stay fairly good (log10 regret about −1 to −2); with any mixed `α` they are at random-search level from step 2. Faithful to the CHI paper, so unchanged; an alternative that scores the improvement of the weighted sum would be a deliberate deviation.
6. **Output scale.** Per D2 and the original's §7, each source term stays on its own source's scale. Population users with larger output scales therefore pull harder. This follows the original; noted here only because the App. B users differ in output scale by up to ±15%.
7. **Exploration after the population's optimum is found.** TAF (and TAF+) has no term that rewards staying near a good design: once the sources predict no improvement, only the target's EI is left. Whether the CHI study's users saw this (its plots look like best-so-far curves) is not stated. Possible follow-ups, none faithful to either paper: an exploitation-leaning target acquisition, or stopping once the population's optimum is reached.
