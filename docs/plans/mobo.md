# Plan: Multi-objective BO with correlated objectives (CEIPV)

Status: Phases 1–7 done; first 2-objective benchmark done (SLF-GP sweep and VLMOP3 still to run). Last updated 2026-10-10.

Reference: A. Shah and Z. Ghahramani, [*Pareto Frontier Learning with Expensive Correlated Objectives*](https://proceedings.mlr.press/v48/shahc16.html), ICML 2016 (PMLR 48:1919–1927). The main paper has been read; the supplementary material (EIPV derivation and test-function definitions) has not.

See also: Y.-C. Liao, G. B. Mo, J. J. Dudley, C.-L. Cheng, L. Chan, P. O. Kristensson and A. Oulasvirta, [*Practical approaches to group-level multi-objective Bayesian optimization in interaction technique design*](https://journals.sagepub.com/doi/full/10.1177/26339137241241313), Collective Intelligence 3(1), 2024. It applies MOBO to groups of users: deriving designs for a group from individual users' data, and initializing new users from group data. Not implemented in OpenBO.

## Decisions

| Topic | Decision |
| --- | --- |
| Hyperparameters | MAP fitting with L-BFGS-B first, like `GPScratch`. Slice sampling (the paper's fully Bayesian setup) may come later behind a flag. |
| Baselines | BoTorch `qLogNEHVI` is the main external baseline. ParEGO and NSGA-II are dropped as outdated. IEIPV and random search stay only as a correlation ablation and a sanity floor, since both are nearly free (see Phases 4–5). |
| Implementation | NumPy core, like `bo_scratch`, with the BoTorch baseline alongside, like `bo_botorch`. |
| Reference point `v_ref` | **Loose**: the worst value of each objective over the whole domain, rounded outward. Every evaluated point adds hypervolume from the start, as in the paper's curves. Compare methods with log hypervolume regret. A tight reference (near the front's nadir) gave random designs zero hypervolume on OKA2 and DTLZ1a. It can still be passed as an override. |
| Scope of first pass | L ≤ 3 objectives, sequential (q = 1) suggestions, synthetic problems only. |

## Paper summary

**Setup**
- The goal is to maximize L objectives `f_1..f_L` over the inputs `x`.
- The quality of a set of points is its **Pareto hypervolume**: the volume dominated by the Pareto set and bounded below by a reference point `v_ref`.
- An upper corner `w_ref ≥ f(x)` for all `x` bounds the region of interest.

**Acquisition: expected increase in Pareto volume (EIPV)**
1. The sorted coordinates of the Pareto points `u_0 = v_ref, u_1..u_P, u_{P+1} = w_ref` split the region into a grid of `(P+1)^L` cells.
2. Only cells that no Pareto point dominates (`C_nd`) can gain volume.
3. Each cell `C` has lower corner `v_C` and upper corner `w_C`.
4. EIPV is `EIPV(x) = Σ_{C∈C_nd} Ψ_C(x)`, where `Ψ_C(x) = ∫ Π_l (y_l − v_{C,l}) · 1[v_{C,l} < y_l ≤ w_{C,l}] p(y | x) dy`.

**IEIPV (independent GPs, Emmerich et al. 2008)**
- `Ψ_C` factorizes into a product of univariate Gaussian integrals, which have a closed form.

**CEIPV (correlated GPs, the paper's contribution)**
- **The problem:** when `p(y | x) = N(μ, Σ)` with Σ non-diagonal, `Ψ_C` is intractable.
- **Approximation:** replace each factor `h(y) = (y − v)·1[v < y ≤ w]` with a scaled Gaussian `z·N(y; λ, τ²)` that matches `h`'s first three moments:
  - `z = ½(w − v)²`
  - `λ = (2w + v)/3` (the paper prints `½(2w + v)`, which looks like a typo; see Open questions)
  - `τ² = (w − v)²/18`
- These parameters depend only on the cell, never on `x`.
- **Result:** the product of Gaussians integrates in closed form: `Ψ̃_C(x) = Π_l z_{C,l} · N(λ_C; μ(x), Σ(x) + diag(τ²_C))`.
- `CEIPV(x) = Σ_{C∈C_nd} Ψ̃_C(x)`, which is deterministic and differentiable in `x`.

**Correlated GP models**
- **MT (multi-task, Bonilla et al. 2008):**
  - `Cov(f_l(x), f_l'(x')) = K_{ll'} · k(x, x')`, with `K = AAᵀ` and `A` lower-triangular.
  - Every objective shares the same input kernel.
- **SLF (semiparametric latent factor, Teh et al. 2004):**
  - `Cov(f_l(x), f_l'(x')) = Σ_s A_{ls} A_{l's} k_s(x, x')`, with `A` lower-triangular and `s = 1..L`.
  - Each latent function has its own kernel `k_s`.
  - This model was best overall in the paper.
- **Shared settings:**
  - Kernel: ARD Matérn-5/2, with amplitude `θ_0` fixed at 1 for CEIPV.
  - Noise: independent Gaussian noise `σ_l²` per objective.

**Experimental protocol**

| Setting | Paper's choice |
| --- | --- |
| Initial design | 5 uniform random points |
| Budget | 100 iterations |
| Repeats | 50 |
| Metric | Pareto hypervolume against iteration (mean ± std) |
| Hyperparameters | 10 slice-sampling samples, with the acquisition averaged over them |

**Test problems**
- **Synthetic SLF-GP draws:** in `[0,1]^3` with 2 objectives.
  - Objective means: 2 and −2.
  - Lengthscales: `θ(1) = [0.7, 0.4, 1]` and `θ(2) = [0.34, 0.9, 0.5]`.
  - Mixing matrix: `A_11 = 1`, `A_21 = ρ`, `A_22 = √(1 − ρ²)`.
  - Correlation: `ρ ∈ {−0.5, 0, 0.25, 0.75}`.
  - Result: correlated models gain most when |ρ| is large.
- **Benchmarks:** `oka2`, `vlmop3` and `dtlz1a`.
- **Real tasks:** `boston`, `rocket` and `llvm`.

## Implementation phases

Repo conventions apply throughout:
- Maximization, with inputs in `[0, 1]^d`.
- NumPy `float64` arrays, where `Y` has shape `(n, L)`.
- Dataclass configs and ask/tell classes with `run_*()` wrappers.
- Tests in `tests/<topic>_test.py`.

### Phase 1: Pareto and hypervolume utilities ✅ done
File: `src/openbo/benchmarks/hypervolume.py` (tests: `tests/hypervolume_test.py`)

**As built**
- `pareto_mask`, `pareto_front`, `nondominated_cells(front, v_ref, w_ref)`, `hypervolume(Y, v_ref)` and `hypervolume_gain(y_new, lower, upper)`.
- The front is clipped into `[v_ref, w_ref]`. Cell domination is found with suffix-OR sweeps over an occupancy grid, which costs O(grid size) rather than O(cells × P).
- `hypervolume_gain` gives the exact gain of a new point: `Σ_C Π_l (min(y_l, w_C,l) − v_C,l)₊`. It matches `HV(Y ∪ y) − HV(Y)` in tests. This is the ground truth for open question 2.
- Tests cover inclusion–exclusion (L = 2, 3), a 2D sweep, 3D Monte Carlo, partition of the box, monotonicity, edge cases and validation.

**Measured cost** (worst case, where every point is Pareto-optimal)
- L = 2, P = 105: about 5.7k non-dominated cells, 2 ms.
- L = 3, P = 105: about 650k non-dominated cells, 0.18 s.
- Phase 4 risk: CEIPV and `eipv_mc` sum over every cell for every candidate (and `hypervolume_gain` builds an `(n, K, L)` array). For L = 3 this needs chunking over candidates and samples, or merging cells (for example, by summing along one axis in closed form).

**Original plan**
- `pareto_mask(Y)` and `pareto_front(Y)`.
- `nondominated_cells(front, v_ref, w_ref)`: returns the arrays `(v_C, w_C)` for the cells in `C_nd`. It is vectorized over the `(P+1)^L` grid and is fine for L ≤ 3.
- `hypervolume(Y, v_ref)`: exact computation via the cell decomposition.
- Tests: compare against brute-force hypervolume for 2 objectives and Monte-Carlo hypervolume for 3, check monotonicity when a point is added, and check that dominated points add 0.

### Phase 2: Multi-objective test problems ✅ done
File: `src/openbo/test_functions/multi_objective.py` (tests: `tests/multi_objective_test.py`)

**As built**
- `MOFunctionSpec`, `MO_REGISTRY` and `get_mo_function_spec(name, v_ref=None, w_ref=None)`. Overriding `v_ref` drops `true_hypervolume`.
- `oka2`, `vlmop3` and `dtlz1a` use the literature definitions, negated for maximization. Their worst and best values over the domain were checked with multi-start L-BFGS-B.

| Problem | `v_ref` | `w_ref` | True hypervolume |
| --- | --- | --- | --- |
| oka2 | (−3.2, −5.1) | (π, 0) | 28.153 (analytic) |
| vlmop3 | (−8.3, −62, −0.2) | (0, −15, 0.1) | unknown (no closed-form front) |
| dtlz1a | (−570, −570) | (0, 0) | 570² − 0.125 (analytic) |

- On DTLZ1a the hypervolume is dominated by reaching the right region (`g → 0`) rather than by the shape of the front. Log hypervolume regret still separates methods: about 4.7 after 5 random points and about 4.0 after 105.
- `make_slf_gp_problem(rho, seed)` uses a 1024-point Sobol support with kernel interpolation and takes about 1.3 s to build.
  - Its `v_ref` and `w_ref` are the extremes of an 8192-point sample, widened by 10% of each objective's range.
  - Its `true_hypervolume` is a dense-sample estimate (a slight underestimate), so regret plots must clamp at `1e-12`.
- It uses a local `cdist`-based Matérn-5/2 function, which a test checks against `models/kernels.py`. This keeps the shared kernel untouched.
- Phase 1 follow-up: `pareto_mask` is now an O(n·P) sort-based filter, checked against the brute-force definition.

**Original plan**
- Add `MOFunctionSpec`, with fields `name, objective, bounds, dim, n_objectives, v_ref, w_ref` and an optional `true_hypervolume`.
- Add a `MO_REGISTRY` and a `get_mo_function_spec`.
- Problems:
  - `oka2`: 3D, 2 objectives.
  - `vlmop3`: 2D, 3 objectives.
  - `dtlz1a`: 6D, 2 objectives.
- Map each problem from `[0,1]^d` to its native domain and negate it for maximization.
- Use the standard literature definitions, and note that the paper's supplement (unread) may differ.
- Add `make_slf_gp_problem(rho, seed)`, which draws the paper's synthetic correlated objectives. Sample them on a dense Sobol set and interpolate with the posterior mean so they are cheap to evaluate.
- `w_ref` must be finite, so pick per-problem upper bounds from known ranges or dense sampling.
- Tests: check shapes, check that inputs are mapped to the right domain, and check that `v_ref` is dominated and `w_ref` dominates on sampled points.

### Phase 3: Multi-output GP ✅ done
File: `src/openbo/models/multi_output_gp.py` (tests: `tests/multi_output_gp_test.py`)

**As built**
- `MultiOutputGP(mode="independent" | "mt" | "slf")` covers all three modes with one formula: `Cov = Σ_s B_s[l,l'] k_s(x,x')`, using unit-variance ARD Matérn-5/2 latent kernels. It has a local kernel helper, so `models/kernels.py` is untouched.
- It standardizes each objective and uses separate noise per objective.
- `posterior(X)` returns the latent `mean` with shape `(m, L)` and `cov` with shape `(m, L, L)`. `output_correlation()` returns the prior correlation between objectives.
- **MAP fitting:**
  - L-BFGS-B with **analytic gradients**, checked against finite differences in every mode.
  - Priors: `log ℓ ~ N(log 0.5, 1)`, `log σ² ~ N(log 1e-3, 2)`, and `N(0, 1)` on the entries of `A` (or on the log variances in independent mode).
  - The first fit uses `optimizer_restarts=4` (the default plus draws from the prior). Later fits warm-start with `refit_restarts=1`.
- **Fit times at n = 105:**

  | Problem | Mode | First fit | Warm refit (+1 point) |
  | --- | --- | --- | --- |
  | L = 2 (oka2) | any | ≤ 1.2 s | ≤ 0.1 s |
  | L = 3 (vlmop3) | independent | 6.5 s | 0.7 s |
  | L = 3 (vlmop3) | mt | 9.3 s | 0.6 s |
  | L = 3 (vlmop3) | slf | 21 s | 4.1 s |

  Inside BO the first fit happens at about 5 points, so it's cheap.
- **Tests:**
  - Independent mode matches one `GPScratch` per objective.
  - Posterior covariances are PSD, and noise-free data is interpolated.
  - Posteriors are equivariant to affine rescaling of the targets.
  - MT and SLF recover the sign of ρ on SLF-GP draws.
  - Fits are deterministic for a seed.
- **Finding:** on OKA2, `mt` collapses to correlation −1 and treats f2 as noise (test error 0.89, against 0.48 for `slf` and `independent`). 16 restarts give the same optimum, so this is a real limitation of the shared kernel, as the paper says about MT. **SLF is the default.**
- If L = 3 SLF fits become a bottleneck in Phase 5, options are fewer `optimizer_maxiter` iterations on refits, or refitting hyperparameters only every few iterations.

**Original plan**
- `MultiOutputGP(mode="independent" | "mt" | "slf")`.
- Stack the training data as an `nL × nL` covariance. It is small: n ≤ ~105 and L ≤ 3.
- Standardize each objective's targets, as `GPScratch` does.
- Fit by MAP: negative log marginal likelihood with L-BFGS-B over the following, with multiple restarts and the optimizer seeded from config:
  - Log lengthscales: one per latent kernel for SLF, shared for MT.
  - The lower-triangular entries of `A`.
  - The log noise per objective.
- `posterior(X)` returns `mu` of shape `(m, L)` and `cov` of shape `(m, L, L)`, the joint covariance across objectives at each point.
- Tests:
  - With `independent` mode and L = 1, the model matches `GPScratch`.
  - On data drawn with ρ = ±0.75, the fitted `AAᵀ` recovers the sign of the correlation.
  - The posterior covariances are positive semi-definite.

### Phase 4: Acquisition functions ✅ done
File: `src/openbo/acquisition/mobo.py` (tests: `tests/mobo_acquisition_test.py`)

**As built**
- `eipv_mc(mean, cov, lower, upper, n_samples, rng, form)`: Monte Carlo EIPV with standard errors, using common random numbers. It is the ground truth for tests.
- `ieipv(mean, cov, lower, upper, form)`: exact closed form for independent objectives (uses only the diagonal of `cov`). It is the correlation ablation.
- `ceipv_terms(front, v_ref, w_ref, form, overflow)` builds the Gaussian terms once per front. `ceipv(mean, cov, terms)` and `log_ceipv(...)` evaluate them per candidate.
- `form="exact"` (default) targets the true hypervolume gain. `form="paper"` keeps the paper's literal eq. 3 for comparison.
- `hypervolume.py` gained `cell_grid(front, v_ref, w_ref)`, which returns the grid coordinates and the non-dominated mask in grid form.

**Open questions 1 and 2 settled (see the tests)**
- **λ:** the ramp's mean is `(2w + v)/3`, checked by quadrature. The printed `½(2w + v)` is not even shift-equivariant, so it is a typo.
- **EIPV form:** the literal eq. 3 badly undercounts. For one point it gave 0.0002 against a true gain of 0.29, and in a test its total is under half the exact one. CEIPV therefore uses an exact **cell-local decomposition** of the gain:
  - For `y` in grid cell `j`, `gain(y) = Σ_S c_S(j) Π_{l∈S} ramp_{j,l}(y_l) Π_{l∉S} box_{j,l}(y_l)`, where `S` ranges over the subsets of objectives.
  - The coefficients `c_S(j)` are exclusive prefix sums of the non-dominated mask, weighted by the cell widths.
  - Every factor is local to one cell, so each is moment-matched like the paper's ramp (a box of width `d` becomes `d·N((v+w)/2, d²/12)`).
  - The paper's form is the single pattern `S = all objectives`.
  - A test checks that total mass and first moments equal the integrals of the exact gain (L = 2 and 3).
- **Overflow above `w_ref`:** our `w_ref` values are tight upper bounds, so near the front the posterior often has mass above them, which the box ignores. This caused a 55% median error and would bias the search against the edges of the front. Three virtual layers at `(0.1, 0.25, 0.5)` × range above `w_ref` credit that mass with the capped gain.
- **Accuracy** against Monte Carlo (L = 2, correlations up to ±0.9): median relative error is about 1–2% for exact CEIPV, against 21–28% for IEIPV. CEIPV also ranks equal-marginal candidates with opposite correlations the same way Monte Carlo does, which IEIPV cannot do.

**Speed**
- `log_ceipv` works in log space and skips terms more than 7 marginal standard deviations away. The nearest term is always kept, so far from the front the log stays finite and acts as a lower bound. The pruned result is within 1e-4 relative of the full sum.

  | Objectives | Front points | Terms | Time per candidate |
  | --- | --- | --- | --- |
  | 2 | 20 | 1.1k | 0.1–0.15 ms |
  | 2 | 60 | 8k | 0.7–1.0 ms |
  | 2 | 105 | 23.5k | 2–3 ms |
  | 3 | 20 | 58k | 6–16 ms |
  | 3 | 40 | 364k | 40–100 ms |

- **Two objectives are fine. Three objectives are the Phase 5 risk.** Random VLMOP3 designs of 105 points give about 18 Pareto points, and BO may give more. Options:
  - a smaller candidate pool for L = 3;
  - prefiltering candidates with IEIPV, then rescoring the top ones with CEIPV;
  - evaluating on the grid structure per candidate instead of a flat term list;
  - multithreaded torch.

**Original plan**
- `eipv_mc(mu, cov, cells, n_samples, rng)`: Monte-Carlo ground truth of the exact hypervolume gain. It is used in tests and diagnostics, not in the optimizer.
- `ieipv(mu, var, cells)`: closed form for independent objectives. It serves as the correlation ablation.
- `ceipv(mu, cov, cells)`: the paper's approximation, in the stable Gaussian-product form `N(λ_C; μ, Σ + diag(τ²_C))`, computed with a Cholesky decomposition and in log space where useful.
- Tests: `ceipv` ≈ `ieipv` with a diagonal Σ and narrow cells; `ceipv` tracks `eipv_mc` (the paper's Fig. 3); CEIPV prefers the candidate that MC prefers when only correlation differs (in place of the paper's Fig. 2 toy).

### Phase 5: Optimizers ✅ done
Files: `src/openbo/optimizers/mobo.py`, `src/openbo/optimizers/mobo_botorch.py` (tests: `tests/mobo_optimizer_test.py`)

**As built**
- `MOBOConfig` has the fields `model ∈ {independent, mt, slf}`, `acquisition ∈ {ceipv, ieipv}`, `eipv_form`, `v_ref`, `w_ref`, `n_init`, `n_candidates=1024`, `n_starts=4`, `n_rescore` (optional), `optimizer_maxiter=50` and `seed`.
- `MOBOSequentialOptimizer` provides `bootstrap()`, `suggest()`, `observe(x, Y)` and `result()`; `run_mobo(objective, config, n_iter)` wraps it. `run_mo_random` is the random floor, with the same result format.
- `MOBORunResult` holds `x_obs`, `y_obs`, `hv_history`, `pareto_x`, `pareto_y` and `final_model_state`. `hv_history` has one entry after the initial design and one per iteration (`n_iter + 1` in total).
- **How the acquisition is maximized:**
  1. Score a Sobol pool by log acquisition.
  2. Run L-BFGS-B from the best `n_starts` points.
  3. Get gradients by batched forward differences: one posterior call over `d + 1` points per gradient.
  4. Floor the log acquisition at −1e4 so L-BFGS-B never sees −∞.
- With `n_rescore` set, the pool is screened with IEIPV and only the top `n_rescore` points are rescored with CEIPV.
- `MOBoTorchSequentialOptimizer` / `run_mobo_botorch`: a `ModelListGP` of `SingleTaskGP`s with `qLogNoisyExpectedHypervolumeImprovement` (q = 1, `prune_baseline=True`, 128 Sobol MC samples, 10 restarts, 512 raw samples). The reference point is `v_ref`.

**Sanity runs** (25 iterations after 5 random points, two seeds; final hypervolume)

| Problem | Random | CEIPV-SLF | IEIPV | BoTorch qLogNEHVI |
| --- | --- | --- | --- | --- |
| OKA2 (true 28.15) | 13.4 / 12.6 | 24.9 / 21.0 | 21.8 / 20.1 | 23.3 / 20.2 |
| SLF-GP ρ = −0.75 (true ≈ 9.69) | 7.85 / 7.95 | 9.11 / 9.06 | 9.28 / 9.55 | 8.31 / 9.03 |

All methods beat random, but two seeds are not enough to rank them; that is Phase 6's job. Wall time per run of this length was 25–56 s for the NumPy methods and 100–180 s for BoTorch.

**VLMOP3 (L = 3) cost:** with about 10 Pareto points the acquisition is cheap. The 3-objective SLF GP fit dominates: about 70 s for the first fit at n = 40, then 12–14 s per suggestion. `n_rescore` barely matters at this front size. Expect roughly 30–45 minutes per 100-iteration seed.

### Phase 6: Benchmarks and plots ✅ done (scripts); full runs pending
- `src/openbo/benchmarks/mobo_runner.py`: `MOBO_METHODS`, `MO_PROBLEMS`, `get_mo_problem(name, rho, problem_seed)` and `run_mobo_method(method, spec, ...)`. Methods are `mobo_ceipv_slf`, `mobo_ceipv_mt`, `mobo_ieipv`, `mobo_botorch` and `random`; problems are `oka2`, `vlmop3`, `dtlz1a` and `slf_gp`. It is kept separate from `benchmarks/runner.py` until Phase 7.
- `scripts/run_mobo_benchmark.py --method M --problem P [--rho R --problem-seed S] --seeds 0 1 2 ... [--n-iter 100 --n-init 5 --n-candidates N --n-rescore K]` writes `test_results/mobo/trajectories/<test_id>_<method>_<problem>.json` (all seeds, with hypervolume histories, x/y values, timings, reference points and true hypervolume). SLF-GP problems are labelled `slf_gp_rho+0.75_p0`.
- `scripts/plot_mobo_results.py --problem P --test-id T` writes `test_results/mobo/plots/<T>_<P>_hypervolume.png` (mean ± std hypervolume, plus log hypervolume regret when the true value is known) and, for 2 objectives, `<T>_<P>_pareto.png` (final staircase fronts for the first seed).
- **First benchmark** (2026-10-10), in `benchmark_results/mobo_2obj` with plots and `run_all.sh`:
  - **Budget:** OKA2 and DTLZ1a, 5 methods, 3 seeds, `n_init = 5`, `n_iter = 40`. Jobs ran single-threaded, 7 in parallel, and took about 6 minutes in total.
  - **Mean final log10 hypervolume regret:**

    | Method | OKA2 | DTLZ1a |
    | --- | --- | --- |
    | Random | 1.14 | 4.22 |
    | CEIPV-SLF | 0.72 ± 0.16 | 1.39 ± 0.19 |
    | CEIPV-MT | 1.03 ± 0.10 | 1.71 ± 0.35 |
    | IEIPV | 0.77 ± 0.09 | 1.28 ± 0.42 |
    | BoTorch | 0.76 ± 0.07 | 3.18 ± 0.19 |

  - **Run times:** 6–18 s per run for the NumPy methods, about 5–6 min for BoTorch.
  - **Reading:**
    - On OKA2, CEIPV-SLF, IEIPV and BoTorch are within noise. BoTorch is fastest early on. CEIPV-MT stalls, as Phase 3 predicted.
    - On DTLZ1a, BoTorch plateaus near 3.2 while the NumPy methods keep improving. The reason has not been investigated; the very large objective range together with the loose `v_ref` is one suspect.
    - Correlation modelling (CEIPV-SLF against IEIPV) shows no significant gain on these two problems.
- **An earlier attempt was abandoned.** A full run (10 seeds × 100 iterations, including VLMOP3 and the SLF-GP ρ sweep) was stopped after about 50 minutes without finishing a single VLMOP3 job. Single-threaded 3-objective SLF fits are the bottleneck.
- **Still to run:**
  - The SLF-GP ρ sweep, where the paper shows the correlation gains.
  - VLMOP3, after making L = 3 cheaper, for example by refitting hyperparameters every few iterations.
  - More seeds and iterations.

**Original plan (Phases 5–6)**
- Phase 5: `MOBOConfig` and `MOBOSequentialOptimizer` with `run_mobo()`, acquisition maximization like `bo_scratch`, results recording `best_hv_history`, the Pareto set and hyperparameters. A BoTorch `ModelListGP` + `qLogNEHVI` baseline with the same interface. Optional later: a correlated BoTorch model (`KroneckerMultiTaskGP`).
- Phase 6: `run_mobo_benchmark.py` and `plot_mobo_results.py`, reproducing Fig. 4 and Fig. 5a–c.

### Phase 7: Server, slice sampling, docs ✅ done

**Slice sampling**
- `models/slice_sampling.py`: coordinate-wise slice sampler (Neal 2003: stepping out, then shrinkage) with box bounds. Tests check that it recovers a correlated 2D Gaussian's moments and a truncated normal's mean, and that it is deterministic for a seed.
- `MultiOutputGP` gains:
  - `log_posterior(theta)`: the value only, about 3× cheaper than with gradients. A test checks it equals minus the MAP objective.
  - `sample_hyperparameters(n, rng, start, burn_in, thin)`, which stays within the MAP bounds.
  - `with_hyperparameters(theta)`: a refitted copy that leaves the original untouched.
  - The prior term moved into a shared `_negative_log_prior`; MAP results are unchanged.
- `MOBOConfig(hyperparameters="slice", n_hyper_samples=10, slice_burn_in=10, slice_thin=2)`:
  - Each iteration refits the MAP and continues the slice chain from its previous state.
  - The acquisition is averaged over samples in log space (`logsumexp − log K`). With one model this reduces exactly to the MAP path.
  - The benchmark method is `mobo_ceipv_slf_slice`.
- **Cost:** 40 sweeps take about 6 s for OKA2 at n = 60 (11 hyperparameters). Samples sit 3–10 log units below the MAP, consistent with ~p/2.

**Server**
- `server_optimizers/mobo_server.py`, `server_scripts/run_mobo_server.py` (port 8768), `server_scripts/run_mobo_fake_client.py` and `configs/server_optimizers/mobo_server.yaml` (set up for OKA2).
- **Protocol** (README §5, "MOBO server protocol"):
  - `observe.y` is a length-L list, each value checked against its objective's `y_range`.
  - `v_ref`/`w_ref` default to the range minimums/maximums.
  - `start` can override the model, acquisition, hyperparameter mode, pool size, `n_rescore`, `n_init` (at least 1), `n_iter` and `seed`.
  - `hv_history` has one entry per observation.
- It is a separate module; `bo_server.py` is unchanged.
- Tests cover a full session, rejection of every malformed observation without advancing, stop/status, config validation, the shipped YAML, and a real socket round trip. A launcher + fake client run was checked by hand.

**Wiring and docs**
- **Decision:** MOBO stays out of `benchmarks/runner.py`. That runner returns single-objective results (best value, regret), which don't fit; `benchmarks/mobo_runner.py` is MOBO's dispatcher.
- README:
  - New §8 "Multi-objective BO", with the model/acquisition summary, a Python example (checked to run), benchmark and plot commands, a methods table, cost notes and the server.
  - §0 scope updated (MOBO moved from the roadmap), §5 server list/port table/protocol, the results folders, and the §11 structure list.
  - Later sections renumbered (results folder convention §9, benchmark results §10, structure §11).

**Original plan**
- Server support: `observe.y` becomes a vector of length L, and `y_range` is set per objective.
- Slice-sampling hyperparameters, averaging the acquisition over samples.
- Wire into `benchmarks/runner.py`, and update the README §0 roadmap, workflows and structure.

## Out of scope (first pass)
- ParEGO and NSGA-II baselines.
- Real-world tasks (`boston`, `rocket`, `llvm`).
- Batch (q > 1) suggestions.
- L > 3 objectives, where the grid cell count `(P+1)^L` becomes too large.

## Open questions to settle during implementation
1. ✅ *Settled in Phase 4: `(2w + v)/3`.* **Moment-matched mean λ.**
   - The mean of the triangular density `h(y) ∝ (y − v)` on `(v, w]` is `(2w + v)/3`, not the printed `½(2w + v)`.
   - Derive it in a test and check numerically.
2. ✅ *Settled in Phase 4: exact cell-local decomposition by default; the literal form undercounts.* **Form of EIPV.**
   - As written, eq. 3/5 credit only the box `[v_C, y]` inside the cell where `y` lands.
   - The true gain (paper Fig. 1b) also includes the parts of other non-dominated cells below and to the left of `y`. For any cell whose upper corner `w_C ≤ y`, that cell's full volume is gained.
   - Compare `ceipv` and `ieipv` against `eipv_mc`.
   - If they differ, implement the exact gain per cell: `Π_l (min(y_l, w_{C,l}) − v_{C,l})₊`. Its integral still factorizes for independent objectives, and the CEIPV moment-matching applies to a piecewise `h`.
   - Record which form we use and why.
3. **Choice of `w_ref`.**
   - The approximation needs finite cells.
   - Decide per problem, and decide how sensitive results are to the choice.
4. **Test-function definitions.**
   - Confirm `oka2`, `vlmop3` and `dtlz1a` against the paper's supplement if we can get it.
