# NAF (MetaBO) implementation plan

Status: **first round done for Branin** (2026-10-12): Phases 0–4, a Branin benchmark (Phase 5, archived in `benchmark_results/naf_branin/`) the README user guide (Phase 6, README §10; results in §12) and a WebSocket server (`server_optimizers/naf_server.py`, port 8771, README §5 and §10.5). Training was stopped at 236 of 300 iterations (best checkpoint: iteration 193). NAF beats TAF-ME-50/TAF-R-50 from about 10 evaluations on, but repeats points after about 8 evaluations and plateaus; EI wins at 30. **Update 2026-10-15:** resumed to the reference's 2000 iterations (one run, one seed), the median log10 regret reaches −5.62 and beats EI from evaluation 10 on (question 12). Handed over for the next round: see "Handoff" below and open questions 10–12.

## Handoff (2026-10-12)

What works: everything for the Branin family, in-process, in the benchmark and through the server (`scripts/train_naf.py`, `run_naf_benchmark.py`, `plot_naf_results.py`; 45 tests in `tests/naf_*_test.py`). What is not done, in suggested order:

1. ✅ **Repeated points (open question 10).** Settled 2026-10-12: the reference does nothing special with repeats, and the training slowdown came from denormal floats in the PPO update, not from the GP. Fixed; see question 10.
2. **Longer training (open question 12)**, done once: 313 iterations on 2026-10-12, then 2000 iterations on 2026-10-15 (`meta-bo-training/naf/branin_metabo50_v2_2000`; see question 12). What is left is **repeating it with other seeds**, because the 2000-iteration result is a single training run. Train into a new `--run-id` and benchmark with a new `--test-id`; do not overwrite the archive.
3. **Goldstein-Price and Hartmann-3** (Phase 5), code done 2026-10-15 and both archived (`benchmark_results/naf_goldstein_price/` and `naf_hartmann3/`, see "Goldstein-Price run", "Hartmann-3 run" and "Hartmann-3 at 2000 iterations" under Phase 5). Goldstein-Price (600 iterations) looks like the paper. Hartmann-3 did **not** at 600 iterations (NAF stalled) and **does** at 2000 (best checkpoint; the stall was training length). Server configs were added 2026-10-17 (`configs/server_optimizers/naf_server_goldstein_price.yaml` and `naf_server_hartmann3.yaml`, each for its archived AF, `y_range` [−3.5, 4.5] and [−1.5, 4.5] from the values of 3050 tasks plus the corners: [−2.62, 3.44] and [−1.08, 3.39]); a full session ran with each model through the real server, and `tests/naf_server_test.py` checks the configs. Goldstein-Price was continued to 2000 iterations on 2026-10-17 and did **not** improve (see "Goldstein-Price at 2000 iterations" under Phase 5). Left: repeat the long runs with other seeds, find out why Goldstein-Price does not improve beyond 600 iterations, and look into the late instability of the Hartmann-3 and Branin training curves.

Rules that still hold: follow the reference code where it and the paper disagree, and never copy it (AGPL-3.0). Record every decision, deviation and measurement in this file.

Reference: M. Volpp, L. P. Fröhlich, K. Fischer, A. Doerr, S. Falkner, F. Hutter and C. Daniel, [*Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization*](https://arxiv.org/abs/1904.02642), ICLR 2020 (MetaBO). The main paper and appendix B (implementation details, Table 4) have been read. The authors' code at [boschresearch/MetaBO](https://github.com/boschresearch/MetaBO) has been read (not run); its findings are below. It is **AGPL-3.0**, so OpenBO reimplements from the paper and these notes and copies no code.

## Rules decided 2026-10-11

- **When the MetaBO paper and its reference code disagree, follow the code.** The code produced the paper's results and settles details the paper leaves vague. Each such case is marked in "Reference code findings", with the paper's version as an option where it is cheap to offer (for example a Sobol global grid).
- **Branin first.** Phases 1–5 run on Branin only; Goldstein-Price and Hartmann-3 come once Branin works.
- **Features follow the code** (open questions 2 and 9): raw `μ, σ, x, t, T`, with Branin's output normalized as in the reference, inside the NAF family.

## Decisions

| Topic | Decision |
| --- | --- |
| Scope of first pass | Reproduce the paper's global-benchmark experiment (Fig. 2) on **Branin** first, with random translations and scalings, budget T = 30, against EI, random search and TAF. Goldstein-Price and Hartmann-3 follow. Single-objective, sequential, synthetic only. |
| Neural AF | A torch MLP, 4 × 200 ReLU, on raw per-point features `[μ_t(x), σ_t(x), x, t, T]` (paper eq. 4, code), one logit per point, weights initialized N(0, 0.01²). It lives in `acquisition/naf.py`, with NumPy at its boundaries, like `PopulationBNN`. |
| RL algorithm | PPO, written from scratch in torch (about 200 lines). No new dependency (no gym or stable-baselines); the episode loop is small and specific to BO. |
| GP surrogate | `GPScratch(kernel_type="rbf", standardize_targets=False)` with **fixed hyperparameters per family**, fitted offline by type-2 maximum likelihood, as in the code (zero mean, no output standardization, near-zero noise). No refitting inside episodes, which also keeps training cheap. |
| Action space | Categorical over `ξ_t = ξ_global ∪ ξ_local,t`, as in the code: a static regular 31 × 31 grid plus the k = 5 best points of local Sobol grids around the AF's 5 best grid points. Test time uses the argmax over the same set. A Sobol global grid (the paper's text) is an option. |
| Reward | `r_t = −log10 R_t` for families with known optima (all of the first pass). Optima per variant come from `families.estimate_optimum`, because translations can move the optimum to the boundary. |
| Training tasks | "MetaBO-50" first, as in the released code: 50 fixed variants on a 3-D Sobol grid over `(t₁, t₂, s)`, the same set that TAF-50 uses as sources. "Full MetaBO" (a new random variant every episode) is a config switch. |
| TAF baselines | **TAF+ with one objective and no decay** (`acquisition/taf_plus.py`, `taf_plus_acquisition` with `L = 1`, `decay=1`). That is exactly the reference's TAF-ME (`weight_mode="poe"`) and TAF-R (`weight_mode="taf_r"`): source improvement over the max of the source mean at the target's iterates, clipped at 0. Use it with the NAF loop's fixed-hyperparameter GPs and grid search, not `TAFPlusSequentialOptimizer`, whose GPs refit and whose search differs. OpenBO's older `taf.py` is not used. |
| Code layout | `acquisition/naf.py` (neural AF, features, adaptive set, hierarchical maximizer), `optimizers/naf.py` (`NAFConfig`, PPO trainer, ask/tell `NAFSequentialOptimizer`, `run_naf`), `benchmarks/naf_runner.py`, `scripts/train_naf.py`, `run_naf_benchmark.py`, `plot_naf_results.py`. Trained AFs are saved as `.pt` plus a JSON config under `meta-bo-training/naf/<run_id>/`. |

## Paper summary

**Neural AF**
- Replaces EI inside an otherwise standard BO loop: `α_θ(x) = MLP[μ_t(x), σ_t(x), x, t, T]`. With `x` as an input, the AF can learn where optima of the family tend to be. Without `x` (the GP-prior experiments), it becomes dimension-agnostic.
- 4 hidden layers × 200 units, ReLU (Table 4).

**RL formulation (Table 1)**
- Episode = one BO run on a function drawn from the training family; episode length = budget T.
- State = `[μ_t(ξ_n), σ_t(ξ_n), ξ_n, t, T]` for all `ξ_n ∈ ξ_t`. Action = the chosen point. Policy = `Cat[α_θ(ξ_1), …, α_θ(ξ_N)]` (logits).
- `ξ_t`: AF evaluated on a static Sobol grid ξ_global (N_MS points), then local maximizations from the k = 5 best points give ξ_local,t.
- Reward: `−R_t` in general, `−log10 R_t` when the optimum is known (the benchmark functions and HPO).
- Value function `V(t, T)` only: a separate 4 × 200 MLP that sees only the step and budget.

**PPO settings (Table 4)**: batch 1200 steps, 4 epochs, 20 minibatches, Adam lr 1e-4, clip 0.15, value coefficient 1.0, entropy coefficient 0.01, γ = 0.98, GAE λ = 0.98.

**AF maximization (B.1)**: hierarchical gridding everywhere, in training and in evaluation. A Sobol grid of N_MS points (1000 for Branin and Goldstein-Price, 2000 for Hartmann-3), then k = 5 local Sobol grids of N_LS = N_MS points, each about one cell of the coarse grid, centred on the 5 best points. The maximum over those is the AF maximum.

**GPs (B.1)**: squared-exponential ARD with Gaussian noise; hyperparameters fitted offline by type-2 ML on the training and test functions and kept fixed. TAF's source GPs use the same hyperparameters. EI starts from the domain midpoint as its only initial design; TAF and MetaBO use no initial design.

**Benchmark protocol (Sec. 5)**
- Families: translations uniform in `[−0.1, 0.1]^D`, scalings uniform in `[0.9, 1.1]`.
- Baselines: EI, random, TAF-R-M and TAF-ME-M with M = 50 source tasks of N_TAF = 100 points each (M = 20 with N = 50 in the appendix). PI and GP-UCB were tried and lost to EI.
- Metric: median simple regret over 100 unseen test functions, with 30%/70% percentiles, as a function of t.
- Result: MetaBO beats EI by a wide margin early on and beats TAF over most of the budget. MetaBO spends a few early, non-greedy evaluations to identify the instance.
- Cost: an episode of T = 30 takes about 0.6 s for MetaBO and 0.13 s for EI, versus 13–60 s for TAF. Training took 30 min to 10 h on 10 CPU workers and 1 GPU. On Branin, a small number of source tasks is already enough (App. A.2).

## Reference code findings (boschresearch/MetaBO, Branin setup)

Where the code settles a detail the paper leaves open. "Paper" deviations are marked.

| Topic | What the code does |
| --- | --- |
| Features | Per point `[μ, σ, x₁, x₂, t, T]`, all **raw**: σ = sqrt of the noiseless GP variance, `t` = evaluations so far (from 0), `T` = 30. No `t/T`, no standardization. Empty GP: μ = 0, σ = sqrt(kernel variance). |
| Policy | MLP 4 × 200 ReLU, linear output, one logit per point. All weights initialized N(0, 0.01²), biases 0, so the initial policy is close to uniform. Training samples from `Categorical(logits)`; evaluation takes the argmax. |
| Value net | Separate MLP 4 × 200 on `[t, T]` only; one Adam optimizer for both networks. |
| Adaptive set | Global part: a fixed **regular grid** of 31 × 31 = 961 points (not Sobol, unlike the paper's text), built once. Local part: the 5 best grid points by the policy's own logits, a fixed 1000-point Sobol set scaled into a cube of side 2/31 around each, and the top 5 of those 5000 points. ξ_t = 5 local + 961 global, recomputed every step. No gradient-based search. EI and TAF use the same procedure in evaluation. |
| Reward | Every step: `−log10(max(1e-20, R_t))` in training. Evaluation records raw regret. |
| PPO | 2000 iterations × 1200 steps (10 workers × 4 episodes), 4 epochs, minibatch 60, Adam 1e-4, clip 0.15, value coefficient 1.0 (plain MSE, no value clipping), entropy 0.01, γ = λ = 0.98, advantages normalized per minibatch, no gradient clipping or LR schedule. The evaluated checkpoint is the one with the **best training reward** (iteration 1635). |
| GP | GPy, ARD RBF, fixed hyperparameters (Branin: lengthscales 0.235 and 0.578, variance 2.0, noise 8.9e-16, from type-2 ML on 100 points), zero mean, **no output standardization**. No initial design for MetaBO and TAF; EI gets one Sobol point. |
| Branin family | Base Branin on [0,1]² with output normalized as `−(f − 54.44)/51.44`. Variant `s · f(x − t)`: translation in **normalized** coordinates, `t ~ U(−0.1, 0.1)²`; scaling multiplies the **output**, `s ~ U(0.9, 1.1)`. The optimum is tracked analytically from the upper-left Branin minimum. |
| Training tasks | The released script trains on **M = 50 fixed tasks** (a 3-D Sobol grid over `(t₁, t₂, s)`), one drawn per episode; the evaluated weights are these "MetaBO-50" ones. Evaluation samples 100 fresh variants, seeds 100–109. |
| TAF baselines | Same 50 tasks as sources, N = 100 Sobol points each, same fixed GP hyperparameters. Source improvement `max(μ_j(x) − y_j,max, 0)` with `y_j,max` = **max of source j's mean over the target's evaluated points** (paper eq. 3); target term = analytic EI. TAF-ME weights: `σ_j(x)⁻² / (M+1)` per point and per model. TAF-R: Epanechnikov kernel (ρ = 1) on the distance between pairwise-ranking vectors of each model on the target points; the target always gets 0.75. |
| Quirks | With no data, the incumbent for EI and TAF is the objective's (transformed) corner value, which is privileged information. TAF-ME's `σ_target⁻²` with noise 9e-16 can blow up at sampled points. |

## Mapping onto OpenBO

| Paper | OpenBO |
| --- | --- |
| Branin, Goldstein-Price, Hartmann-3 | Self-contained in `test_functions/metabo_families.py` (not in `synthetic.py` or `REGISTRY`): the reference normalizes each function differently from OpenBO's textbook versions, so the NAF versions are their own functions. Selected through `NAF_FAMILY_SPECS` in `optimizers/naf.py`. |
| Translations and scalings | `TaskVariantSpec` (input shift and scale in normalized coordinates, output scale) and `generate_variants`. Check that "scaling" means output scaling (open question 3). |
| GP with fixed hyperparameters | `GPScratch(optimize_hyperparameters=False)`. Needs a diagonal-only posterior first (Phase 0). |
| EI baseline | `ScratchSequentialOptimizer` / `run_bo_scratch` exists. Add a "midpoint initial design" option, or document the difference. |
| TAF baseline | `TAFSequentialOptimizer`. Two known deviations from the paper's eq. 3 (open question 5). |
| Family splits | `families.py` and `configs/family_splits/` for the fixed "-50" source and test sets. |

## Implementation phases

### Phase 0: Prerequisites
- ✅ *Done 2026-10-11:* `GPScratch.posterior` now computes only the variance diagonal (`variance − Σ v²`), pinned by `test_posterior_variance_matches_full_covariance_diagonal` against the full covariance (both kernels, with and without standardization). 6000 test points with 30 training points: about 18 ms.
- **Fast GP posterior** (original note). `GPScratch.posterior` builds the full `m × m` test covariance and keeps only its diagonal. Each MetaBO step evaluates the GP at about 6000 points (1000 global + 5 × 1000 local), which needs 290 MB and dominates the run time. For a stationary kernel the prior diagonal is just the signal variance, so the variance can be computed in `O(m n)` instead. This also speeds up ConBO's grid predictions, TAF and the plotting scripts. Pin with a test against the current implementation.
- `goldstein_price` and `hartmann3`, with tests against their known optima: moved to after Branin works.
- The NAF Branin family: the reference's output normalization `−(f − 54.44)/51.44` on the textbook Branin (in OpenBO's maximization form, `(y + 54.44)/51.44` with `y = −Branin`), as a wrapper inside the NAF family.

### Phase 1: Families and fixed GP hyperparameters ✅ done (Branin)
- Done 2026-10-11:
  - `test_functions/metabo_families.py`: `normalized_branin` (the reference's normalization, maximum about 1.0506), `BraninTask(translation, scaling)` (`s · g(x − t)`, with `argmax` and `optimum` tracked from Branin's upper-left minimizer), `sample_branin_tasks(n, seed)` (random tasks: test functions and full MetaBO) and `branin_grid_tasks(50)` (MetaBO-50: an unscrambled 3-D Sobol grid over `(t₁, t₂, s)`, skipping the corner point; the reference's exact grid is not reproduced).
  - `optimizers/naf.py`: `FamilyGPHyperparameters` (`new_gp()` gives a fixed-hyperparameter, zero-mean, unstandardized RBF `GPScratch`), `BRANIN_GP_HYPERPARAMETERS` and `fit_family_hyperparameters(task, dim, n_points, seed)`.
  - Tests: `tests/naf_family_test.py` (15).
- **Branin uses the reference's hyperparameters** (lengthscales 0.235 and 0.578, variance 2.0), by the code-wins rule. OpenBO's own type-2 ML fit on 100 Sobol points of normalized Branin finds a different, higher-likelihood optimum (lengthscales 0.32 and 2.3, variance 290; NLML −377 vs −294 for the reference values on the same points). The difference matters because the neural AF reads the raw posterior std: with variance 290 it would reach about 17 instead of 1.4.
- `fit_family_hyperparameters` fits **one** task (the reference fits one function's 100 points). Pooling several translated tasks under near-zero noise forced lengthscales of 0.09 and 0.18, since nearby points from different tasks disagree.
- The tracked optimum was checked against a dense grid plus L-BFGS-B on 250 tasks: never beaten.
- Original plan notes:
- `make_metabo_family(base, n, seed)`: variants with translations in `[−0.1, 0.1]^D` and scalings in `[0.9, 1.1]`, each with an estimated optimum.
- `fit_family_hyperparameters(base, n_functions, n_points, seed)`: type-2 ML on Sobol samples from several training variants, giving one shared `gp_state` per family. Stored next to the trained AF.
- Tests: the variants stay in the stated ranges; the optimum estimate is at least as good as the base optimum's mapped location; the fitted hyperparameters are stable across seeds.

### Phase 2: Neural AF and the adaptive set ✅ done
- Done 2026-10-11, in `acquisition/naf.py`, following the code:
  - `naf_features(mean, std, x, t, budget, use_x)`: raw `[μ, σ, x, t, T]`, `t` counting evaluations from 0.
  - `NeuralAF(dim, use_x, hidden=(200,)*4, seed)`: one logit per point; weights N(0, 0.01²), zero biases, so the initial logits are below 1e-3 (a near-uniform policy). **float32**, as the reference (PyTorch's default); `logits()` returns float64 NumPy. Seeded without touching the global torch RNG.
  - `AdaptiveSet.build(dim, points_per_dim=31, k=5, n_local=1000, local_side=2/31, seed)`: regular global grid; one fixed scrambled Sobol cloud, centred on each of the `k` best grid points and **clipped to the unit cube** (the reference's handling of the boundary is not recorded); `candidates(score)` returns the `k` best local points first, then the grid (966 points for 2D); `maximize(score)` is the test-time argmax.
  - `gp_mean_std` (the prior `(0, sqrt(variance))` before any data, as in the reference) and `naf_score_fn(af, gp, t, budget)`.
  - Tests: `tests/naf_acquisition_test.py` (7).
- **Measured cost (8-core CPU, no GPU).** One candidate-set step with a 29-point GP: 95 ms in float64 (72 ms of it the MLP on about 6000 points, 14 ms the GP), about half in float32. A forward plus backward pass over 40 steps × 966 candidates: 1.7 s (float64), 0.77 s (float32), on one thread.
  - **Implication for Phase 3.** One PPO iteration of the reference (1200 rollout steps, then 4 epochs × 20 minibatches of 60 steps, each step's loss over all 966 candidates) is roughly 10 s of rollouts on 7 workers plus about 90 s of updates on one thread (perhaps 30 s multithreaded). The reference ran about 2000 iterations (best checkpoint at 1635) on a GPU; on this CPU that is about a day. Options to decide before Phase 3: fewer iterations with the best-reward checkpoint, fewer threads per rollout worker and a multithreaded update, or a smaller training grid (a deviation).
- Original plan notes:
- `NeuralAF(dim, use_x=True, hidden=(200,)*4)`: `forward(features) -> logits`.
- `naf_features(gp, x, t, T, use_x)`: `[μ, σ, x, t, T]`, with μ and σ in the GP's standardized units (open question 4).
- `adaptive_set(af, gp, t, T, global_grid, k, n_local, rng)`: evaluate on the global grid, take the k best, build local Sobol grids around them, and return ξ_t (global grid plus the k local maxima).
- `maximize_naf(...)`: the hierarchical grid argmax used at test time.
- Tests: shapes; the local maxima are at least as good as their starting points; with `use_x=False` the AF is invariant to permuting the input dimensions.

### Phase 3: Episodes and PPO ✅ done
- Done 2026-10-11, in `optimizers/naf.py`:
  - `BOEpisode(task, gp_hyperparameters, adaptive_set, budget)`: `observation(af)` builds `ξ_t` with the current AF and returns features and logits; `step(index)` evaluates, refits the fixed-hyperparameter GP and returns `−log10(max(1e-20, R_t))`. No initial design.
  - `NAFConfig` (frozen): Table 4 and the code's choices as defaults (budget 30, 4 × 200, 31 × 31 grid, k = 5, 1000 local points, 2000 iterations of 1200 steps, 4 epochs, minibatch 60, Adam 1e-4, clip 0.15, value coefficient 1.0, entropy 0.01, γ = λ = 0.98), plus `n_workers` and `update_threads`.
  - `run_episodes` (sampled or greedy), `gae_advantages` (whole episodes, no bootstrap past the last step), `ppo_loss` (advantages normalized per minibatch, plain MSE value loss, entropy bonus), `train_naf(config, gp_hyperparameters, task_pool=... | task_sampler=..., checkpoint_dir, log)`. The value net is a separate 4 × 200 MLP on raw `[t, T]`; one Adam optimizer covers both nets. `best_af` is the iteration with the highest mean training return (the reference's choice). Rollouts run in worker processes with one torch thread each; results are independent of timing, but differ between `n_workers=1` and `> 1` (tasks and seeds are split per worker).
  - `save_naf`/`load_naf` (`<name>.pt` plus `naf_config.json`) and `save_training` (`af.pt`, `best_af.pt`, `training_history.json`), written after every iteration when `checkpoint_dir` is set. There is no resume yet (the Adam state is not saved).
  - Tests: `tests/naf_training_test.py` (7): GAE and PPO loss against hand computations, episode rewards, reproducibility and checkpoints, parallel rollouts, a task sampler, and a 1D family peaking at x = 0.8 that 300 short iterations (about 30 s) learn: the greedy first pick moves to within 0.05 of 0.8.
- **Measured on Branin with the reference settings** (8-core CPU, no GPU): about 57 s per iteration, 15 s of rollouts on 7 workers and 42 s of PPO update on 8 threads (103 s on one). The update is near the CPU's arithmetic limit (about 3.4 TFLOP per iteration). The reference's 2000 iterations would take about 32 hours.
- Original plan notes:
- `BOEpisode`: `reset(objective, optimum)`, `observation()` (features on ξ_t), `step(index) -> reward`, fixed-hyperparameter GP update after every step. No initial design.
- `NAFConfig` (frozen): Table 4 defaults plus `budget`, `n_iterations`, `n_workers`, `use_x`, `reward` (`"log_regret"` or `"regret"`), `task_pool` (`None` for full MetaBO, or a fixed list for MetaBO-50), `seed`.
- `train_naf(config, family) -> NAFTrainingResult`: rollouts (1200 steps = 40 episodes of T = 30), GAE with `V(t, T)`, PPO clipped loss with value and entropy terms, Adam. It logs the mean final log regret per iteration and saves checkpoints.
- Parallel rollouts with `multiprocessing` (one process per worker, each with its own seeded RNG and a CPU copy of the policy).
- Tests: the PPO loss matches a hand-computed toy batch; a 1D toy family where the optimum is always at the same `x`, which a short training run must learn (the regret after 1 step falls well below EI's); training is reproducible for a fixed seed and one worker.

### Phase 4: Optimizer and runner ✅ done
- Done 2026-10-11:
  - `optimizers/naf.py`: `NAFSequentialOptimizer(af, gp_hyperparameters, budget, adaptive_set)` (`from_directory(dir, name="best_af")`, greedy `suggest()`, `observe(x, y)`, stops at its budget, since `T` is an AF input) and `run_naf`.
  - `benchmarks/naf_runner.py`: `NAF_METHODS = (naf, ei, taf_me_50, taf_r_50, random)`, `naf_test_tasks` (100 unseen tasks, seed 100), `TAFSources` (50 MetaBO-50 source GPs, 100 Sobol points each, global-grid predictions cached), `run_naf_task` and `run_naf_method` (optional worker processes; identical results). All GPs use the family's fixed hyperparameters and all AFs are maximized over the same `AdaptiveSet`.
  - EI starts from one scrambled Sobol point (the code; the paper says the midpoint).
  - TAF is `taf_plus_acquisition` with one objective and no decay. **Deviation:** with no data, the reference's TAF uses the task's own corner value as incumbent (privileged information); OpenBO's first TAF point maximizes the sources' product-of-experts mean instead, as in TAF+.
  - `scripts/run_naf_benchmark.py` (one method → `test_results/naf/trajectories/<test_id>_<method>_<family>.json`) and `scripts/plot_naf_results.py` (median log10 regret with 30–70% bands, a table, and with `--training-dir` the training curve).
  - Tests: `tests/naf_runner_test.py` (9).
- Original plan notes:
- `NAFSequentialOptimizer(af_path or af, gp_state, budget)`: ask/tell with `suggest()` / `observe(x, y)`; `run_naf(objective, ...)`.
- `benchmarks/naf_runner.py`: `NAF_METHODS = (naf, naf_50, ei, random, taf_r_50, taf_m_50)`. Each method sees the same 100 test variants per family.
- `scripts/train_naf.py`, `scripts/run_naf_benchmark.py`, `scripts/plot_naf_results.py`, following the ConBO and MOBO scripts (median and 30/70 percentiles of log regret per step).

### Phase 5: Benchmarks (Branin done; other families open)
- **First Branin run (2026-10-12, test id `v1`, 100 unseen tasks, test seed 100, budget 30).**
  - Training: `scripts/train_naf.py --n-iterations 300 --run-id branin_metabo50_v1` (MetaBO-50, 7 workers). Mean final log10 regret went from −1.67 (iteration 0) to about −3.4 (iterations 180–209), flat between iterations 100 and 160, then slightly worse. Stopped at iteration 236 (user's call; the best-return checkpoint, iteration 193, return 90.6, is what the benchmark uses).
  - Iterations slowed from about 52 s to 140 s as the policy improved, with no other load on the machine. The guess at the time (the GP's Cholesky retrying on repeated points) was wrong: the cause was denormal floats in the PPO update, fixed 2026-10-12 (question 10).
  - Median log10 simple regret:

    | method | @1 | @3 | @5 | @10 | @15 | @20 | @30 |
    |---|---|---|---|---|---|---|---|
    | NAF | −0.66 | −1.35 | −1.60 | **−3.31** | **−3.35** | −3.36 | −3.36 |
    | TAF-ME-50 | **−1.19** | −1.74 | **−1.93** | −2.26 | −2.36 | −2.48 | −2.86 |
    | TAF-R-50 | **−1.19** | **−1.77** | −1.80 | −1.95 | −2.20 | −2.54 | −2.97 |
    | EI | −0.15 | −0.65 | −0.87 | −1.53 | −2.76 | **−3.37** | **−4.98** |
    | Random | −0.19 | −0.75 | −0.86 | −1.13 | −1.25 | −1.31 | −1.58 |

  - Mean log10 regret over evaluations 1–10 / 1–15 / 1–30: NAF −1.92 / **−2.28** / −2.67, TAF-ME −1.96 / −2.10 / −2.39, TAF-R −1.84 / −1.95 / −2.37, EI −0.98 / −1.46 / **−2.77**. NAF has a better best-so-far value than TAF-R on 11% of tasks at step 1, 77% at step 10 and 64% at step 30.
  - **NAF probes first.** Its first point is the corner (1, 0) on every task (not near an optimum); its second point differs per task (88 distinct of 100). TAF's first point is an optimum of the averaged sources, (0.546, 0.154). The reward sums log regret over all 30 steps, so a probe that helps later steps beats a good first guess.
  - **NAF repeats points.** In evaluations 11–30, 24% of NAF's points exactly repeat an earlier one and 46% are within 1e-3 of one (EI: 3% and 3%). This is the plateau after evaluation 8.
  - Cost: NAF 38 s, EI 8 s, random <1 s, TAF-ME 38 min and TAF-R 46 min for 100 tasks on 8 workers. TAF scores the 5000 local points against 50 source GPs every step (about 3 s/step). With 8 workers each running a multithreaded BLAS, the first TAF-ME attempt oversubscribed the CPU (load 57) and ran for an hour without finishing; `_task_worker` and `_rollout_worker` now run under `threadpool_limits(1)`.
- Test 1, Branin (cheapest; App. A.2 says few source tasks are enough). Then Goldstein-Price and Hartmann-3.
- **Goldstein-Price and Hartmann-3: settings (2026-10-15, code done).** Read from the reference's `objectives.py` and `metabo_gym.py` (read only, nothing copied) and the paper's Table 4:
  - **Functions.** Goldstein-Price on `[-2, 2]^2` mapped from the unit square, log-normalized as `-(log f − 8.693) / 2.427`; Hartmann-3 on `[0, 1]^3`, normalized as `-(f + 0.93) / 0.95`. Both come out with mean about 0 and standard deviation about 1 on uniform points (checked: −0.004 and 1.004; 0.019 and 1.008). Tasks are `s · g(x − t)` with `t ~ U(−0.1, 0.1)^D` and `s ~ U(0.9, 1.1)`, MetaBO-50 a `(D+1)`-dimensional Sobol grid, exactly as for Branin. The reference clips each translation to a range that keeps the optimum inside the domain (Goldstein-Price `[−0.5, 0.5] × [−0.25, 0.75]`, Hartmann-3 `[−0.11, 0.88] × [−0.55, 0.44] × [−0.85, 0.14]`); the translations here never reach those ranges, so OpenBO does not clip.
  - **Optima.** Goldstein-Price: textbook minimum 3 at `(0, −1)`, which is `(0.5, 0.25)` on the unit square. Hartmann-3: `(0.114589, 0.555649, 0.852547)`, polished to machine precision (the reference uses the rounded `0.114614 ...`). The tracked `task.optimum` was compared with a dense search plus local polish on 40 tasks per family: nothing exceeds it (largest excess 4e-16).
  - **Match with the reference.** OpenBO's normalized functions and translated, scaled tasks agree with the reference's `gprice`, `hm3`, `gprice_var` and `hm3_var` to 9e-16 on random points (a scratch comparison; the reference is not in the repo).
  - **Candidate set (Table 4 and the code).** `N_MS` = 1000 for Goldstein-Price and 2000 for Hartmann-3, `N_LS = N_MS`, `k = 5`. The code builds a regular grid of `floor(N_MS^(1/D))` points per dimension: 31 × 31 = 961 (+5) for Goldstein-Price, 12³ = 1728 (+5) for Hartmann-3 (the paper's 2000 is not reached exactly), and each local cloud is a cube of side `2 / points_per_dim`. `NAFFamily` encodes this.
  - **GP hyperparameters (a decision made on a wrong premise, corrected 2026-10-16).** I first wrote that the reference ships fixed values only for Branin. That was wrong: I had downloaded only the Branin script, and the repository also has `train_metabo_gprice.py` and `train_metabo_hm3.py` with the authors' values (Goldstein-Price lengthscales 0.130 and 0.07, variance 0.616, noise 1e-6; Hartmann-3 0.716, 0.298 and 0.186, variance 0.83, noise 1.688e-11). The fits below are close to them. What follows is what was done at the time: the paper says the values come from type-2 ML on functions of the family. OpenBO's `fit_family_hyperparameters` (100 Sobol points per task, 32 restarts) was run on 10 random tasks per family and the **median** kept: Goldstein-Price lengthscales (0.109, 0.087), variance 0.652; Hartmann-3 (0.709, 0.305, 0.173), variance 0.694. Across the 10 tasks the variances stay between 0.51 and 0.93 and between 0.52 and 0.89, and the lengthscales within about 25%, unlike Branin, where the same fit jumps between variances of 1 and 400 from task to task (so Branin's reference values cannot be reproduced by refitting). The short Goldstein-Price lengthscales reflect its sharp peak.
  - **A first guess about TAF (checked later, see the Goldstein-Price run).** With 100 points per source GP at lengthscale about 0.1, the 50 sources may interpolate Goldstein-Price poorly: on three test tasks the first TAF point (the sources' product-of-experts maximum) had regret 1.1 to 1.75, against 0.1 to 0.85 on Hartmann-3. The 100-task benchmark gave a median of about 1.1 for that first point, so this was a small sample read too strongly.
  - **Cost (8 cores, alone).** About 9 s per PPO iteration for Goldstein-Price (2000 iterations about 5 h) and about 17 s for Hartmann-3 (about 9.5 h).
  - **Checks.** `tests/naf_more_families_test.py` (36 tests): textbook values, normalization scale, tracked optima, task ranges and reproducibility, candidate-set sizes, GP quality on a task, every method on both families, the optimizer, and one PPO iteration on Hartmann-3. Changing a normalization constant or the Goldstein-Price argmax makes it fail. Branin is unchanged: EI, random and TAF-R trajectories on 6 tasks are identical before and after, on the same machine (the archive itself does not reproduce bit for bit across machines, a difference of about 5e-6 in the EI values). The server was run with an untrained 3D AF: a full session works and a 2D client is refused with a clear message.
- ✅ Branin archived in `benchmark_results/naf_branin/` (README, `run_all.sh`, trained model, trajectories, plots). Archive the other families the same way.
- **Goldstein-Price run (2026-10-15, test id `v1`, 100 unseen tasks, test seed 100, budget 30).** Archived in `benchmark_results/naf_goldstein_price/` (README, `run_all.sh`, model, trajectories, plots).
  - Training: `scripts/train_naf.py --family goldstein_price --n-iterations 600`, MetaBO-50, seed 0, 7 workers, 8.6 s per iteration (1.4 h). Best return 55.5 at iteration 499. The mean final log10 regret fell from about 0 to −2.1 in 80 iterations, stayed at −2.1 to −2.2 until about iteration 450, then dropped to about −2.4 and was still falling at 600.
  - Median log10 simple regret:

    | method | @1 | @3 | @5 | @10 | @15 | @20 | @30 | mean 1–30 |
    |---|---|---|---|---|---|---|---|---|
    | NAF | 0.56 | 0.07 | **−0.50** | **−1.59** | **−1.99** | **−2.17** | **−2.37** | **−1.72** |
    | TAF-ME-50 | **0.05** | −0.13 | −0.27 | −0.40 | −1.29 | −1.56 | −1.68 | −1.13 |
    | TAF-R-50 | **0.05** | **−0.15** | −0.30 | −0.46 | −0.99 | −1.14 | −1.32 | −0.92 |
    | EI | 0.52 | 0.37 | 0.26 | 0.03 | −0.02 | −0.12 | −1.28 | −0.40 |
    | Random | 0.49 | 0.37 | 0.28 | 0.19 | 0.12 | 0.11 | 0.04 | 0.11 |

  - NAF has a better best-so-far value than EI on 94% / 93% / 85% of tasks at evaluations 5 / 10 / 30, than TAF-R on 52% / 79% / 82%, and than TAF-ME on 52% / 87% / 65%. It is behind both TAFs until evaluation 4 and passes them at 5.
  - **NAF probes first**: its first point is (0.702, 0.0) on every task (median first-step regret 3.6); its second point differs per task (34 distinct of 100). **Repeats are rarer than on Branin**: in evaluations 11–30, 13% of NAF's points exactly repeat an earlier one and 14% are within 1e-3 (EI: 0% and 0%; the Branin run: 24% and 46%).
  - **EI is slow here**: median regret about 1 until evaluation 20, then −1.28 at 30, because the peak is sharp. It does not catch up as on Branin.
  - **Against the paper's Figure 2(b)**, read by eye from a log-scale figure (so only approximate): MetaBO-50 about −2.5, TAF-ME about −2, TAF-R about −1.3, EI about −1, random about 0 at evaluation 30; this run: −2.37, −1.68, −1.32, −1.28, 0.04. Close.
  - **A misreading, corrected.** Three test tasks had shown TAF's first-point regret as 1.1 to 1.75 on this family, and I called that far from the optimum. Over 100 tasks the median is log10 0.05, which is a regret of about 1.1: the same size, not a better one (I first misread the log value as an improvement). Whether that is "far" depends on the reference point: the first point of EI and random has a regret of about 3.3 to 3.6, so TAF's is about three times closer, and the function's normalized range is about 5.2. It is not a bug in the baseline.
  - **Limits.** One run of 600 of the reference's roughly 2000 iterations, one seed, no confidence interval. The training curve looked as if it was still falling at 600 iterations; it was not (see "Goldstein-Price at 2000 iterations"). Both TAFs improve sharply around evaluations 10–13; not investigated.
- **Goldstein-Price at 2000 iterations (2026-10-17): no improvement.** The 600-iteration run was resumed to 2000 (1400 more iterations, 3.3 h at 8.4 s, uninterrupted). Archived in `benchmark_results/naf_goldstein_price/` (results `v2`; only `af.pt`, the config, the history, the log and the training plot are kept, since `best_af.pt` is unchanged).
  - **The best-return checkpoint is still iteration 499** (55.5), byte-identical to the 600-iteration model; in iterations 600–1999 the best return was 48.9 (iteration 1836). So the `v1` benchmark above is also the benchmark of the 2000-iteration run's best checkpoint.
  - **The last iteration's model is worse.** Median log10 regret on the 100 test tasks, 600-iteration best checkpoint against iteration 1999: @1 0.56 / 0.67, @5 −0.50 / 0.40, @10 −1.59 / −0.36, @15 −1.99 / −1.51, @20 −2.17 / −1.75, @30 −2.37 / −1.97; mean over 30 evaluations −1.72 / −1.09; repeats 13% / 1%. The later model beats the earlier on 1% / 12% / 33% / 41% of tasks at evaluations 5 / 10 / 20 / 30, and is about as good as TAF-ME on the mean (−1.09 against −1.13).
  - **The training curve is flat after the first 80 iterations**: mean return per 200 iterations 25.7, 36.3, 44.2, 36.2, 35.1, 37.3, 38.3, 38.7, 40.9, 38.2; mean final log10 regret −2.1 to −2.4 throughout. The peak around iterations 450–550 (50-iteration mean return 45–47) faded by iteration 650. **Correction of an earlier reading:** the 600-iteration curve looked as if it was still falling, and I wrote that longer training would probably help; it did not. A checkpoint chosen as the maximum of 2000 noisy returns is partly a lucky draw (the best single iteration, 55.5, is well above the 45–47 mean around it), which the independent test tasks do not hide but which can make "best checkpoint" look better than the typical iteration.
  - **What it means.** At 600 iterations NAF is already at the paper's level on Goldstein-Price (−2.37 against about −2.5 by eye) and more training with this setup does not move it; Branin and Hartmann-3 did improve a lot. Why is not known (the Branin curve had long plateaus too; the family's sharp peak and the candidate set's floor of −3.88 are possible reasons, not tested). One run, one seed.
- **Hartmann-3 run (2026-10-15, test id `v1`, 100 unseen tasks, test seed 100, budget 30).** Archived in `benchmark_results/naf_hartmann3/` (README, `run_all.sh`, model, trajectories, plots).
  - Training: `scripts/train_naf.py --family hartmann3 --n-iterations 600`, MetaBO-50, seed 0, 7 workers, 16.2 s per iteration (2.7 h). Best return 58.0 at iteration 591. Mean return per 100 iterations: 32, 43, 49, 52, 54, 54 (nearly flat at the end); mean final log10 regret fell to about −1.8 by iteration 40, stayed there until about 160, and reached about −2.25 after 400.
  - Median log10 simple regret:

    | method | @1 | @3 | @5 | @10 | @15 | @20 | @30 | mean 1–30 |
    |---|---|---|---|---|---|---|---|---|
    | NAF | 0.56 | −0.07 | −0.81 | **−2.03** | **−2.12** | −2.12 | −2.12 | −1.72 |
    | TAF-ME-50 | **−0.46** | −0.85 | −1.11 | −1.81 | −2.06 | **−2.64** | **−3.06** | **−2.10** |
    | TAF-R-50 | **−0.46** | **−0.96** | **−1.29** | −1.44 | −1.61 | −1.80 | −2.05 | −1.63 |
    | EI | 0.53 | 0.50 | 0.48 | −0.18 | −1.33 | −2.05 | −2.61 | −1.23 |
    | Random | 0.55 | 0.41 | 0.30 | 0.07 | −0.04 | −0.13 | −0.24 | −0.03 |

  - **At 600 iterations this does not reproduce the paper** (2000 iterations do, see below). NAF is the best method only at evaluations 10 and 15; its median is flat at −2.12 from evaluation 14. TAF-ME passes it at 17 and EI at 22. NAF has a better best-so-far value than TAF-ME on 57% / 33% / 14% of tasks at evaluations 10 / 20 / 30, than EI on 89% / 56% / 22%, and than TAF-R on 82% / 69% / 51%.
  - **NAF stops exploring**: in evaluations 11–30, 86% of its points exactly repeat an earlier one (EI and TAF-ME: 0%; Branin 24%, Goldstein-Price 13%), and for 86% of the tasks the best value at 30 equals that at 12. Its first point is the corner (0, 1, 1) on every task (median regret 3.6); its second point takes only 14 distinct values over 100 tasks (Branin 88, Goldstein-Price 34).
  - **Against the paper's Figure 2(c)**, read by eye (approximate): MetaBO-50 and EI both near −2.4 at 30 and TAF-ME near −2.2, MetaBO best. Here NAF is −2.12, EI −2.61, TAF-ME −3.06.
  - **Candidate causes (first listed 2026-10-15, tested 2026-10-16; see "Hartmann-3 diagnosis" below).** (1) 600 iterations are too few for 3D. (2) The GP hyperparameters are OpenBO's own fit. (3) The candidate set (`N_MS` 2000 in the paper, 12³ = 1728 grid points by the code's rule).
  - **Cost.** NAF, EI and random take seconds; TAF-ME and TAF-R about 11 minutes each on 8 workers (Goldstein-Price: about 5).
  - **Limits.** One run of 600 of the reference's roughly 2000 iterations, one seed, no confidence interval.
- **Hartmann-3 diagnosis (2026-10-16).** Cheap tests on the archived 600-iteration model and the archived trajectories, no retraining. The scripts were scratch files (not in the repo); the numbers below are from 100 test tasks (test seed 100). The greedy runs reproduce the archive exactly (Branin −5.62, Goldstein-Price −2.37, Hartmann-3 −2.12 at 30), so the setup is the same.
  - **Candidate-set resolution: ruled out.** The best value reachable from the local clouds around the 5 grid points nearest the true optimum has a median log10 regret of −3.59 for Hartmann-3 (Goldstein-Price −3.88, Branin −5.56); NAF stops at −2.12, TAF-ME reaches −3.06. For Branin the floor (−5.56) **is** the flat −5.62 of the 2000-iteration run: that plateau is the candidate set's resolution, not the policy's limit, and probably the paper's ≈1e-5 plateau too. For Goldstein-Price and Hartmann-3 the floor is far below what NAF reaches.
  - **Greedy versus sampled actions: not the cause.** Sampling from the AF's softmax (as in training) instead of the argmax gives Hartmann-3 −2.15 at 30 (greedy −2.12), mean −1.77 (−1.72), with 68% exact repeats in evaluations 11–30 (86%). The reference also evaluates with the argmax, so OpenBO's protocol matches. For Branin sampling is slightly worse (−5.15, 4% repeats) and for Goldstein-Price about equal.
  - **No train/test gap.** Mean final log10 regret on the test tasks (greedy) is −2.18; over the last 100 training iterations (sampled, training tasks) it is −2.24.
  - **What NAF does.** It evaluates about 12 distinct points (median 13, range 10–17 of 30), then evaluates one point over and over for the remaining ~18 evaluations; once it does, the GP and the features no longer change, so the choice repeats (an absorbing state). The repeated point is usually **not** the incumbent (only 8% of evaluations 11–30 re-evaluate the best point so far; EI and TAF-ME: 0%). Its best point is at a median distance of 0.072 from the true optimum (TAF-ME 0.023, EI 0.031); the regret (0.0075) is far smaller than the gap to the second local maximum (about 0.15), so it is in the right basin but not refined.
  - **Banning repeats helps little.** If the already evaluated points may not be chosen again, Hartmann-3 only improves from −2.12 to −2.20 at 30 (Goldstein-Price −2.37 to −2.38, Branin −5.62 to −5.63). The AF's ranking among unvisited points does not drive refinement either; the absorbing state alone does not explain the stall.
  - **GP hyperparameters: they matter, but cannot explain it alone.** Scaling the lengthscales and variance of the Hartmann-3 GP (baselines only; NAF would have to be retrained): EI at 30 evaluations is −2.61 with our values, −2.52 with lengthscales ×0.5, −0.50 with ×2, −2.64 with variance ×0.25 and −2.21 with ×4. TAF-ME is −3.06 with our values, −2.87 with lengthscales ×0.5 and −2.27 with ×2. With ×2 TAF-ME matches the paper's TAF-ME (about −2.2, by eye), but then EI (−0.50) is far from the paper's (about −2.4); no single scaling matches both, so this is not the whole story.
  - **The authors' pretrained weights, run in OpenBO's environment (2026-10-16): the decisive test.** The reference repository ships trained MetaBO-50 weights (`metabo/iclr2020/gobfcts/<family>/M_50/...`, the checkpoints with the best training return: Branin iteration 1635, Goldstein-Price 1703, Hartmann-3 1988). Their policy network is a 4 × 200 ReLU MLP on `[μ, σ, x, t, T]` in the same order as OpenBO's, so the weights load layer by layer into `NeuralAF`. They were run in OpenBO's tasks, candidate set, GP and greedy evaluation (100 test tasks, test seed 100); the weights file was only loaded in a scratch directory and is not in the repo (AGPL-3.0). Median log10 regret:

    | family | GP hyperparameters | @1 | @5 | @10 | @15 | @30 | mean 1–30 | repeats 11–30 |
    |---|---|---|---|---|---|---|---|---|
    | Branin | reference's | −0.66 | −1.60 | −4.19 | −5.07 | **−5.07** | −3.53 | 49% |
    | Hartmann-3 | reference's | 0.33 | −0.77 | −2.69 | −3.13 | **−3.14** | −2.52 | 10% |
    | Hartmann-3 | OpenBO's own fit | 0.31 | −0.73 | −2.58 | −3.19 | **−3.21** | −2.51 | 10% |
    | Goldstein-Price | reference's | 0.64 | −0.32 | −1.57 | −2.29 | **−2.59** | −1.69 | 50% |
    | Goldstein-Price | OpenBO's own fit | 0.64 | −0.46 | −1.68 | −2.12 | **−2.45** | −1.74 | 18% |

    - **OpenBO's environment and evaluation are sound.** The authors' Branin weights reach −5.07, the paper's level (our own 2000-iteration Branin run reached −5.62).
    - **GP hyperparameters are not the cause.** The authors' Hartmann-3 AF is as good with OpenBO's fitted hyperparameters (−3.21) as with its own (−3.14).
    - **The Hartmann-3 gap is in OpenBO's training.** In the same environment the authors' AF gives −3.14 and 10% repeats at 30 evaluations; our 600-iteration AF gives −2.12 and 86% repeats. TAF-ME (−3.06) is about as good as the authors' AF here.
    - **Goldstein-Price is fine**: our 600-iteration AF (−2.37) is about as good as the authors' (−2.45 to −2.59).
    - **Repeats are not specific to our training**: the authors' Branin and Goldstein-Price AFs repeat 49% and 50% (18% with OpenBO's hyperparameters), as much as ours; what is special to our Hartmann-3 AF is that it repeats instead of refining.
    - Caveat: the authors' task sampling (a Sobol grid of 50 tasks, uniform test tasks) and their GPy-based environment differ in small ways from OpenBO's, and the AFs still work, so the AFs are robust to those. Only the best-return checkpoint of each was used.
  - **Remaining candidates for the Hartmann-3 gap (training only).** (1) Training length: 600 iterations against the authors' 1988; the training curve was still falling slowly at 600, and Branin needed about 2000. Not tested; `--resume --n-iterations 2000` takes about 6 more hours and is now the most likely explanation. (2) A detail of the training procedure that differs from the reference's (advantage normalization per minibatch, the candidate set used inside training, the checkpoint rule); Branin trained for 2000 iterations reaches the paper's level, so a large difference is unlikely. (3) Retraining with the authors' Hartmann-3 hyperparameters (0.716, 0.298, 0.186; variance 0.83; noise 1.688e-11) would make the comparison with their AF exact; OpenBO's fit is close and the authors' AF does not care, so this should matter little.
- **Hartmann-3 at 2000 iterations (2026-10-17).** The 600-iteration run was resumed to 2000 (`--resume --n-iterations 2000`, 1400 more iterations, 6.3 h at 16 s, uninterrupted). Archived as `benchmark_results/naf_hartmann3/training/hartmann3_metabo50_v2_2000` (results `v2`). Best return 79.2 at iteration 1454 (58.0 at 600). Median log10 simple regret, same 100 test tasks:

  | method | @1 | @3 | @5 | @10 | @15 | @20 | @30 | mean 1–30 | repeats 11–30 |
  |---|---|---|---|---|---|---|---|---|---|
  | NAF, 2000 its, checkpoint 1454 | 0.25 | 0.21 | −0.74 | **−2.61** | **−3.22** | **−3.26** | **−3.27** | **−2.61** | 56% |
  | NAF, 2000 its, last iteration 1999 | 0.56 | 0.24 | −0.19 | −0.76 | −2.18 | −3.26 | −3.30 | −1.91 | 25% |
  | NAF, 600 its | 0.56 | −0.07 | −0.81 | −2.03 | −2.12 | −2.12 | −2.12 | −1.72 | 86% |
  | TAF-ME-50 | **−0.46** | **−0.85** | −1.11 | −1.81 | −2.06 | −2.64 | −3.06 | −2.10 | 0% |
  | TAF-R-50 | **−0.46** | **−0.96** | **−1.29** | −1.44 | −1.61 | −1.80 | −2.05 | −1.63 | 0% |
  | EI | 0.53 | 0.50 | 0.48 | −0.18 | −1.33 | −2.05 | −2.61 | −1.23 | 0% |

  - **The stall at 600 iterations was training length.** The 2000-iteration checkpoint is the best method from evaluation 6 on and has the lowest mean regret over 30 evaluations; it beats EI on 85–96% of tasks from evaluation 5 on and TAF-ME on 83–95% at evaluations 10–20 (63% at 30), and its median at 30 (−3.27) is a little better than the authors' own Hartmann-3 weights in OpenBO's environment (−3.14). This confirms the diagnosis above: the environment, the evaluation and the GP hyperparameters were not the problem.
  - **Behaviour.** The first point is (0.0, 0.462, 1.0) on every task (the 600-iteration model used (0, 1, 1)); the second point takes 70 distinct values and the third 80 (14 for the second point at 600 iterations). A median of 18.5 distinct points per task (13 at 600). 56% of evaluations 11–30 are exact repeats and 27% of them re-evaluate the incumbent (8% at 600 iterations), so the lock-in is now mostly on the best point, after the search has converged. The median is flat at −3.27 from about evaluation 15, close to the candidate set's floor (−3.59).
  - **Training curve.** Mean final log10 regret per 200 iterations: −1.66, −2.06, −2.23, −2.34, −2.77, −3.02, −3.06, −3.07, −3.03, −3.04; mean return per 200 iterations: 37.7, 50.6, 53.8, 55.3, 61.6, 66.7, 68.5, 68.4, 62.1, 60.7. A step near iteration 800, the best around 1450 (about −3.3), then spikes near iterations 1520, 1620, 1760 and 1960 (single-iteration worst values −1.72 near 1510, −2.38 near 1624, −2.37 near 1766 and −1.23 near 1972, against −3.3 at the best) from which it recovered; the mean return falls from about 68 to about 61. The same shape as Branin's 2000-iteration curve (a best checkpoint just before a setback). The cause is not known; the last iteration's model is as good late (−3.30 at 30 evaluations) but worse early (−0.76 at 10), and the reward weights the early evaluations, which is why the best-return checkpoint is better on the mean.
  - **Limits.** One run, one seed, no confidence interval; the checkpoint is the best-return one of a curve with spikes, so this is the best case of the run. In the paper's Figure 2(c) (by eye, approximate) MetaBO and EI end near −2.4; here NAF is −3.27 and EI −2.61.
- Optional, after the paper reproduction: compare NAF with ConBO on the ConBO user sequences. The two use transfer differently (offline meta-training vs continual learning), and that comparison is not in either paper.

### Phase 6: Docs and server ✅ done
- ✅ README §11 (user guide: pieces, Python example, training, benchmark, results and open problems), §0, §4, §12 results folders, §13 results, §14 structure, citation table.
- ✅ Server (2026-10-12): `server_optimizers/naf_server.py` (`NAFServerRuntimeConfig`, `NAFServerSession`, `serve_naf_websocket`, port 8771), `configs/server_optimizers/naf_server.yaml` (the archived Branin AF), `server_scripts/run_naf_server.py`, `run_naf_fake_client.py` (a Branin-family task, `--task-seed`) and `tests/naf_server_test.py` (6 tests, including a session that matches `run_naf` exactly and a real socket). One session is one task; `start` may set `n_iter` up to the AF's budget `T`; `observe.y` is a scalar checked against `y_range`. A 30-step session takes about a second.

## Cost estimate (CPU only, to measure in Phase 3)
- **Measured (2026-10-11/12, 8 cores):** 52–60 s per PPO iteration at first (about 15 s rollouts on 7 workers, 42 s update), growing to about 140 s by iteration 230 (see question 10). After the denormal fix (2026-10-12): about 63 s per iteration at iteration 236, so 2000 iterations take about 1.5 days, not hours. The estimates below were too optimistic.
- One training step: GP posterior and MLP forward on about 6000 points (2000 for the global grid if the local grids are smaller), with T ≤ 30 data points. Expected a few ms after Phase 0.
- One PPO iteration: 1200 steps of rollouts plus 4 × 20 minibatch updates. Roughly 10–20 s on one core, so 2–4 s with 7 workers.
- Iterations to converge are not stated in the main text; the paper's figure files mention about 1,900 iterations for Branin. That is about 1–2 hours of training per family on 7 workers. The 1D toy test in Phase 3 should take under a minute.

## Out of scope (first pass)
- The simulation-to-real Furuta pendulum and the HPO tasks (they need external data).
- The GP-prior families and the dimension-agnostic variant (`use_x=False`) beyond a unit test. These are cheap to add later.
- Online GP hyperparameter fitting, batch suggestions, GPUs.

## Open questions
1. ✅ *Settled 2026-10-10: read the reference code, copy nothing (AGPL-3.0).* See "Reference code findings".
2. ✅ *Decided 2026-10-11: follow the code.* Raw `μ, σ, t, T`, with Branin normalized as in the reference (question 9). Standardized features would be a deviation.
3. ✅ *Settled by the code:* translations in normalized coordinates, scaling on the output, `[−0.1, 0.1]` and `[0.9, 1.1]`. Matches `TaskVariantSpec` (input shift plus `output_scale`).
4. ✅ *Settled by the code:* local maxima come from local Sobol grids on the policy's own logits; no gradients. Note the global grid is regular, not Sobol.
5. ✅ *Settled 2026-10-11 by TAF+:* `taf_plus_acquisition` with one objective and no decay is the reference's TAF-ME and TAF-R. The old note follows. **TAF fidelity.** OpenBO's TAF differs from the reference in three ways: its source reference is a fixed quantile of each source's own data (the reference uses the max of the source mean over the target's iterates), it uses softplus instead of ReLU, and its `taf_m` weights come from meta-feature distances, whereas the reference's TAF-ME uses per-point inverse variances. Proposed: add the reference forms as options (`source_reference_mode="target_iterates"`, `relu`, `taf_weight_mode="taf_me"`) before using TAF as a NAF baseline, and leave the defaults alone.
6. ✅ *Follows the code-wins rule:* near-zero noise. **Noise.** The reference uses effectively noise-free GPs (9e-16). Proposed: the same, with OpenBO's jitter.
7. ✅ *Follows the code-wins rule:* the best-training-reward checkpoint by default. **Stopping and checkpoints.** The reference picks the checkpoint with the best training reward. Proposed: the same by default (fair reproduction), plus an optional held-out validation set.
8. ✅ *Decided 2026-10-11: MetaBO-50 first (the code).* **Which MetaBO to reproduce.** The released script trains on the 50 fixed tasks (MetaBO-50); "full MetaBO" (a fresh task per episode) is a config change. Proposed: MetaBO-50 first, since TAF-50 uses the same tasks.
9. ✅ *Decided 2026-10-11: follow the code.* **Output scale.** The reference normalizes Branin's output. Proposed: use the same normalization inside the NAF family (a wrapper, not a change to `synthetic.branin`), so the fixed GP hyperparameters and raw features transfer.
10. ✅ **Repeated points.** *Settled 2026-10-12.*
    - **The reference allows repeats.** Its environment appends whatever point the policy picks and refits GPy on it (duplicates included, with GPy's own jitter). There is no masking, no deduplication and no penalty: a repeat simply earns no improvement in the reward. So OpenBO keeps repeats too (code-wins rule). The reference's answer is long training (about 2000 iterations; question 12). Masking evaluated points at test time would be a deviation, not done.
    - **The slowdown was not the GP.** A 30-point Cholesky costs microseconds even with retries. Profiling the archived state: rollouts took 15 s at iteration 0 and at iteration 236; the PPO update grew from 9 s to 21–25 s per epoch. A trained policy puts logits far apart (spread up to 310 per step, median 79), so 4.8% of the softmax probabilities, and the gradients the softmax passes back into the network, are float32 denormals, which the CPU multiplies 3× slower or worse.
    - **Fix** (`_ppo_update`): `torch.set_flush_denormal(True)` for the update (restored afterwards), plus a hook that zeroes denormal entries of the logits' gradient, because the flag only covers the calling thread and the update runs on 8 threads. One update epoch on a saved iteration-236 rollout: 22.6 s → 10.3 s (8 threads); 81 s → 25 s (1 thread). Resuming the archived run: about 63 s per iteration (was 140 s; 96 s with the flag alone). Rollouts are untouched, so trained AFs, the benchmark and the server behave as before; training is still deterministic, but resumed runs differ slightly from before the fix (gradients below 1.2e-38 are now zero).
11. **Early-step weighting (open).** The MetaBO reward sums log regret over all steps, so the policy spends its first evaluation on a probe. In human-in-the-loop use the first designs a user sees matter more. Not a MetaBO change; a note for human-in-the-loop extensions.
12. **Longer training (answered once, 2026-10-15; the 2026-10-12 part is below the table).** The reference trains about 2000 iterations; we stopped at 236. A copy of the archived run, `meta-bo-training/naf/branin_metabo50_v1_long`, was resumed (cap first 2000, then 500, then 300 by the user's choice to save time) and stopped after iteration 312 (313 iterations; best return 97.7 at iteration 307, against 90.6 at 193). The training return was still rising slowly (20-iteration means 81.7 → 87.0 → 88.5 → about 90). Benchmark of its `best_af` on the same 100 tasks (`test_results/naf/trajectories/v2_long313_naf_branin.json`), median log10 regret:

    | AF | @1 | @3 | @5 | @10 | @15 | @20 | @30 | mean 1–10 | mean 1–30 | exact repeats 11–30 | within 1e-3 |
    |---|---|---|---|---|---|---|---|---|---|---|---|
    | v2, iteration 307 | −0.66 | −1.29 | −1.60 | **−3.82** | −3.88 | −3.88 | −3.88 | **−2.03** | **−2.95** | 26% | 47% |
    | v1, iteration 193 (archive) | −0.66 | −1.35 | −1.60 | −3.31 | −3.35 | −3.36 | −3.36 | −1.92 | −2.67 | 24% | 46% |
    | EI (archive) | −0.15 | −0.65 | −0.87 | −1.53 | −2.76 | −3.37 | −4.98 | −0.98 | −2.77 | 3% | 3% |

    Longer training raised the plateau (−3.88 vs −3.36 from evaluation 10 on) and the mean over 30 evaluations now beats EI and both TAFs, but the policy still plateaus after about 10 evaluations and repeats points as often as before. Whether 2000 iterations remove the plateau is still open; resume `branin_metabo50_v1_long` with `--resume --n-iterations N` to continue.

    **2026-10-15: trained to 2000 iterations.** The 2000-iteration copy is `meta-bo-training/naf/branin_metabo50_v2_2000` (log `branin_metabo50_v2_2000.log`). It was trained from a fresh `--run-id` in three segments with the README's own commands (300 iterations, `--resume` to 600, `--resume` to 2000), with unmodified code. About 8.5 s per iteration alone and 13–24 s while other jobs shared the CPU (about 5 hours in all). Best checkpoint: iteration 1585 (return 135.4; 96.2 for the same run at 600 iterations, and the last-50-iteration mean is about 111, so the curve is noisy). Benchmark of its `best_af` on the same 100 tasks (`test_results/naf/trajectories/v3_2000_naf_branin.json`; rerunning the archived model reproduces it exactly):

    | AF | @1 | @3 | @5 | @10 | @15 | @20 | @30 | mean 1–10 | mean 1–30 | exact repeats 11–30 | within 1e-3 |
    |---|---|---|---|---|---|---|---|---|---|---|---|
    | v3, iteration 1585 (2000 its) | −0.75 | −1.23 | −1.57 | **−4.97** | **−5.62** | **−5.62** | **−5.62** | **−2.59** | **−4.62** | 49% | 59% |
    | v2, iteration 307 (313 its) | −0.66 | −1.29 | −1.60 | −3.82 | −3.88 | −3.88 | −3.88 | −2.03 | −2.95 | 26% | 47% |
    | v1, iteration 193 (archive) | −0.66 | −1.35 | −1.60 | −3.31 | −3.35 | −3.36 | −3.36 | −1.92 | −2.67 | 24% | 46% |
    | EI (archive) | −0.15 | −0.65 | −0.87 | −1.53 | −2.76 | −3.37 | −4.98 | −0.98 | −2.77 | 3% | 3% |

    **Answer: the plateau is mostly a training-length effect.** At 2000 iterations the best checkpoint's median log10 regret is −5.62, close to the paper's Figure 2(a) (about −5) and below EI at every budget from evaluation 10 on (EI reaches −4.98 at 30); see the limits below for the last iteration's model. The shortfall against the paper that this plan recorded after 313 iterations (about two orders of magnitude) is gone. The policy repeats points *more* (49% exact repeats in evaluations 11–30, against 26%), but it now does so after it has found the optimum, so the repeats cost nothing; the number of distinct points per run is 19.6 of 30 (24.4 for v1). Its first point is still the probe corner.

    **Limits.** One training run (seed 0), one set of 100 test tasks, no confidence interval. The checkpoint was chosen by the best training return, and the training curve has an abrupt setback right after it (`test_results/naf/plots/v3_2000_training.png`): mean final log10 regret about −5.2 around iteration 1585, about −3.7 around 1620, about −4.4 at 2000, and the mean return falls from 127.6 (iterations 1560–1584) to 110.3 (1586–1610). The last iteration's model (`af.pt`), benchmarked on the same 100 tasks, has median log10 regret −0.66 / −1.63 / −4.49 / −4.88 / −4.88 / −4.88 at evaluations 1 / 5 / 10 / 15 / 20 / 30 and mean −3.51 over 1–30, against −5.62 and −4.62 for the best checkpoint; it is slightly behind EI (−4.98) at 30. So "below EI at every budget from evaluation 10 on" holds for the best checkpoint only. The cause of the setback is not known. Repeat with other seeds before treating −5.6 as a typical result. Two known differences from the reference were left as they are, and the shortfall closed anyway: the local clouds are clipped to the unit square where the reference truncates its cube, and the EI baseline starts from a random Sobol point where the reference starts from the midpoint [0.5, 0.5] (so the README's earlier wording that EI follows the reference was wrong; the benchmark text now says so).
