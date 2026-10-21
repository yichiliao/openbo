# TAF-EHVI: transfer for multi-objective BO

[← Guide index](README.md) · [README](../../README.md)

TAF-EHVI is one of OpenBO's two implementations of **TAF-MOBO**, the transfer
acquisition function for multi-objective BO (the other, recommended, is
[TAF-EIPV](taf-eipv.md)). It carries TAF's idea, reusing models of earlier
tasks or users, over to multi-objective BO. It was contributed by Mark Colley (PR #1, 2026-10-21) and is
OpenBO's own method: there is no paper for it yet. It builds on BoTorch's
qLogNEHVI ([Multi-objective BO](mobo.md)), not on OpenBO's NumPy CEIPV/IEIPV.

Like every OpenBO objective, the objectives are **maximized** and inputs live in
`[0, 1]^d`; an objective returns `(n, M)` values for `(n, d)` designs.

## How it works

The single-objective TAF blends the target's EI with each source's predicted
improvement. TAF-EHVI does the same with hypervolume:

| | TAF (`acquisition/taf.py`) | TAF-EHVI (`acquisition/taf_mo_ehvi.py`) |
| --- | --- | --- |
| Target term | EI of the target GP | qLogNEHVI of the target's GPs (one per objective) |
| Source term | improvement of the source's mean | log hypervolume improvement of the source's mean `μᵢ(x)` over the source's own Pareto front |
| Weights | TAF-M (meta-features) or TAF-R (ranking) | TAF-M, TAF-R per objective (default), or TAF-R on Pareto relations |
| Blend | `(w_t·T + Σ wᵢ·Sᵢ) / (w_t + Σ wᵢ)` | the same weighted average, computed in log space |

- A source contributes through its posterior **mean** only, as in TAF.
- The blend is `logsumexp(log w + log term) − log Σw`, exactly the log of the
  weighted average. Log space keeps a useful gradient for candidates that the
  sources' fronts already dominate.
- **With no sources, or all source weights at 0, the acquisition is exactly
  plain qLogNEHVI** (tested), so any difference from qLogNEHVI comes from
  transfer.
- **TAF-R weights** (`taf_weight_mode="taf_r"`, the default) score each source
  by how often its means order pairs of the target's observations the same way
  as the observations, objective by objective. A source that contradicts the
  observations drops to 0. Unlike OpenBO's single-objective TAF-R and the
  original TAF paper, the source weights are **normalized to sum to 1**. Below
  two observations there is no ranking evidence, so every source gets the same
  weight. `"taf_r_pareto"` scores pairs by their Pareto relation instead (kept
  for comparison); `"taf_m"` uses meta-feature distances.
- Optional: `source_only_warmup_iters` (first suggestions from the sources
  only), and a TAF+-style decay of the source weights (`decay_start_iter`,
  `decay_rate`). Both are off by default.

## The pieces

- `optimizers/mobo_taf.py`:
  - `MOTAFConfig`: `MOBoTorchConfig` (bounds, `v_ref`, `n_init`, `num_restarts`,
    `raw_samples`, `mc_samples`, `seed`) plus `taf_run_dir`, `n_iter`,
    `taf_weight_mode`, `rho`, `target_weight`, `source_reference_mode`
    (`"front"` or `"quantile"`), the warmup and the decay.
  - `MOTAFSequentialOptimizer`: ask/tell, `bootstrap()` / `suggest()` /
    `observe(x, y)` with `y` of shape `(n, M)` / `result()`. The result has
    `x_obs`, `y_obs`, `hypervolume_history` (one entry per evaluation),
    `pareto_front`, and `final_state` (number and names of the sources, the
    last source weights, the decay factor).
  - `run_mobo_taf(objective, bounds, ref_point, taf_run_dir, n_init, n_iter, ...)`.
  - `write_population_sources(population, directory)`: turns a TAF+ population
    (`TAFPlusPopulation`) into TAF-EHVI sources.
- `acquisition/taf_mo_ehvi.py`: the source term, the weights and the blend.

**Sources** use the TAF run layout: `<taf_run_dir>/gp_states/<task>.json` with
`{"gp_state": {"objectives": [one GP state per objective]}}` (or one shared
state), and `<taf_run_dir>/trajectories/<task>.json` with `x_values` `(n, d)` and
`y_values` `(n, M)`. Each source's GPs are rebuilt in BoTorch with the saved
hyperparameters, never refitted. A malformed or unpaired source, or one whose
front does not dominate `v_ref`, is skipped with a warning.

## Run in Python

Earlier users as sources (here the archived TAF+ population), then one new user,
ask/tell:

```python
import numpy as np

from openbo.optimizers.bo_taf_plus import TAFPlusPopulation
from openbo.optimizers.mobo_taf import MOTAFConfig, MOTAFSequentialOptimizer, write_population_sources
from openbo.test_functions.taf_plus_users import make_sphere_users

# Sources: 10 earlier users (the archived TAF+ population, range 0.1), one GP per objective each
population = TAFPlusPopulation.load("benchmark_results/taf_plus_sims/populations/v1_sim2_r01_s0")
source_dir = write_population_sources(population, "test_results/taf_ehvi/sources_demo")

# A new user: (n, 4) designs -> (n, 3) objective values, all maximized
user = make_sphere_users(1, shift_range=0.1, scale_range=0.1, seed=7)[0]
rng = np.random.default_rng(0)

config = MOTAFConfig(
    bounds=[(0.0, 1.0)] * 4,
    v_ref=[-1.0, -1.0, -1.0],  # hypervolume reference point, below every objective's worst
    taf_run_dir=source_dir,
    n_init=1,                  # TAF-EHVI needs at least one observation
    n_iter=9,
    num_restarts=4,
    raw_samples=256,
    seed=0,
)
opt = MOTAFSequentialOptimizer(config)
opt.bootstrap(lambda x: user.observe(x, rng))  # the random start
for _ in range(config.n_iter):
    x = opt.suggest()                     # (1, 4)
    opt.observe(x, user.observe(x, rng))  # y: (1, 3)

result = opt.result()  # x_obs, y_obs, hypervolume_history, pareto_front, final_state
print(result.pareto_front)
print(result.final_state["last_source_weights"])
```

This takes about two minutes on one thread (each step fits the target GPs and
scores 10 sources). For your own sources, save each earlier task's data and
per-objective GP states in the layout above; a TAF+ population is the easiest
way to get them.

## Benchmark and plot

TAF-EHVI is benchmarked through the TAF+ simulations, on the same users and
populations as TAF+ ([TAF+ benchmark](taf-plus.md#benchmark-and-plot)):

```bash
# TAF-EHVI with the population as sources, and plain qLogNEHVI (the same without sources)
OMP_NUM_THREADS=1 uv run python scripts/run_taf_plus_benchmark.py --method taf_ehvi --sim sim2_r01 --test-id demo
OMP_NUM_THREADS=1 uv run python scripts/run_taf_plus_benchmark.py --method mobo --sim sim2_r01 --test-id demo

# Weighted-sum regret (--metric best or instant) and hypervolume regret (--metric hv)
uv run python scripts/plot_taf_plus_results.py --test-id demo --metric hv
```

Both start from one random point, then run 19 steps with 256 raw samples and 4
restarts; a setting takes about 100 (TAF-EHVI) and 75 (qLogNEHVI) minutes on one
thread.

## Results

On the TAF+ Simulation 2 users (4 similarity ranges, 10 new users each, 20
evaluations, one seed), with full tables in
[`taf_plus_sims`](../../benchmark_results/taf_plus_sims/README.md#taf-ehvi-and-plain-qlognehvi-added-2026-10-21):

- **Transfer helps early.** TAF-EHVI is ahead of plain qLogNEHVI from its second
  evaluation in every range, and to about evaluation 10 for ranges up to 0.2.
- **By evaluation 20 the two are level** (within the 95% bands).
- **Against TAF+:** TAF+'s first design (the population's best guess) is
  unmatched early, and TAF-EHVI needs about 10 evaluations to catch up; by
  evaluation 20 TAF-EHVI and qLogNEHVI have 2.5 to 3.6 times less hypervolume
  regret than TAF+. TAF+ optimizes the weighted sum, not hypervolume, so this
  measures how much of the front its designs happen to cover.
- **Against TAF-EIPV** ([TAF-EIPV](taf-eipv.md)), OpenBO's other multi-objective
  TAF, whose source pull fades as the user explores: TAF-EIPV is clearly better on
  hypervolume for ranges up to 0.2, and about 25 times faster per step.

## Limitations and open work

- It needs at least one observation before its first suggestion (`n_init >= 1`);
  it cannot start from the sources alone, as TAF and TAF+ do. That start might
  close the early gap to TAF+.
- No WebSocket server yet, and not registered in the MOBO benchmark runner
  (`benchmarks/mobo_runner.py`); it is benchmarked only through the TAF+ runner.
- No design plan in `docs/plans/` and no paper in `relevant_papers/references.bib`.
- Only its defaults were benchmarked (TAF-R weights, no warmup, no decay), with
  one seed.
