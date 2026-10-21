# TAF-EIPV: the original TAF for multi-objective BO

[← Guide index](README.md) · [README](../../README.md)

TAF-EIPV is one of OpenBO's two implementations of **TAF-MOBO**, the transfer
acquisition function for multi-objective BO (the other is [TAF-EHVI](taf-ehvi.md)). It carries the original TAF (Wistuba, Schilling and Schmidt-Thieme,
Machine Learning 2018) over to multi-objective BO, built on OpenBO's own NumPy
MOBO ([Multi-objective BO](mobo.md)). It reuses earlier tasks or users (the
*sources*) to find a new user's Pareto front faster, and it can start from the
sources alone, with no random designs. It is OpenBO's own method (2026-10-21);
the design record, with every decision and open question, is
[`docs/plans/taf_eipv.md`](../plans/taf_eipv.md).

Like every OpenBO objective, the objectives are **maximized** and inputs live in
`[0, 1]^d`; an objective returns `(n, M)` values for `(n, d)` designs.

## How it works

```
a(x) = (w_t · EIPV_t(x) + Σ_i w_i · I_i(x)) / (w_t + Σ_i w_i)
```

- **Target term** `EIPV_t`: the new user's expected increase in Pareto volume,
  IEIPV on independent GPs by default (CEIPV with `acquisition="ceipv"`), as in
  plain multi-objective BO.
- **Source term** `I_i`: the hypervolume that source `i`'s posterior mean
  `μ_i(x)` adds to a **reference front**: what source `i` predicts at the
  designs the new user has already tried. This is the original TAF's reference,
  and it fades: once the user has tried what a source promised, that source
  stops pulling, and the target's EIPV takes over. Both terms are volumes, so
  the blend is in one unit.
- **Before any data** the reference is empty, so the first suggestion maximizes
  the sources' weighted predicted hypervolume: the population's best guess at
  the whole front. Use `n_init=0`.
- **Weights**: TAF-R, as in the original: each source's weight is the
  Epanechnikov kernel of how often its means rank pairs of the user's
  observations differently from the observations, pooled over objectives.
  A source that ranks them all the wrong way gets 0. The weights are not
  normalized, and the target gets `3/4`. Options: `weight_mode="uniform"`, and
  TAF+'s decay of the source weights (`decay=True`, `d1`, `d2`).
- **With no sources** it is exactly OpenBO's plain multi-objective BO (tested).

TAF-EHVI ([TAF-EHVI](taf-ehvi.md)) is the BoTorch sibling: it measures each
source's improvement against the source's own data front, which never fades,
and needs one random start. On the benchmark below TAF-EIPV is clearly better
and about 25 times faster per step.

## The pieces

- `optimizers/mobo_taf_eipv.py`:
  - `TAFEIPVConfig`: `MOBOConfig` (bounds, `v_ref`, `w_ref`, `n_init`, the
    model and acquisition, the search settings, `seed`) plus `taf_run_dir`,
    `weight_mode`, `rho`, `target_weight` and the decay. Defaults:
    `model="independent"`, `acquisition="ieipv"`.
  - `TAFEIPVSequentialOptimizer(config, population=None)`: ask/tell, with
    `suggest()` returning `(1, d)` and `observe(x, y)` taking `y` of shape
    `(n, M)`. `result()` is a `MOBORunResult` (`x_obs`, `y_obs`, `hv_history`,
    `pareto_x`, `pareto_y`), with the sources and their last weights in
    `final_model_state`.
  - `run_mobo_taf_eipv(objective, config, n_iter, population=None)`.
  - `load_sources(dir)` and `sources_from_population(population)`.
- `acquisition/taf_eipv.py`: `reference_cells`, `source_improvement`,
  `taf_r_source_weights` and the blend `log_taf_eipv`.

**Sources** are a `TAFPlusPopulation` (pass it to the optimizer), or a
directory in the TAF run layout (`config.taf_run_dir`): `gp_states/<task>.json`
with one GP state per objective (as saved by `TAFPlusPopulation.save`, or by
`mobo_taf.write_population_sources`) and `trajectories/<task>.json` with
`x_values (n, d)` and `y_values (n, M)`. Each source's GPs are rebuilt with
their saved hyperparameters, never refitted.

## Run in Python

Earlier users as sources (here the archived TAF+ population), then one new
user, ask/tell, starting from the sources:

```python
import numpy as np

from openbo.optimizers.bo_taf_plus import TAFPlusPopulation
from openbo.optimizers.mobo_taf_eipv import TAFEIPVConfig, TAFEIPVSequentialOptimizer
from openbo.test_functions.taf_plus_users import make_sphere_users

# Sources: 10 earlier users (the archived TAF+ population, range 0.1), one GP per objective each
population = TAFPlusPopulation.load("benchmark_results/taf_plus_sims/populations/v1_sim2_r01_s0")

# A new user: (n, 4) designs -> (n, 3) objective values, all maximized
user = make_sphere_users(1, shift_range=0.1, scale_range=0.1, seed=7)[0]
rng = np.random.default_rng(0)

config = TAFEIPVConfig(
    bounds=[(0.0, 1.0)] * 4,
    v_ref=[-1.0, -1.0, -1.0],  # lower corner: below every objective's worst value
    w_ref=[1.2, 1.2, 1.2],     # upper corner: above every objective's best value
    n_init=0,                  # no random start: the first design comes from the sources
    seed=0,
)
opt = TAFEIPVSequentialOptimizer(config, population)
for _ in range(10):
    x = opt.suggest()                     # (1, 4)
    opt.observe(x, user.observe(x, rng))  # y: (1, 3)

result = opt.result()  # x_obs, y_obs, hv_history, pareto_x, pareto_y, final_model_state
print(result.pareto_y)
print(result.final_model_state["last_source_weights"])
```

This takes a few seconds. For your own study, model each earlier participant
with one GP per objective (`TAFPlusPopulation.add_user(name, x, y)` fits them),
save the population, and load it for each new participant.

## Benchmark and plot

TAF-EIPV is benchmarked on the TAF+ simulation users, next to TAF+, TAF-EHVI
and the methods without sources ([TAF+ benchmark](taf-plus.md#benchmark-and-plot)):

```bash
# TAF-EIPV with the population as sources, and plain IEIPV (the same without sources)
OMP_NUM_THREADS=1 uv run python scripts/run_taf_plus_benchmark.py --method taf_eipv --sim sim2_r01 --test-id demo
OMP_NUM_THREADS=1 uv run python scripts/run_taf_plus_benchmark.py --method ieipv --sim sim2_r01 --test-id demo

# Hypervolume regret (--metric hv), and weighted-sum regret (--metric best or instant)
uv run python scripts/plot_taf_plus_results.py --test-id demo --metric hv
```

A setting takes about 4.5 minutes for TAF-EIPV and 1.5 for IEIPV on one thread.

## Results

On the TAF+ Simulation 2 users (4 similarity ranges, 10 new users each, 20
evaluations, one seed); full tables in
[`taf_plus_sims`](../../benchmark_results/taf_plus_sims/README.md#taf-eipv-and-plain-ieipv-added-2026-10-21):

- **On hypervolume** (what it optimizes), TAF-EIPV is the best method at almost
  every evaluation for ranges 0.05 to 0.2. Its first design, from the sources
  alone, matches TAF+'s, and it keeps improving: at range 0.05 its log
  hypervolume regret ends at -0.95, against -0.62 for TAF-EHVI, -0.58 for
  qLogNEHVI and -0.41 for plain IEIPV. For the least similar users (range 0.3)
  it leads through about evaluation 13, and the other multi-objective methods
  finish slightly ahead.
- **On the weighted sum** (what TAF+ optimizes, with the weights TAF-EIPV never
  sees), TAF+ keeps its edge for similar users; for more varied users TAF-EIPV is
  ahead from evaluation 3.

## Limitations and open work

- No WebSocket server yet (planned: one server for TAF-EIPV and TAF-EHVI).
- Only its defaults were benchmarked (TAF-R weights, no decay, IEIPV), with one
  seed; CEIPV and the decay are untested in the benchmark.
- Not registered in the MOBO benchmark runner (`benchmarks/mobo_runner.py`).
