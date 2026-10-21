# TAF-EIPV: the original TAF on OpenBO's multi-objective BO

Status: **implemented and benchmarked** (2026-10-21): `acquisition/taf_eipv.py`,
`optimizers/mobo_taf_eipv.py`, `tests/taf_eipv_test.py`, the `taf_eipv` and `ieipv`
methods of `benchmarks/taf_plus_runner.py`, results in
`benchmark_results/taf_plus_sims/README.md`, user guide `docs/guide/taf-eipv.md`.
Served (with TAF-EHVI) by `server_optimizers/taf_mobo_server.py` (port 8772, 2026-10-21).

Result in one line: on hypervolume TAF-EIPV is the best method at almost every
evaluation on the TAF+ simulation users for ranges up to 0.2, starting level with
TAF+'s first design and then improving; open question 1 below is answered for
those users (it does not stall), though for the least similar users (range 0.3)
the methods without a fading pull finish slightly ahead.

## Goal

Transfer for multi-objective BO, built on OpenBO's own NumPy MOBO
(`optimizers/mobo.py`: a `MultiOutputGP` with IEIPV or CEIPV) instead of
BoTorch. It is the multi-objective counterpart of the original TAF of Wistuba,
Schilling and Schmidt-Thieme (Machine Learning, 2018), in hypervolume units.
TAF-EHVI (`optimizers/mobo_taf.py`, contributed in PR #1) is the BoTorch
sibling; the two differ in their reference front (below) and their weights.

Decided with the user on 2026-10-21: follow the original TAF's reference
(fading, not a fixed front), IEIPV by default with CEIPV optional, and the name
TAF-EIPV.

## The acquisition

All objectives are maximized; `v_ref` and `w_ref` bound the objective space as
in `MOBOConfig`.

```
a(x) = (w_t · EIPV_t(x) + Σ_i w_i · I_i(x)) / (w_t + Σ_i w_i)
```

- **Target term** `EIPV_t(x)`: OpenBO's IEIPV (default) or CEIPV of the new
  task's multi-output GP over its current Pareto front, exactly as
  `MOBOSequentialOptimizer` computes it.
- **Source term** `I_i(x)`: the hypervolume that source `i`'s posterior mean
  `μ_i(x)` adds to a reference front `F_i`, computed exactly with
  `benchmarks.hypervolume.nondominated_cells(F_i, v_ref, w_ref)` and
  `hypervolume_gain`. Both terms are volumes in the same objective space.
- **Reference front** `F_i` (original TAF, Wistuba et al. eq. 47, read as in
  `docs/plans/taf_plus.md` D2): the Pareto front of `{μ_i(x_j)}` over the
  designs `x_j` the target has already evaluated, that is, what source `i`
  predicts for the points already tried. As the target explores what a source
  promised, that source's term fades and the target's EIPV takes over. TAF-EHVI
  instead uses the source's own data front, which never changes.
- **No target data yet**: `F_i` is empty, so `I_i(x)` is the hypervolume of the
  single point `μ_i(x)` above `v_ref`, and the target term is absent. The first
  suggestion maximizes the sources' weighted predicted hypervolume: TAF-EIPV can
  start from the sources alone (`n_init = 0`), which TAF-EHVI cannot. This is
  the hypervolume analogue of TAF+'s population first point.
- **Log space**: the optimizer maximizes `log a(x)` (a `logsumexp` over the
  positive terms). From the first observation on, the target term is positive
  everywhere, and before it every source term is positive wherever `μ_i(x)`
  dominates `v_ref`, so no smoothing of the source term is needed.

## Weights

- **TAF-R** (default; did best in the original): source `i`'s distance is the
  fraction of ordered pairs of the target's observations that `μ_i` ranks
  differently from the observations, pooled over objectives (the mean of the
  per-objective distances of `acquisition/taf_plus.ranking_distance`), and its
  weight is the Epanechnikov kernel `γ(d / ρ)`, `γ(t) = ¾(1 − t²)` for
  `t ≤ 1`. Not normalized, as in the original. With fewer than two
  observations the distance is 0 and every source gets `¾`.
- **Target weight**: `γ(0) = ¾`, as in the original (kernel regression with the
  target as its own descriptor). OpenBO's single-objective TAF uses 1; TAF-EHVI
  normalizes the source weights to sum to 1.
- **Uniform** (option): every source `¾`, for ablations.
- **Decay** (option, off by default): the source weights multiplied by TAF+'s
  `δ(t)` (`decay_factor(t, d1, d2)`, CHI 2024 eq. 6), `t` the number of target
  observations plus one.

## Sources

`GPScratch` models, one per objective (fixed hyperparameters, refit on the
source's data), so a `TAFPlusPopulation` is used directly. A directory in the
TAF run layout also works: `gp_states/<task>.json` with
`{"gp_state": {"objectives": [one state per objective]}}` and
`trajectories/<task>.json` with `x_values (n, d)` and `y_values (n, M)`, which
is what `mobo_taf.write_population_sources` writes, so TAF-EIPV and TAF-EHVI
can share sources.

## Implementation

- `acquisition/taf_eipv.py`: `source_improvement(mean, reference_front, v_ref,
  w_ref)`, `taf_r_source_weights(y_target, source_means_at_target, rho)`,
  `log_taf_eipv(...)` (the blend in log space).
- `optimizers/mobo_taf_eipv.py`: `TAFEIPVConfig(MOBOConfig)` (adds the sources,
  `weight_mode`, `rho`, `target_weight`, the decay), `TAFEIPVSequentialOptimizer
  (MOBOSequentialOptimizer)` (overrides `_log_acquisition`, and `suggest` for
  the source-only first step), `run_mobo_taf_eipv`, and a source loader.
- Search: unchanged from `MOBOSequentialOptimizer` (Sobol pool, then L-BFGS-B
  with finite-difference gradients).

## Tests (`tests/taf_eipv_test.py`)

- With no sources (or all weights 0), the acquisition equals plain IEIPV/CEIPV
  and the run equals `MOBOSequentialOptimizer` step for step.
- `source_improvement` equals a brute-force `hypervolume(F ∪ {μ}) − hypervolume(F)`.
- It is 0 at a design the target has already evaluated (the fading reference).
- With no target data, the first suggestion maximizes the sources' weighted
  predicted hypervolume.
- TAF-R weights: a copy of the target scores `¾`, a reversed source 0, and the
  values match `taf_plus.ranking_distance`.
- Loading from a TAF+ population and from the TAF run layout gives the same
  sources.

## Benchmark

The TAF+ Simulation 2 users (`benchmarks/taf_plus_runner.py`), as for
TAF-EHVI: new methods `taf_eipv` (sources from the population, starting from
the sources alone) and `ieipv` (the same optimizer without sources), next to
TAF-EHVI, qLogNEHVI, TAF+, BO and random, on the weighted-sum and hypervolume
metrics. Archive the results in `benchmark_results/taf_plus_sims/`.

## Open questions

1. Whether the fading reference makes TAF-EIPV stall after the sources'
   promise is used up, as TAF+ does. Unlike TAF+, its target term is EIPV over
   the whole front, which kept qLogNEHVI improving late in the TAF-EHVI
   benchmark.
2. Whether TAF-PoE (variance-based) weights, TAF+'s default, should be an option.
3. Source term with uncertainty (each source's own IEIPV instead of the
   improvement of its mean), if the deterministic term proves too sharp.
