# Multi-objective BO

[← Guide index](README.md) · [README](../../README.md)

Multi-objective BO with correlated objectives, following Shah & Ghahramani (ICML 2016); the website's [Methods](https://www.csie.ntu.edu.tw/~yichi/openbo-website/methods.html) page has the overview. All objectives are **maximized**, inputs live in `[0, 1]^d`, and observations have shape `(n, L)`. The design decisions and the paper reproduction notes are in [`docs/plans/mobo.md`](../plans/mobo.md).

## The pieces

- **Model** (`models/multi_output_gp.py`): `MultiOutputGP(mode=...)` models the
  objectives jointly and returns, at each input, a mean vector and an `L × L`
  covariance across objectives.
  - `independent`: one GP per objective.
  - `mt`: a multi-task GP with one kernel shared by all objectives.
  - `slf` (default): a semiparametric latent factor model with one kernel per
    latent function. It was the best model in the paper.
- **Hyperparameters**: MAP by default. `hyperparameters="slice"` follows the
  paper: it draws 10 hyperparameter samples per iteration by slice sampling
  and averages the acquisition over them.
- **Acquisition** (`acquisition/mobo.py`): the expected increase in Pareto volume.
  - **CEIPV** uses the predicted correlations between objectives.
  - **IEIPV** treats the objectives as independent and is exact in that case.
  - `eipv_mc` is a Monte Carlo ground truth for tests.
- **Baselines**: BoTorch **qLogNEHVI** with one independent GP per objective
  (`optimizers/mobo_botorch.py`), and random search.
- **Problems** (`test_functions/multi_objective.py`):

  | Problem | d | Objectives | Notes |
  | --- | --- | --- | --- |
  | `oka2` | 3 | 2 | analytic true hypervolume |
  | `vlmop3` | 2 | 3 | no closed-form front |
  | `dtlz1a` | 6 | 2 | analytic true hypervolume |
  | `slf_gp` | 3 | 2 | correlated GP draws with correlation `rho` |

  Each problem has a loose `v_ref` (the worst value over the domain) and a
  finite upper corner `w_ref`.

Three implementation notes, each verified by tests (see the plan for details):

- **The paper's printed λ is a typo.** The moment-matched mean of the CEIPV
  ramp is `(2w + v)/3`, not the printed `½(2w + v)`.
- **The paper's EIPV formula undercounts.** Read literally, eq. 3 credits
  only part of the hypervolume gain. CEIPV here uses an exact cell-local
  decomposition of the gain. The literal form remains available as
  `eipv_form="paper"`.
- **Predictions above `w_ref`.** Thin overflow layers above `w_ref` count
  predictive mass there, with the gain capped at `w_ref`.

## Run in Python

The website's Methods page has a `run_mobo` example on OKA2.

`MOBOSequentialOptimizer` exposes the same `bootstrap()` / `suggest()` /
`observe(x, y)` / `result()` ask/tell interface as the single-objective
optimizers.

## Benchmark and plot

```bash
# One method, one problem, several seeds (paper protocol: 5 initial points + 100 iterations)
uv run python scripts/run_mobo_benchmark.py --method mobo_ceipv_slf --problem oka2 \
  --n-iter 100 --seeds 0 1 2 3 4 --test-id oka2_demo

# Correlated synthetic problem: objectives drawn from an SLF GP with correlation rho
uv run python scripts/run_mobo_benchmark.py --method mobo_botorch --problem slf_gp --rho -0.5 \
  --n-iter 100 --seeds 0 1 2 --test-id slf_demo

# Hypervolume and log-hypervolume-regret curves, plus final Pareto fronts
uv run python scripts/plot_mobo_results.py --problem oka2 --test-id oka2_demo
```

Methods:

| Method | What it runs |
| --- | --- |
| `mobo_ceipv_slf` | CEIPV with the SLF model (MAP) |
| `mobo_ceipv_slf_slice` | CEIPV with the SLF model, slice-sampled hyperparameters (the paper's setup) |
| `mobo_ceipv_mt` | CEIPV with the multi-task model |
| `mobo_ieipv` | IEIPV with independent GPs (correlation ablation) |
| `mobo_botorch` | BoTorch qLogNEHVI (baseline) |
| `random` | uniform random search |

Results go to `test_results/mobo/trajectories/<test_id>_<method>_<problem>.json`,
where SLF-GP problems are labelled like `slf_gp_rho-0.50_p0`. Plots go to
`test_results/mobo/plots/`. `plot_mobo_results.py --test-id a b c --output-id all`
merges seeds that were saved under several test ids.

Cost: with three objectives (`vlmop3`), the 3-objective SLF GP fit dominates the run
time. `--n-rescore N` scores
the candidate pool with IEIPV and rescores only the top `N` with CEIPV, which
helps when the Pareto front is large.

## Run through the server

```bash
# Terminal 1 (port 8768; the default config is set up for OKA2)
uv run python server_scripts/run_mobo_server.py

# Terminal 2
uv run python server_scripts/run_mobo_fake_client.py --function oka2 --n-init 5 --n-iter 10
```

The protocol is described in [MOBO server protocol](servers.md#mobo-server-protocol).
