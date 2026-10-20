# TAF+: transfer across users with several objectives

[← Guide index](README.md) · [README](../../README.md)

TAF+ from Liao et al. (CHI 2024), built on the original TAF of Wistuba, Schilling and Schmidt-Thieme (Machine Learning, 2018); the website's [Methods](https://www.csie.ntu.edu.tw/~yichi/openbo-website/methods.html) page has the overview. **Wherever the CHI 2024 paper and the original TAF paper disagree, OpenBO follows the original.** In particular:

- the source term is the improvement of each earlier user's posterior mean
  over the best value it predicts among the new user's evaluated designs,
  clipped at 0;
- the model weights are the original's TAF-PoE (inverse predictive
  variance), which the CHI paper calls TAF-M;
- the default kernel is SE-ARD (RBF), with Matérn-5/2 as an option.

This is different from OpenBO's older TAF ([TAF workflow](taf.md#transfer-acquisition-function-taf-workflow)). The decisions,
deviations and open questions are in
[`docs/plans/taf_plus.md`](../plans/taf_plus.md).

## The pieces

- **Acquisition** (`acquisition/taf_plus.py`): each objective `l` gets the
  original TAF terms. These are combined across objectives with the weights
  `α` (CHI eq. 4), which also combine the model weights (eq. 5). The
  population weights are then multiplied by a decay `δ(t)` (eq. 6): `δ = 1`
  for the first `d1` evaluations, then it falls by `d2` per evaluation until
  it reaches 0. From then on TAF+ is plain EI on the new user's GPs.
- **Population** (`optimizers/bo_taf_plus.py`): `TAFPlusPopulation` fits
  one GP per objective for each earlier user (`add_user(name, x, y)`, with
  `y` of shape `(n, L)`). `save`/`load` use a directory with `gp_states/`,
  `trajectories/` and `taf_plus_population.json`. `subset(exclude=name)`
  supports leave-one-out runs.
- **Optimizer**: `TAFPlusSequentialOptimizer` (ask/tell, `observe(x, y)`
  with `y` of shape `(1, L)`) and `run_bo_taf_plus`.
  - The first design maximizes the population's product-of-experts mean
    (`first_point="population"`, the default), or is the domain centre
    (`first_point="centre"`, also the fallback for an empty population).
  - After that, each step takes the TAF+ argmax over a fresh Sobol pool
    (`n_candidates`, default 1024) and polishes it with L-BFGS-B.
- **Workflow** (`optimizers/taf_plus_workflow.py`), the CHI paper's steps
  2 and 3:
  - `optimize_objective_weights` is Algorithm 1. It picks `α` from each
    population user's ratings of their own Pareto-optimal designs.
  - `search_decay` is the leave-one-out grid search for `(d1, d2)` over the
    paper's grid (`d1` in 1–9, `d2` in 0.1–0.3). Each population user in turn
    plays the new user. Pass `n_workers` to use several processes.

`TAFPlusConfig` switches (defaults first):

- `decay=True`: `False` keeps the population weights at full strength.
  That is the paper's "TAF (multi-objective)" baseline, and with one
  objective it is the original TAF.
- `d1=2`, `d2=0.3`: the CHI study's decay.
- `weight_mode="poe"`: `"taf_r"` uses the original's ranking-based TAF-R
  weights, with bandwidth `rho`.
- `kernel_type="rbf"`: `"matern52"` is the CHI paper's kernel.
- `resample_candidates=True`, `refine_suggestion=True`: `False` gives one
  fixed pool and the paper's pure pool search.

## Run in Python

```python
import numpy as np

from openbo.optimizers.bo_taf_plus import TAFPlusConfig, TAFPlusPopulation, run_bo_taf_plus
from openbo.optimizers.taf_plus_workflow import search_decay
from openbo.test_functions.taf_plus_users import make_sphere_users

# 4 earlier users and 1 new one: 4D input, 3 maximized objectives
users = make_sphere_users(5, shift_range=0.1, scale_range=0.1, seed=0)
earlier, new_user = users[:4], users[4]
rng = np.random.default_rng(0)

# Step 1: one GP per objective for each earlier user, from their data
population = TAFPlusPopulation(input_dim=4, n_objectives=3)
for i, user in enumerate(earlier):
    x = rng.random((30, 4))
    population.add_user(f"user_{i:02d}", x, user.observe(x, rng))
population.save("test_results/taf_plus/demo_population")

# Step 3 (optional): leave-one-out search for the decay (d1, d2)
config = TAFPlusConfig(input_dim=4, objective_weights=(0.3, 0.5, 0.2), n_candidates=256)
decay = search_decay(population, config, d1_values=(2, 4), d2_values=(0.1, 0.3), n_iter=5)
d1, d2 = decay.best

# Step 4: adapt to the new user
config = TAFPlusConfig(input_dim=4, objective_weights=(0.3, 0.5, 0.2), d1=d1, d2=d2)
result = run_bo_taf_plus(lambda x: new_user.observe(x, rng), config, population, n_iter=10)
print(result.best_so_far)    # best weighted sum y @ alpha after each evaluation
print(result.decay_history)  # delta(t) per step
```

Most of the time goes to the decay search. For
real users, step 1 is usually a multi-objective BO session per person
([Multi-objective BO](mobo.md)). Step 2 (`optimize_objective_weights`) needs each person's
ratings of their Pareto-optimal designs. Without ratings, set `α` by hand,
as the paper's simulations do.

## Benchmark and plot

The runner (`benchmarks/taf_plus_runner.py`) reproduces the paper's
App. B simulations. The base function has three 2D spheres on a 4D
input, giving 3 objectives. Each simulated user shifts the inputs and scales
the outputs within a similarity range `r`. Each setting has 10 population
users, modelled with qLogNEHVI, and 10 new users. Settings (`--sim`):
`sim1` (`r = 0.01`, used with six `--alpha` vectors), and `sim2_r005`,
`sim2_r01`, `sim2_r02`, `sim2_r03` (`r` = 0.05–0.3, `α = (0.3, 0.5, 0.2)`).

```bash
# Population models (built on first use and cached), then the decay search
uv run python scripts/run_taf_plus_benchmark.py --method decay_search --sim sim2_r01 --workers 7 --test-id demo

# Each method on the new users (taf_plus reads the decay from the search above)
uv run python scripts/run_taf_plus_benchmark.py --method taf_plus --sim sim2_r01 --test-id demo
uv run python scripts/run_taf_plus_benchmark.py --method taf --sim sim2_r01 --test-id demo
uv run python scripts/run_taf_plus_benchmark.py --method bo --sim sim2_r01 --test-id demo
uv run python scripts/run_taf_plus_benchmark.py --method random --sim sim2_r01 --test-id demo

# Multi-objective methods on the same users: TAF-EHVI (the population as sources) and plain qLogNEHVI
OMP_NUM_THREADS=1 uv run python scripts/run_taf_plus_benchmark.py --method taf_ehvi --sim sim2_r01 --test-id demo
OMP_NUM_THREADS=1 uv run python scripts/run_taf_plus_benchmark.py --method mobo --sim sim2_r01 --test-id demo

# Regret curves (each design, --metric best, or --metric hv for hypervolume), decay heatmaps and a table
uv run python scripts/plot_taf_plus_results.py --test-id demo
```

| Method | What it runs |
| --- | --- |
| `taf_plus` | TAF+ with the searched decay (or `--d1/--d2`) |
| `taf` | the same without decay (the paper's multi-objective TAF) |
| `bo` | scratch EI on the noisy weighted sum, 5 random starts |
| `random` | uniform random designs |
| `taf_ehvi` | TAF-EHVI ([Multi-objective BO](mobo.md#taf-ehvi-and-the-botorch-compatibility-api)) with the population's users as sources; optimizes hypervolume and never sees `alpha`; 1 random start, then 19 steps |
| `mobo` | plain qLogNEHVI with the same start and settings (TAF-EHVI without sources) |

`--metric hv` scores every method on the noise-free hypervolume of the designs evaluated so far, against the user's true Pareto front (traced by maximizing 861 weighted sums; the first plot computes it, about 25 s per user, and caches it in `plots/pareto_hypervolume.json`). TAF-EHVI and qLogNEHVI take about 100 and 75 minutes per setting on one thread; run one process per setting with `OMP_NUM_THREADS=1`.

Building a population is the slowest step; it is cached on disk and reused. Results go to `test_results/taf_plus/` ([Results folders](results.md#results-folder-convention)). The archived
run and its findings are in [Archived results](results.md#archived-benchmark-results): transfer gives an excellent first
design, but TAF+ rarely improves after it. TAF-EHVI beats plain qLogNEHVI early (to about evaluation 10 for ranges up to 0.2), and both end with 2.5 to 3.6 times less hypervolume regret than TAF+ by evaluation 20.

## Run through the server

The TAF+ server runs one new user per WebSocket session, with a fixed
population loaded from `population_dir`:

```bash
# Terminal 1 (port 8770; the default config uses the archived range-0.1 population)
uv run python server_scripts/run_taf_plus_server.py

# Terminal 2: one simulated new user, 10 evaluations
uv run python server_scripts/run_taf_plus_fake_client.py --range 0.1 --user-seed 100
```

The protocol is described in [TAF+ server protocol](servers.md#taf-server-protocol).
