# NAF (MetaBO): a learned acquisition function

[← Guide index](README.md) · [README](../../README.md)

MetaBO from Volpp et al. (ICLR 2020): a neural acquisition function (NAF) trained with PPO on a family of related tasks; the website's [Methods](https://www.csie.ntu.edu.tw/~yichi/openbo-website/methods.html) page has the overview.

**Where the paper and the authors' reference code
([boschresearch/MetaBO](https://github.com/boschresearch/MetaBO)) disagree,
OpenBO follows the code.** The code is AGPL-3.0, so OpenBO reimplements its
behaviour and copies none of it. The design record, with the paper and code
notes, decisions, measurements and open questions, is
[`docs/plans/naf.md`](../plans/naf.md). Read it before changing NAF code.

## The pieces

- **Task families** (`test_functions/metabo_families.py`): MetaBO's three
  global benchmarks. `normalized_branin` is `-Branin` rescaled as in the
  reference code (`(y + 54.44) / 51.44`, maximum about 1.05);
  `normalized_goldstein_price` (2D, `-(log f − 8.693) / 2.427`) and
  `normalized_hartmann3` (3D, `-(f + 0.93) / 0.95`) use the reference's
  normalizations too, so all three have mean about 0 and standard deviation
  about 1. A task (`BraninTask`, `GoldsteinPriceTask`, `Hartmann3Task`) is
  `s · g(x − t)`, with translation `t` uniform in `[−0.1, 0.1]^D` and scaling
  `s` uniform in `[0.9, 1.1]`; `task.optimum` is tracked exactly.
  `sample_branin_tasks(n, seed)` (and `sample_goldstein_price_tasks`,
  `sample_hartmann3_tasks`) draw random tasks, and `branin_grid_tasks(50)` (and
  `goldstein_price_grid_tasks`, `hartmann3_grid_tasks`) give the 50 fixed
  training tasks of "MetaBO-50" (a Sobol grid over `(t, s)`).
  `NAF_FAMILY_SPECS` in `optimizers/naf.py` ties each family to its dimension,
  candidate set, GP hyperparameters and tasks, and `get_naf_family(name).config()`
  builds a matching `NAFConfig`.
- **GP** (`optimizers/naf.py`): every method uses a zero-mean RBF GP with
  **fixed** hyperparameters per family and almost no noise, as in the
  reference. For Branin these are the reference's values,
  `BRANIN_GP_HYPERPARAMETERS` (lengthscales 0.235 and 0.578, variance 2.0).
  `GOLDSTEIN_PRICE_GP_HYPERPARAMETERS` (lengthscales 0.109 and 0.087, variance
  0.652) and `HARTMANN3_GP_HYPERPARAMETERS` (lengthscales 0.709, 0.305 and 0.173,
  variance 0.694) are OpenBO's own type-2 ML fits, the median over 10 tasks. The
  reference's training scripts give the authors' values for these too
  (Goldstein-Price 0.130 and 0.07, variance 0.616, noise 1e-6; Hartmann-3 0.716,
  0.298 and 0.186, variance 0.83, noise 1.688e-11); they were noticed after the
  runs in [Archived results](results.md#archived-benchmark-results) were made, and the fits are close to them.
  `FamilyGPHyperparameters.new_gp()` builds one; `fit_family_hyperparameters`
  fits values for a new family.
- **Neural AF** (`acquisition/naf.py`): `NeuralAF` is an MLP (4 hidden layers
  of 200 ReLU units) that gives one score (logit) per candidate point from
  that point's features `[μ(x), σ(x), x, t, T]` (`naf_features`): the GP's
  posterior mean and standard deviation, the location, the step `t` and the
  budget `T`. The features are raw, not standardized, as in the code.
- **Candidates** (`AdaptiveSet`): each step the AF scores a 31 × 31 grid,
  then a cloud of 1000 local points around each of its 5 best grid points,
  and chooses among the grid plus the 5 best local points (966 candidates).
  During training it samples from the softmax of the logits; at test time it
  takes the argmax.
- **Training** (`train_naf`, `NAFConfig`): one BO run on a task is one RL
  episode (`BOEpisode`) of `T = 30` steps. The reward at step `t` is
  `−log10(simple regret)`, so the policy is trained to minimize the summed
  log regret over the whole run. PPO with GAE, settings from the paper's
  Table 4: 1200 steps (40 episodes) per iteration, 4 epochs, minibatches of
  60, Adam with learning rate 1e-4, clip 0.15, γ = λ = 0.98, entropy
  coefficient 0.01, and a separate value network on `[t, T]`. Rollouts run in
  worker processes. The checkpoint kept for testing (`best_af.pt`) is the
  iteration with the best mean training return, as in the reference.
- **Optimizer**: `NAFSequentialOptimizer` (ask/tell; `from_directory(dir)`
  loads `best_af.pt` with the GP hyperparameters and budget it was trained
  with) and `run_naf(objective, optimizer)`. The budget `T` is one of the
  AF's inputs, so the optimizer refuses to run past the training budget.

## Run a trained AF in Python

```python
import numpy as np

from openbo.optimizers.naf import NAFSequentialOptimizer, run_naf
from openbo.test_functions.metabo_families import sample_branin_tasks

# A new Branin-family task: translated by t in [-0.1, 0.1]^2, scaled by s in [0.9, 1.1]
task = sample_branin_tasks(1, seed=123)[0]

# Load the archived neural AF (best_af.pt) with its GP hyperparameters and budget (T = 30)
optimizer = NAFSequentialOptimizer.from_directory("benchmark_results/naf_branin/training/branin_metabo50_v1")

# Ask/tell by hand for the first evaluations ...
for _ in range(5):
    x = optimizer.suggest()  # shape (1, 2), in [0, 1]^2
    optimizer.observe(x, task(x))

# ... or let run_naf use the rest of the budget
x_obs, y_obs = run_naf(task, optimizer, n_iter=optimizer.budget - optimizer.t)
regret = task.optimum - np.maximum.accumulate(y_obs)
print(np.log10(np.maximum(regret[[0, 4, 9, 29]], 1e-20)))  # log10 regret after 1, 5, 10, 30 evaluations
```

This takes a few seconds. The objective takes `(n, 2)` points in `[0, 1]²`
and returns `(n,)` values to maximize, like every OpenBO objective. To use
your own objective, it has to belong to the family the AF was trained on.

## Train a neural AF

```bash
# MetaBO-50: train on the 50 fixed Branin tasks (7 rollout workers, 8 threads for the update)
uv run python scripts/train_naf.py --tasks metabo50 --n-iterations 300 --run-id my_run

# Continue a run to more iterations (only --n-iterations may change on resume)
uv run python scripts/train_naf.py --tasks metabo50 --n-iterations 600 --run-id my_run --resume

# Another family (the default run id is <family>_metabo50); Hartmann-3 is 3D and
# takes about twice as long per iteration as Branin
uv run python scripts/train_naf.py --family goldstein_price --n-iterations 2000
uv run python scripts/train_naf.py --family hartmann3 --n-iterations 2000
```

`--tasks full` draws a fresh random task for every episode ("full MetaBO")
instead of the 50 fixed ones. A run writes `meta-bo-training/naf/<run-id>/`
after every iteration: `af.pt` (latest), `best_af.pt` (best mean training
return), `naf_config.json`, `training_history.json` (per-iteration return,
final log regret, entropy, losses and time) and `training_state.pt`
(everything needed to resume). The log prints one line per iteration; mean
final log10 regret falling and entropy falling mean the policy is learning.
Stopping the process between iterations is safe, since the checkpoint is
written before each log line.

**Cost** (no GPU): the time per iteration is dominated by the PPO update. The archived run
slowed down as training went on, because a trained policy creates denormal floats in
the PPO update; the update now flushes them ([Open problems](#open-problems)).

## Benchmark and plot

The benchmark (`benchmarks/naf_runner.py`) follows the paper's Fig. 2: each
method runs on the same 100 unseen test tasks (`--test-seed 100`), 30
evaluations each. Every method uses the family's fixed-hyperparameter GP and
the same candidate set, so only the acquisition function differs.

```bash
AF=benchmark_results/naf_branin/training/branin_metabo50_v1   # or meta-bo-training/naf/<run-id>
for m in naf ei random taf_me_50 taf_r_50; do
  uv run python scripts/run_naf_benchmark.py --method $m --af-dir $AF --workers 8 --test-id demo
done
uv run python scripts/plot_naf_results.py --test-id demo --training-dir $AF
```

This block is for the Branin family, the default. **For Goldstein-Price or
Hartmann-3, pass `--family goldstein_price` or `--family hartmann3` to both
scripts**, with `--af-dir` pointing at a model of that family (for example
`benchmark_results/naf_goldstein_price/training/goldstein_price_metabo50_v1` or
`benchmark_results/naf_hartmann3/training/hartmann3_metabo50_v2_2000`); the
family sets the test tasks, the GP and the baselines' candidate set. If you
forget it, `run_naf_benchmark.py` checks the AF against the family: an AF of
another dimension (a Hartmann-3 AF with the default Branin) stops with an error that
says so, and an AF of the same dimension but another family (a Goldstein-Price AF
with Branin, both 2D) gets a warning that its GP hyperparameters belong to another
family, but still runs on the wrong tasks and saves `..._branin.json`, so heed the
warning. (Before this check existed, the Goldstein-Price AF ran silently on Branin
tasks and the Hartmann-3 one stopped with a shape error.)
```bash
uv run python scripts/run_naf_benchmark.py --method naf --family hartmann3 \
  --af-dir benchmark_results/naf_hartmann3/training/hartmann3_metabo50_v2_2000 --workers 8 --test-id demo
uv run python scripts/plot_naf_results.py --family hartmann3 --test-id demo \
  --training-dir benchmark_results/naf_hartmann3/training/hartmann3_metabo50_v2_2000
```
(The plot also needs the other methods' files for the same `--test-id` and
`--family`: run `ei`, `random`, `taf_me_50` and `taf_r_50` the same way.)

| Method | What it runs |
| --- | --- |
| `naf` | the trained AF (`best_af.pt` in `--af-dir`), greedy |
| `ei` | EI, starting from one random (scrambled Sobol) point. The paper (App. B.1) and the reference code start EI from the domain midpoint `[0.5, 0.5]`; OpenBO deviates. Starting from the midpoint changes EI's curve up to about evaluation 20 (median log10 regret at 15: −2.76 → −2.11) but not its value at 30 (−4.98 → −5.07) |
| `taf_me_50` | the original TAF (Wistuba et al.) with the 50 training tasks as sources (100 Sobol points each), product-of-experts (TAF-ME) weights |
| `taf_r_50` | the same with ranking-based TAF-R weights |
| `random` | uniform random points |

TAF is `taf_plus_acquisition` ([TAF+](taf-plus.md)) with one objective and no decay,
which is the reference's TAF-ME and TAF-R. One deviation: before any data,
the reference's TAF uses the task's own value at a corner as its incumbent,
which a real user could not provide; OpenBO's first TAF point maximizes the
sources' product-of-experts mean instead, as in TAF+.

Results go to `test_results/naf/trajectories/<test-id>_<method>_<family>.json`
(points, values and simple regret per step for every task) and
`test_results/naf/plots/` (median log10 regret with 30–70% bands, a table,
and with `--training-dir` the training curve). `naf`, `ei` and `random` take
under a minute for 100 tasks; `taf_me_50` and `taf_r_50` take about 40
minutes each on 8 workers, because every step scores 5000 local points
against 50 source GPs. Worker processes use one BLAS thread each (more
oversubscribes the CPU and makes TAF several times slower).

## Run through the server

The NAF server runs one task per WebSocket session, with a trained AF loaded
from `af_dir` (by default the archived Branin model):

```bash
# Terminal 1 (port 8771)
uv run python server_scripts/run_naf_server.py

# Terminal 2: one simulated Branin-family task, 30 evaluations (about a second)
uv run python server_scripts/run_naf_fake_client.py --task-seed 100
```

A session can stop early (`n_iter` in the `start` message) but never run
past the AF's training budget. The objective must come from the family the
AF was trained on, on the same value scale. The protocol and a client loop
are in [NAF server protocol](servers.md#naf-server-protocol).

The server does not depend on the family: it only checks `input_dim` and
`y_range` in its config. Two more configs are shipped, each for its archived AF:
`naf_server_goldstein_price.yaml` (2D, `y_range` [-3.5, 4.5], the 600-iteration
model) and `naf_server_hartmann3.yaml` (3D, `y_range` [-1.5, 4.5], the
2000-iteration model). Start one with `--config-path` and pass `--family
goldstein_price` or `--family hartmann3` to the fake client:

```bash
uv run python server_scripts/run_naf_server.py --config-path configs/server_optimizers/naf_server_hartmann3.yaml
uv run python server_scripts/run_naf_fake_client.py --family hartmann3
```

A full session with each shipped model ran through the server (one task, so only
a smoke test: log10 regret about -3.4 for Goldstein-Price and -3.1 for
Hartmann-3). A 2D client is refused by a 3D server with a clear message. The
`y_range` values cover the observed values of 3050 tasks of each family with a
margin. For an AF you train yourself, set `input_dim` and a `y_range` that
covers the family's values.

## Open problems

Results are in [Archived results](results.md#archived-benchmark-results). Open questions 10–12 in [`docs/plans/naf.md`](../plans/naf.md):

- **The first point is a probe.** It is the same corner on every task, and its
  value tells the policy how the task is shifted and scaled. MetaBO's reward
  sums the log regret over all `T` steps, so the policy gives up a good first
  guess for better later steps, and TAF starts better. For human-in-the-loop
  use, note that this reward weights every step equally.
- **Repeated points.** A trained AF often re-evaluates points it has already
  seen once it has converged. The reference code allows repeats (no masking),
  so OpenBO keeps them. The training slowdown they seemed to cause was
  denormal floats in the PPO update, now fixed (about 63 s per iteration
  instead of 140 s).
- **Training length.** Longer training lowered the Branin plateau a lot
  ([Archived results](results.md#archived-benchmark-results)) and fixed Hartmann-3's stall at 600 iterations, but
  Goldstein-Price did not improve beyond 600 iterations (it was continued to
  2000). Each long run is **one training seed**, picked by best training
  return; repeat with other seeds before relying on the numbers. The Branin and
  Hartmann-3 training curves are unstable late in training. Resume a run with
  `--resume --n-iterations N`.
