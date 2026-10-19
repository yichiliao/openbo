# Continual BO (ConBO)

[← Guide index](README.md) · [README](../../README.md)

Continual BO across a sequence of users, following Liao et al. (CHI 2025); the website's [Methods](https://www.csie.ntu.edu.tw/~yichi/openbo-website/methods.html) page has the overview. The design decisions, deviations from the paper and measured costs are in [`docs/plans/conbo.md`](../plans/conbo.md).

## The pieces

- **Population model** (`models/bnn.py`): `PopulationBNN`, an MC-dropout
  network (3 × 100 ReLU) with mean and log-variance heads. Its predictive
  variance combines the predicted noise and the spread over dropout samples.
- **Generative replay** (`models/conbo_replay.py`): finished users are stored
  as GPs, not as a growing dataset. After each user, the BNN is retrained on
  the stored GPs' predictions over a fixed input set. Predictions whose
  variance is above a threshold are dropped, so uncertain users don't teach
  the population. By default the threshold is scale-free
  (`replay_variance_fraction=0.05`); set `replay_max_variance` (`λ`, in raw
  output units) for the paper's fixed threshold.
- **Mixed acquisition** (`acquisition/conbo.py`):
  `a(x) = w·EI_pop(x) + (1 − w)·EI_user(x)`, with `w` held at 1 for the first
  `α₁` evaluations of a user and then falling by `α₂` per evaluation. The
  defaults (`α₁ = 5`, `α₂ = 0.2`) hand over fully by the 10th evaluation.
- **Random starts**: user `u` gets `r_u = max(0, r₀ − (u − 1)·d_r)` random
  points first. Early users explore while the population model is still weak.
- **Optimizers** (`optimizers/conbo.py`):
  - `ConBOPopulation` holds the BNN and the memory of finished users, and can
    be saved to and loaded from a directory.
  - `ConBOSequentialOptimizer` runs one user, ask/tell.
  - `run_conbo_sequence` loops over users.

## Run in Python

The website's Methods page has a `run_conbo_sequence` example with the paper's Test 1 settings (15 simulated users).

One user at a time, for example inside a study session. The population
persists between sessions:

```python
import numpy as np

from openbo.optimizers.conbo import ConBOPopulation

def run_trial(x):
    """Stand-in for your experiment: maximized, shape (1,). Here a Branin-like peak."""
    return np.array([-((x[0, 0] - 0.25) ** 2 + (x[0, 1] - 0.75) ** 2)])

population = ConBOPopulation.load("test_results/conbo/memory_demo")
opt = population.new_user("participant_16")
for _ in range(30):
    x = opt.suggest()          # shape (1, d)
    y = run_trial(x)           # your experiment; returns shape (1,)
    opt.observe(x, y)
report = population.finish_user(opt)   # store the user and retrain by replay
population.save()
print(report.n_inputs_kept, report.kept_fraction)
```

Notes:
- Each user adapts a **private copy** of the population model. If a session is
  abandoned (never passed to `finish_user`), the population is unchanged.
- Users must be finished **in the order they were started**.
- `finish_user` takes about 10 s on a CPU (1,200 replay epochs), between
  users rather than during a session.
- The memory directory uses the TAF layout (`gp_states/`, `trajectories/`),
  so it can also be passed to TAF as `taf_run_dir`.
- Switches for the paper's ablations and open questions:
  - `use_user_gp=False` (ConBO without the user GP)
  - `replay_filter=False` (no variance filter)
  - `retrain="continue"` (keep training the BNN instead of resetting it)
  - `weight_clock="bo_steps"` (start the `α₁/α₂` clock after the random starts)
  - `refine_suggestion=False` (pure grid search, as in the paper)
- `λ` depends on the problem's output scale; the paper tuned it per problem
  (Branin 50, McCormick 10). For a new problem, set `replay_variance_fraction`
  instead and check `report.kept_fraction`.

## Benchmark and plot

```bash
# One method over a sequence of 15 simulated users, several seeds
uv run python scripts/run_conbo_benchmark.py --method conbo --seeds 0 1 2 --test-id demo
uv run python scripts/run_conbo_benchmark.py --method bo --seeds 0 1 2 --test-id demo

# Regret curves over users 4-15, per-user panels and a summary table
uv run python scripts/plot_conbo_results.py --test-id demo --from-user 4
```

Methods (`benchmarks/conbo_runner.py`); all see the same users per seed:

| Method | What it runs |
| --- | --- |
| `conbo` | ConBO |
| `bo` | standard BO, each user from scratch with `r₀` random starts |
| `taf_m` / `taf_r` | OpenBO's TAF with all previous users as sources |
| `single_gp` | one GP on all previous users' data plus the current user's |

Only `--test branin` (the paper's Test 1) is defined so far. Results go to
`test_results/conbo/trajectories/<test_id>_<method>_<test>_s<seed>.json`. The
archived 5-seed comparison is described in [Archived results](results.md#archived-benchmark-results).

## Run through the server

The ConBO server runs one participant per WebSocket session and keeps the
population on disk between sessions:

```bash
# Terminal 1 (port 8769)
uv run python server_scripts/run_conbo_server.py

# Terminal 2: four simulated users in a row
uv run python server_scripts/run_conbo_fake_client.py --n-users 4 --n-evaluations 20 --prefix sim
```

The protocol is described in [ConBO server protocol](servers.md#conbo-server-protocol).
