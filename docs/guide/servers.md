# WebSocket servers

[← Guide index](README.md) · [README](../../README.md)

Every optimizer family has a WebSocket server, for objectives that live outside this process (a study app, a simulator, another language). One WebSocket connection is one session. The website's [Get started](https://www.csie.ntu.edu.tw/~yichi/openbo-website/get-started.html#servers) page lists the servers with their default ports and configs, and has a minimal client. This page is the full reference.

## Step 1. Start the optimizer server

Use one of the backend-specific config files:
- `configs/server_optimizers/bo_server_botorch.yaml`
- `configs/server_optimizers/bo_server_scratch.yaml`
- `configs/server_optimizers/bo_taf_server.yaml`
- `configs/server_optimizers/mobo_server.yaml`
- `configs/server_optimizers/conbo_server.yaml`
- `configs/server_optimizers/taf_plus_server.yaml`
- `configs/server_optimizers/naf_server.yaml` (Branin; also `naf_server_goldstein_price.yaml` and `naf_server_hartmann3.yaml` for the other two families)
- `configs/server_optimizers/taf_mobo_server.yaml`

Without `--config-path`, `run_bo_server.py` uses
`configs/server_optimizers/bo_server_botorch.yaml`. Pass
`configs/server_optimizers/bo_server_scratch.yaml` for the scratch backend.
Every server rejects an `observe.y` that is NaN or infinite, and a missing or
invalid config file is reported to the client as an `error` message.

Start server with BoTorch backend:

```bash
uv run python server_scripts/run_bo_server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --config-path configs/server_optimizers/bo_server_botorch.yaml
```

Start server with scratch backend:

```bash
uv run python server_scripts/run_bo_server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --config-path configs/server_optimizers/bo_server_scratch.yaml
```

Start dedicated TAF server:

```bash
uv run python server_scripts/run_taf_server.py \
  --host 127.0.0.1 \
  --port 8766 \
  --config-path configs/server_optimizers/bo_taf_server.yaml
```

Start the dedicated MOBO server (the default config is set up for OKA2):

```bash
uv run python server_scripts/run_mobo_server.py \
  --host 127.0.0.1 \
  --port 8768 \
  --config-path configs/server_optimizers/mobo_server.yaml
```

Start the dedicated ConBO server. The population lives in the config's
`memory_dir` and survives restarts:

```bash
uv run python server_scripts/run_conbo_server.py \
  --host 127.0.0.1 \
  --port 8769 \
  --config-path configs/server_optimizers/conbo_server.yaml
```

Start the dedicated TAF+ server. The population is read from the config's
`population_dir` and is never changed by a session:

```bash
uv run python server_scripts/run_taf_plus_server.py \
  --host 127.0.0.1 \
  --port 8770 \
  --config-path configs/server_optimizers/taf_plus_server.yaml
```

Start the dedicated NAF server. The trained AF is read from the config's
`af_dir` and is never changed by a session:

```bash
uv run python server_scripts/run_naf_server.py \
  --host 127.0.0.1 \
  --port 8771 \
  --config-path configs/server_optimizers/naf_server.yaml
```

Start the TAF-MOBO server (transfer for multi-objective BO, port 8772). It serves
[TAF-EIPV](taf-eipv.md) (default) or [TAF-EHVI](taf-ehvi.md). The sources are
read from the config's `population_dir` or `taf_run_dir` once, when the server
starts, and are never changed by a session; restart the server after editing the
config:

```bash
uv run python server_scripts/run_taf_mobo_server.py \
  --host 127.0.0.1 \
  --port 8772 \
  --config-path configs/server_optimizers/taf_mobo_server.yaml
```

The server reads the config file at session start.
Example BoTorch config:

```yaml
optimizer: bo_botorch
input_dim: 2
y_range: [-350.0, 1.0]
n_init_default: 2
n_iter_default: 8
num_restarts_default: 5
raw_samples_default: 64
```

Example scratch config:

```yaml
optimizer: bo_scratch
input_dim: 2
y_range: [-350.0, 1.0]
n_init_default: 2
n_iter_default: 8
auto_save_scratch_artifacts: true
scratch_artifacts_dir: meta-bo-training/server-scratch-artifacts
```

For `bo_scratch`, search settings (`n_candidates`, `n_starts`,
`search_strategy`, `kernel_type`, `optimize_hyperparameters`) are read only
from the `start` message; the `*_default` keys in `bo_server_scratch.yaml` are
currently ignored by the generic server. The TAF server does honor its
`*_default` keys.

Example TAF config (the shipped `bo_taf_server.yaml`):

```yaml
input_dim: 2
y_range: [-350.0, 1.0]
taf_run_dir: meta-bo-training/taf-gps/branin_train_v1
n_init_default: 0
n_iter_default: 10
n_candidates_default: 512
n_starts_default: 8
search_strategy_default: multistart
kernel_type_default: matern52
optimize_hyperparameters_default: true
taf_weight_mode_default: taf_r  # or taf_m
taf_rho_default: 1.0
source_only_warmup_iters_default: 3
track_acquisition_default: true
```

Example MOBO config (`mobo_server.yaml`, OKA2). `y_range` holds one
`[min, max]` per objective, and the hypervolume reference box defaults to it:

```yaml
input_dim: 3
y_range: [[-3.2, 3.1416], [-5.1, 0.0]]
model: slf            # independent | mt | slf
acquisition: ceipv    # ceipv | ieipv
hyperparameters: map  # map | slice
n_init_default: 5
n_iter_default: 20
n_candidates_default: 1024
```

Example ConBO config (`conbo_server.yaml`). The `conbo:` settings are used only
when `memory_dir` holds no population yet; a saved population keeps the
settings it was created with:

```yaml
input_dim: 2
y_range: [-350.0, 1.0]   # one scalar per evaluation, maximized
memory_dir: meta-bo-training/conbo-memory/branin_demo
n_evaluations_default: 20
conbo:
  r0: 6                          # random starts for user 1, decreasing by dr per user
  dr: 2
  alpha1: 5.0                    # population weight held at 1 for 5 evaluations,
  alpha2: 0.2                    # then falls by 0.2 per evaluation
  replay_variance_fraction: 0.05 # scale-free replay filter; or replay_max_variance (raw units)
  population_epochs: 1200
  online_epochs: 15
  candidate_grid_points: 40
  seed: 0
```

Example TAF+ config (`taf_plus_server.yaml`, set up for the CHI 2024 App. B
sphere users and their archived population):

```yaml
input_dim: 4
y_range: [[-12.0, 2.0], [-12.0, 2.0], [-12.0, 2.0]]   # one [min, max] per objective
population_dir: benchmark_results/taf_plus_sims/populations/v1_sim2_r01_s0
objective_weights_default: [0.3, 0.5, 0.2]   # alpha; must sum to 1
decay_default: true
d1_default: 8          # decay (d1, d2); this population's search result
d2_default: 0.1
weight_mode_default: poe   # poe | taf_r
rho_default: 1.0
n_iter_default: 10
n_candidates_default: 1024
refine_suggestion_default: true
first_point_default: population   # population | centre
```

`population_dir` must hold a population written by
`TAFPlusPopulation.save` ([TAF+ in Python](taf-plus.md#run-in-python)) or by
`scripts/run_taf_plus_benchmark.py --method population` (under
`test_results/taf_plus/populations/`). Its input dimension and number of
objectives must match `input_dim` and `y_range`. The new user's GPs use
the population's kernel.

Example NAF config (`naf_server.yaml`, set up for the archived Branin AF):

```yaml
input_dim: 2
y_range: [-12.0, 2.0]   # Branin-family tasks stay inside about [-10.3, 1.16]
af_dir: benchmark_results/naf_branin/training/branin_metabo50_v1
af_name: best_af        # best_af (best training return) | af (last iteration)
n_iter_default: null    # null = the AF's training budget T (30 here)
```

`af_dir` must hold a `scripts/train_naf.py` output folder (`<af_name>.pt`
and `naf_config.json`), and the AF's input dimension must match `input_dim`.
The GP hyperparameters and the budget come from the AF's own config.

Example TAF-MOBO config (`taf_mobo_server.yaml`, set up for the TAF+ sphere
users with the archived range-0.1 population as sources):

```yaml
input_dim: 4
y_range: [[-12.0, 2.0], [-12.0, 2.0], [-12.0, 2.0]]   # one [min, max] per objective
v_ref: [-1.0, -1.0, -1.0]   # hypervolume box; defaults to the y_range corners
w_ref: [1.2, 1.2, 1.2]
population_dir: benchmark_results/taf_plus_sims/populations/v1_sim2_r01_s0   # or taf_run_dir
method_default: taf_eipv    # taf_eipv | taf_ehvi
n_iter_default: 10          # total evaluations per session
n_init_default: null        # random designs first; null = 0 for taf_eipv, 1 for taf_ehvi
weight_mode_default: taf_r  # taf_eipv: taf_r | uniform; taf_ehvi: taf_r | taf_r_pareto | taf_m
rho_default: 1.0
n_candidates_default: 1024  # taf_eipv
num_restarts_default: 4     # taf_ehvi
raw_samples_default: 256    # taf_ehvi
```

Set exactly one of `population_dir` (a saved `TAFPlusPopulation`) and
`taf_run_dir` (the TAF run layout with one GP state per objective, as written by
`mobo_taf.write_population_sources`). The sources' input dimension and number of
objectives must match `input_dim` and `y_range`. Choose `v_ref` below each
objective's worst value of interest and `w_ref` above its best: the hypervolume
is measured between them.

Assumptions for server optimization:
- input is normalized to `[0, 1]^d`
- optimization is **maximization**
- you must set `input_dim` and `y_range` in the config file for your application
- current default config is Branin-oriented (`y = -Branin(x)`), so `y_range` is set for flipped Branin scale
- choose backend with `optimizer: bo_botorch` or `bo_scratch` for the generic server
- use dedicated TAF server for `bo_taf` and set `taf_run_dir` in config
- for `bo_scratch`, you can auto-save TAF-compatible `trajectories/*.json` and `gp_states/*.json`
  using `auto_save_scratch_artifacts: true`

## Step 2. WebSocket message protocol (JSON)

Client -> Server:
- `start`: create a new optimization session
- `suggest`: not needed. The server sends the next design after `start` and after each `observe`, and a `suggest` sent while a design is pending returns an `error`.
- `observe`: return evaluation result for the latest suggested design
- `status`: get current progress
- `stop`: stop current session early and return partial results

Server -> Client:
- `suggest`: next design vector `x`
- `done`: optimization finished; returns full trajectory and best solution
- `status`: progress snapshot
- `error`: validation/protocol error

`start` payload example:

```json
{
  "type": "start",
  "n_init": 5,
  "n_iter": 25,
  "seed": 0,
  "num_restarts": 5,
  "raw_samples": 64
}
```

`observe` payload example:

```json
{
  "type": "observe",
  "x": [0.12, 0.88],
  "y": -1.234
}
```

`stop` payload example:

```json
{
  "type": "stop",
  "reason": "user_cancelled"
}
```

## Step 3. Per-server protocols and client loops (Python)

A minimal client for the generic BO server is on the website's Get started page. Notes that hold for every server:

- One WebSocket connection corresponds to one session.
- The server enforces ask/tell order (`suggest` then `observe`).
- `done` includes `x_values`, `y_values`, and `best_y_history` for logging/debugging.
- `stop` returns a `stopped` payload with partial trajectory and best-so-far values.
- `observe.y` must stay within configured `y_range`.

### MOBO server protocol

The MOBO server (`run_mobo_server.py`) uses the same message types as the
generic BO server, but `observe.y` is a **list with one value per objective**
(all maximized):

```json
{
  "type": "observe",
  "x": [0.31, 0.72, 0.05],
  "y": [1.84, -3.97]
}
```

- The config's `y_range` holds one `[min, max]` per objective. Every value must
  lie inside its range, and a scalar `y` or one of the wrong length is rejected.
- `y_range` also sets the hypervolume reference box. `v_ref` defaults to the
  minimums and `w_ref` to the maximums; both can be set in the YAML.
- `start` may override `model` (`independent` / `mt` / `slf`), `acquisition`
  (`ceipv` / `ieipv`), `hyperparameters` (`map` / `slice`), `n_candidates`,
  `n_rescore`, `n_init` (at least 1), `n_iter` and `seed`.
- `done` and `stopped` return `x_values`, `y_values` (n × L), `pareto_x`,
  `pareto_y`, and `hv_history` (one entry per observation).
- `server_scripts/run_mobo_fake_client.py --function oka2` is a ready-made
  client (see [Multi-objective BO](mobo.md)).

### MOBO client loop

Like the generic BO loop, but `y` is a list with one value per objective:

```python
import asyncio
import json
import websockets

async def evaluate(x):
    # Replace with your app: one value per objective, all maximized,
    # each inside the config's y_range.
    return [-(x[0] - 0.3) ** 2, -(x[1] - 0.7) ** 2]

async def main():
    async with websockets.connect("ws://127.0.0.1:8768") as ws:
        await ws.send(json.dumps({"type": "start", "n_init": 5, "n_iter": 10, "seed": 0}))
        msg = json.loads(await ws.recv())
        while msg["type"] == "suggest":
            y = await evaluate(msg["x"])
            await ws.send(json.dumps({"type": "observe", "x": msg["x"], "y": y}))
            msg = json.loads(await ws.recv())
        if msg["type"] == "error":
            raise RuntimeError(msg["message"])
        print("Pareto front:", msg["pareto_y"])
        print("final hypervolume:", msg["hv_history"][-1])

asyncio.run(main())
```

### ConBO server protocol

The ConBO server (`run_conbo_server.py`) uses the same message types as the
generic BO server, with a scalar `y` checked against `y_range`. What differs is
that **one session is one user**, and the population outlives the session:

- **One user at a time.** ConBO must finish users in the order it started them,
  so while a session is open, another `start` gets an `error`. Run participants
  one after another.
- **`start`** may set `user_name` (letters, digits, `_`, `.`, `-`; it must be new
  to the memory; default `user_000`, `user_001`, ...) and `n_evaluations` (this
  user's budget). There is no `n_init`: the user's random starts follow the
  population's schedule (`r0`, `dr`).
- **`suggest`** carries `x`, `phase` (`random` or `bo`), `population_weight`
  (`null` during random starts, then the population's share of the
  acquisition), `user_name` and `user_index`.
- **`done`** arrives when the budget is used up. The user is stored first
  (replay retraining plus save, about 10 s), and `done` reports `stored`,
  `n_users_in_memory` and the `replay` report, besides the usual `x_values`,
  `y_values`, `best_x`, `best_value` and `best_y_history`.
- **`stop`** discards the user by default (the population is unchanged). Send
  `{"type": "stop", "store": true}` to keep a partial session with at least one
  observation. A dropped connection also discards the user.
- `status` reports `n_users_in_memory`, `n_random` and the current
  `population_weight`.
- The memory directory uses the TAF layout, so it can also be passed to the TAF
  server as `taf_run_dir`.

### ConBO client loop

The same loop as the generic BO client; run it once per participant:

```python
import asyncio
import json
import websockets

async def run_trial(x):
    # Replace with your experiment: one scalar, maximized, inside y_range.
    return -((x[0] - 0.25) ** 2 + (x[1] - 0.75) ** 2)

async def main(user_name):
    async with websockets.connect("ws://127.0.0.1:8769") as ws:
        await ws.send(json.dumps({"type": "start", "user_name": user_name, "n_evaluations": 15}))
        msg = json.loads(await ws.recv())
        while msg["type"] == "suggest":
            # msg["phase"] is "random" or "bo"; msg["population_weight"] shows the handover
            y = await run_trial(msg["x"])
            await ws.send(json.dumps({"type": "observe", "x": msg["x"], "y": y}))
            msg = json.loads(await ws.recv())
        if msg["type"] == "error":
            raise RuntimeError(msg["message"])
        print("best:", msg["best_value"], msg["best_x"])
        print("users in memory:", msg["n_users_in_memory"])

asyncio.run(main("participant_01"))
```

### TAF+ server protocol

The TAF+ server (`run_taf_plus_server.py`) uses the same message types as the
MOBO server: `observe.y` is a list with one value per objective, each checked
against its `y_range`. **One session is one new user**:

- **`start`** may override `objective_weights` (one per objective, summing to
  1), `decay`, `d1`, `d2`, `n_iter`, `n_candidates`, `weight_mode`, `rho`,
  `first_point`, `refine_suggestion` and `seed`. There is no `n_init`: the
  first design comes from the population.
- **`suggest`** carries `x`, `phase` (`population` for the first design, then
  `taf_plus`), `iteration` and `decay` (`null` for the first design, then
  `δ(t)`, the population's weight factor).
- **`done`** and **`stopped`** carry `x_values`, `y_values` (`n × L`),
  `weighted_y` (`y @ objective_weights`), `best_so_far`, `decay_history`,
  `objective_weights` and `population_users`.
- `status` reports `n_observations`, `best_weighted_y` and
  `n_population_users`.
- The population is loaded at each `start` and never changed: finished users
  are not added to it. To grow it, add their data with
  `TAFPlusPopulation.add_user` and save it again.

### TAF+ client loop

```python
import asyncio
import json
import websockets

async def run_trial(x):
    # Replace with your experiment: one value per objective, all maximized.
    return [1.0 - 8.0 * ((x[0] - 0.55) ** 2 + (x[1] - 0.4) ** 2), 0.0, 0.0]

async def main():
    async with websockets.connect("ws://127.0.0.1:8770") as ws:
        await ws.send(json.dumps({"type": "start", "objective_weights": [0.3, 0.5, 0.2], "n_iter": 10}))
        msg = json.loads(await ws.recv())
        while msg["type"] == "suggest":
            # msg["phase"] is "population" or "taf_plus"; msg["decay"] is delta(t)
            y = await run_trial(msg["x"])
            await ws.send(json.dumps({"type": "observe", "x": msg["x"], "y": y}))
            msg = json.loads(await ws.recv())
        if msg["type"] == "error":
            raise RuntimeError(msg["message"])
        print("best weighted value:", msg["best_so_far"][-1])

asyncio.run(main())
```

### NAF server protocol

The NAF server (`run_naf_server.py`) uses the same message types as the TAF
server, with a scalar `observe.y` checked against `y_range`. **One session is
one task**:

- **`start`** may set `n_iter`, from 1 up to the AF's training budget `T`
  (default: the config's `n_iter_default`, or `T`). There is no `n_init` and
  no seed: the AF chooses every point, greedily, so a session is deterministic
  given the observed values.
- **`suggest`** carries `x`, `iteration`, `n_iter` and `budget` (`T`).
- **`done`** and **`stopped`** carry `x_values`, `y_values`, `best_so_far`,
  `budget` and `af_dir`.
- `status` reports `n_observations`, `has_pending` and `best_y`.
- **The objective must belong to the family the AF was trained on, with values
  on that family's scale.** The AF reads the GP's raw posterior mean and
  standard deviation, and the GP uses the family's fixed hyperparameters, so
  rescaled or unrelated objectives give the AF inputs it never saw. For your
  own application, train an AF on a family that models it ([Train a neural AF](naf.md#train-a-neural-af)).

### NAF client loop

```python
import asyncio
import json
import websockets

async def run_trial(x):
    # Replace with your experiment: one value, maximized, on the training family's scale.
    return 1.0 - 8.0 * ((x[0] - 0.55) ** 2 + (x[1] - 0.15) ** 2)

async def main():
    async with websockets.connect("ws://127.0.0.1:8771") as ws:
        await ws.send(json.dumps({"type": "start", "n_iter": 15}))
        msg = json.loads(await ws.recv())
        while msg["type"] == "suggest":
            y = await run_trial(msg["x"])
            await ws.send(json.dumps({"type": "observe", "x": msg["x"], "y": y}))
            msg = json.loads(await ws.recv())
        if msg["type"] == "error":
            raise RuntimeError(msg["message"])
        print("best value:", msg["best_so_far"][-1])

asyncio.run(main())
```

### TAF-MOBO server protocol

The TAF-MOBO server (`run_taf_mobo_server.py`) uses the same message types as the
MOBO server: `observe.y` is a list with one value per objective, each checked
against its `y_range`. **One session is one new user**:

- **`start`** may set `method` (`taf_eipv` or `taf_ehvi`), `n_iter` (total
  evaluations), `n_init` (random designs first; TAF-EHVI needs at least 1),
  `weight_mode`, `rho`, `seed`, and `n_candidates` (TAF-EIPV) or
  `num_restarts` and `raw_samples` (TAF-EHVI).
- **`suggest`** carries `x`, `method`, `phase` (`random` for the random
  designs, `sources` for TAF-EIPV's first design from the sources alone, then
  `taf`), `iteration`, `n_iter` and `source_weights` (one per source, `null` for
  random designs).
- **`done`** and **`stopped`** carry `x_values`, `y_values` (`n × L`),
  `pareto_x`, `pareto_y`, `hv_history` (hypervolume above `v_ref` after each
  observation), `v_ref` and `sources`.
- `status` reports `n_observations`, `has_pending`, `hypervolume` and
  `n_sources`.
- A TAF-EIPV suggestion takes about a second with 10 sources, a TAF-EHVI one
  several seconds to half a minute ([computation time](taf-eipv.md#computation-time)).

### TAF-MOBO client loop

```python
import asyncio
import json
import websockets

async def run_trial(x):
    # Replace with your experiment: one value per objective, all maximized,
    # each inside the config's y_range.
    return [1.0 - 8.0 * ((x[0] - 0.55) ** 2 + (x[1] - 0.4) ** 2),
            1.0 - 8.0 * ((x[1] - 0.6) ** 2 + (x[2] - 0.45) ** 2),
            1.0 - 8.0 * ((x[2] - 0.65) ** 2 + (x[3] - 0.35) ** 2)]

async def main():
    async with websockets.connect("ws://127.0.0.1:8772") as ws:
        await ws.send(json.dumps({"type": "start", "method": "taf_eipv", "n_iter": 8}))
        msg = json.loads(await ws.recv())
        while msg["type"] == "suggest":
            # msg["phase"] is "sources" (first design, from earlier users), "random" or "taf"
            y = await run_trial(msg["x"])
            await ws.send(json.dumps({"type": "observe", "x": msg["x"], "y": y}))
            msg = json.loads(await ws.recv())
        if msg["type"] == "error":
            raise RuntimeError(msg["message"])
        print("Pareto front:", msg["pareto_y"])
        print("hypervolume:", msg["hv_history"][-1])

asyncio.run(main())
```

## Step 4. Run the provided fake clients

OpenBO includes a ready-to-run fake client that acts like an external application
and evaluates server-suggested points on Branin:

```bash
uv run python server_scripts/run_fake_client.py --uri ws://127.0.0.1:8765
```

Useful options:

```bash
uv run python server_scripts/run_fake_client.py \
  --uri ws://127.0.0.1:8765 \
  --n-init 2 \
  --n-iter 8 \
  --seed 0 \
  --save-json test_results/trajectories/fake_client_done.json
```

For dedicated TAF server testing:

```bash
uv run python server_scripts/run_fake_client.py \
  --uri ws://127.0.0.1:8766 \
  --n-init 0 \
  --n-iter 10 \
  --seed 0 \
  --save-json test_results/trajectories/fake_client_taf_done.json \
  --plot-x-locations
```

This writes `test_results/trajectories/fake_client_taf_done_x_locations.png` next to the JSON
(same 2D heatmap + iteration-colored points as `scripts/run_benchmark.py --plot-x-locations`).
Use `--plot-output path/to/plot.png` to choose the PNG path; Branin is assumed (`input_dim: 2`).

For the MOBO server (`mobo_server.yaml`, OKA2):

```bash
uv run python server_scripts/run_mobo_fake_client.py --function oka2 --n-init 5 --n-iter 10
```

For the ConBO server, the fake client runs a sequence of simulated users
(shifted, rescaled Branin variants), one session each, and prints each user's
regret as the population grows. User names must be new to the memory, so pick
a fresh `--prefix` for each run:

```bash
uv run python server_scripts/run_conbo_fake_client.py --n-users 4 --n-evaluations 20 --prefix sim
```

For the TAF+ server, the fake client simulates one new sphere user
(CHI 2024 App. B) with similarity range `--range` and prints each
objective vector. It saves the server's final message plus the user's
noise-free regret to `test_results/taf_plus/taf_plus_server_smoke.json`:

```bash
uv run python server_scripts/run_taf_plus_fake_client.py --range 0.1 --user-seed 100
```

For the NAF server, the fake client draws one task from MetaBO's Branin
family (`--task-seed`), prints each point with its regret, and saves the
server's final message plus the task and its simple regret to
`test_results/naf/naf_server_smoke.json`. A full 30-evaluation session takes
about a second:

```bash
uv run python server_scripts/run_naf_fake_client.py --task-seed 100
```

For the TAF-MOBO server, the fake client simulates one new sphere user (CHI 2024
App. B) with similarity range `--range`, prints each objective vector, and saves
the server's final message plus the share of the user's true Pareto front covered
after each evaluation to `test_results/taf_mobo/taf_mobo_server_smoke.json`:

```bash
uv run python server_scripts/run_taf_mobo_fake_client.py --range 0.1 --user-seed 100
uv run python server_scripts/run_taf_mobo_fake_client.py --method taf_ehvi --n-iter 5
```

With the default config, a 10-evaluation TAF-EIPV session takes about 20 seconds;
it covered 83% of the front after its first design and 97% at the end.

**The fake clients write to `test_results/`, which holds committed files.** Two of the
commands on this page overwrite them: `run_naf_fake_client.py` (default output
`test_results/naf/naf_server_smoke.json`) and the TAF example above (`--save-json
test_results/trajectories/fake_client_taf_done.json` and its `_x_locations.png`). Pass
`--output <path>` (or another `--save-json`) to write elsewhere, or restore the files with
`git checkout -- test_results`.
