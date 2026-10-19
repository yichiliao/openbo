# Project structure

[← Guide index](README.md) · [README](../../README.md)

- `README.md` - short overview, installation and links.
- `docs/guide/` - the user guide: one file per workflow and method.
- `docs/plans/` - design notes and implementation plans (`mobo.md`, `conbo.md`, `taf_plus.md`, `naf.md`).
- `relevant_papers/references.bib` - BibTeX for the papers OpenBO implements directly (PDFs are not redistributed; follow the DOIs).
- `pyproject.toml` - dependencies, build config, and project metadata.
- `configs/` - YAML configs and reusable artifacts.
  - `benchmark.yaml` - benchmark config placeholder (not read by any script).
  - `methods/*.yaml` - method-level config placeholders (not read by any script).
  - `family_splits/*.json` - persisted train/test task-family splits.
  - `server_optimizers/*.yaml` - runtime configs for BO servers (e.g. `input_dim`, `y_range`, TAF defaults).
- `src/openbo/` - main package (`import openbo`; distribution name is `open-bo`).
  - `test_functions/` - synthetic objectives, transforms, family/task variants and registry utilities; multi-objective problems (`multi_objective.py`), ConBO's simulated users (`families.py`), TAF+'s sphere users (`taf_plus_users.py`) and MetaBO's Branin, Goldstein-Price and Hartmann-3 families for NAF (`metabo_families.py`).
  - `models/` - GP implementations and kernels (the NumPy GP in `gp_scratch.py`), including the multi-output GP in `multi_output_gp.py`, a slice sampler in `slice_sampling.py`, and ConBO's MC-dropout population BNN (`bnn.py`) and generative replay (`conbo_replay.py`).
  - `acquisition/` - EI, TAF, EIPV/CEIPV for MOBO (`mobo.py`), ConBO's mixed population/user EI (`conbo.py`), TAF+ with objective weights and decay (`taf_plus.py`; see [`docs/plans/taf_plus.md`](../plans/taf_plus.md)), and the neural AF with its features and candidate set (`naf.py`).
  - `optimizers/` - random search (`random_search.py`), standard BO, TAF; MOBO in `mobo.py` and the qLogNEHVI baseline in `mobo_botorch.py`; continual BO across a sequence of users in `conbo.py`; TAF+ (transfer from earlier users with objective weights and decay) in `bo_taf_plus.py`, with its workflow steps (objective weights from ratings, decay search) in `taf_plus_workflow.py`; NAF / MetaBO (fixed family GPs, RL episodes, PPO training, ask/tell optimizer) in `naf.py`.
  - `server_optimizers/` - WebSocket session adapters for generic BO, TAF, MOBO, ConBO, TAF+ and NAF.
  - `benchmarks/` - `run_simple_benchmark` method dispatcher, Pareto/hypervolume utilities (`hypervolume.py`), the MOBO method dispatcher (`mobo_runner.py`), the ConBO user-sequence dispatcher (`conbo_runner.py`), the TAF+ simulation runner (`taf_plus_runner.py`), the MetaBO benchmark runner (`naf_runner.py`), and placeholder metrics/plotting/seeds modules.
- `scripts/` - local benchmark/training/plot entrypoints.
  - `run_benchmark.py` - run one method on one function; optional `--noisy` and 2D x-location plot.
  - `plot_results.py` - single-function multi-method log-regret plots (rerun or stored trajectories).
  - `create_family_split.py` - create and save train/test splits for task families.
  - `run_family_benchmark.py` - run one method across a task family and save per-task trajectories.
  - `plot_family_results.py` - family mean/std and best-so-far log-regret plots.
  - `plot_family_from_benchmark_runs.py` - merge any number of saved `run_family_benchmark.py` run folders into one comparison plot (custom legend labels per `--run`).
  - `train_taf.py` - produce TAF source artifacts (`trajectories/`, `gp_states/`) from training tasks.
  - `plot_taf_gp_predictions.py` - 2D GP mean/std heatmaps from saved TAF run artifacts.
  - `plot_taf_acquisition_heatmap.py` - visualize stored TAF acquisition query values by iteration.
  - `run_mobo_benchmark.py` - run one MOBO method on one multi-objective problem over several seeds.
  - `plot_mobo_results.py` - hypervolume and log-hypervolume-regret curves and Pareto fronts; merges seeds across test ids.
  - `run_conbo_benchmark.py` - run one ConBO-benchmark method over a sequence of simulated users, for several seeds.
  - `plot_conbo_results.py` - per-user log-regret curves and a summary table for the ConBO benchmark.
  - `run_taf_plus_benchmark.py` - one TAF+ simulation step: population modeling, decay search, or one method on the new users.
  - `train_naf.py` - train a MetaBO neural AF with PPO (Branin; checkpoints and resume).
  - `run_naf_benchmark.py` - run one method of the MetaBO benchmark (NAF, EI, TAF-ME-50, TAF-R-50, random) on unseen test tasks.
  - `plot_naf_results.py` - median log10 regret curves with 30–70% bands, and a NAF training curve.
  - `plot_taf_plus_results.py` - TAF+ regret curves (each design or best so far), decay-search heatmaps and a summary table.
  - `run_botorch_fake_client.py` - toy client loop for BoTorch-oriented external-objective testing.
  - `aggregate_results.py` - placeholder aggregation script.
- `server_scripts/` - server and WebSocket client helpers for ask/tell workflows.
  - `run_bo_server.py` - generic BO server (`bo_botorch` or `bo_scratch`) via config/start message.
  - `run_taf_server.py` - dedicated TAF server.
  - `run_mobo_server.py` - dedicated multi-objective BO server.
  - `run_mobo_fake_client.py` - multi-objective client (`--url`, `--function`, default OKA2) for MOBO server smoke tests.
  - `run_conbo_server.py` - continual BO server (one session per user, population kept in `memory_dir`).
  - `run_conbo_fake_client.py` - simulated user sequence for ConBO server smoke tests (`--url`, `--n-users`, `--n-evaluations`, `--prefix`).
  - `run_taf_plus_server.py` - TAF+ server (one session per new user, fixed population from `population_dir`).
  - `run_taf_plus_fake_client.py` - one simulated sphere user for TAF+ server smoke tests (`--url`, `--range`, `--user-seed`, `--alpha`, `--n-iter`).
  - `run_naf_server.py` - NAF (MetaBO) server (one session per task, trained AF from `af_dir`).
  - `run_naf_fake_client.py` - one simulated task of a MetaBO family for NAF server smoke tests (`--url`, `--family`, `--task-seed`, `--n-iter`).
  - `run_fake_client.py` - fake Branin client; optional 2D x-location plotting.
  - `run_manual_family_train_clients.py` - one session per train variant against scratch server.
  - `run_manual_family_test_clients.py` - one session per test variant against TAF server; optional saved JSON per session.
  - `plot_manual_family_trajectories.py` - manual-workflow regret plotting from saved train/test trajectory artifacts.
- `tests/` - test suite for objectives, GP/acquisition behavior, benchmark paths, and server logic (`*_test.py`; run with `uv run pytest tests`).
- `test_results/` - day-to-day outputs ([Results folders](results.md#results-folder-convention)).
- `benchmark_results/` - archived milestone runs ([Archived results](results.md#archived-benchmark-results)), including `naf_branin/` with the trained Branin NAF.
- `meta-bo-training/` - saved TAF source runs (`taf-gps/`), scratch-server artifacts (`server-scratch-artifacts/`) and NAF training runs (`naf/`).
- `notebooks/` - planned teaching notebooks (`01_bo_from_scratch` … `05_meta_bo_taf_conbo_naf`); currently empty placeholders.

## Stubs and out of scope

- Several modules under `src/openbo` are **stubs or placeholders** for future work and are **not** exposed through the benchmark or server CLIs: `acquisition/pi.py`, `acquisition/ucb.py`, `models/botorch_gp.py`, `benchmarks/metrics.py`, `benchmarks/plotting.py`, `scripts/aggregate_results.py`.
- `configs/methods/*.yaml` and `configs/benchmark.yaml` are placeholders; no script reads them. Scripts are configured through command-line flags (and server YAMLs for the WebSocket servers).
- NAF: MetaBO's GP-prior and real-world tasks are not implemented.
- The teaching notebooks are still empty.
