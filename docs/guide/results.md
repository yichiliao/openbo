# Results and archived benchmarks

[← Guide index](README.md) · [README](../../README.md)

## Results folder convention

By default, artifacts are organized under `test_results/`:

- `test_results/trajectories/`
  - single-function run JSONs:
    - `{test_id}_{method}_{function}.json`
  - family run folders:
    - `{test_id}_{method}_{base_function}_{subset}/`
    - containing one task file per task plus `summary.json`
- `test_results/plots/`
  - single-function and family plot PNGs
- `test_results/mobo/`
  - `trajectories/{test_id}_{method}_{problem}.json` (all seeds of one method)
    and `plots/` from the MOBO benchmark scripts, plus MOBO server smoke-test JSON
- `test_results/conbo/`
  - `trajectories/{test_id}_{method}_{test}_s{seed}.json` (one user sequence per file)
    and `plots/` from the ConBO benchmark scripts, plus ConBO server smoke-test JSON
    (`<prefix>_server_smoke.json`)
- `test_results/taf_plus/`
  - `populations/{test_id}_{sim}_s{seed}/` (cached population models),
    `decay/{test_id}_{sim}_{alpha}_s{seed}.json` (decay searches),
    `trajectories/{test_id}_{method}_{sim}_{alpha}_s{seed}.json` and `plots/`,
    plus TAF+ server smoke-test JSON (`taf_plus_server_smoke.json`)
- `test_results/naf/`
  - `trajectories/{test_id}_{method}_{family}.json` (all test tasks of one method)
    and `plots/` (`{test_id}_{family}.png`, `{test_id}_training.png`), plus NAF
    server smoke-test JSON (`naf_server_smoke.json`)
- `meta-bo-training/naf/{run_id}/` - NAF training runs (`best_af.pt`, `af.pt`,
  `naf_config.json`, `training_history.json`, `training_state.pt`)

Practical workflow convention:
- Keep using `test_results/` as the default for day-to-day experiments and quick iterations.
- Use `--results-dir benchmark_results` for large, milestone-style benchmark runs that you want to keep stable over time.
- Prefer unique `--test-id` values for archived runs in `benchmark_results/` to avoid accidental overwrite.

## Archived benchmark results

The headline tables and plots are on the website's [Benchmarks](https://www.csie.ntu.edu.tw/~yichi/openbo-website/benchmarks.html) page. Each archive under `benchmark_results/` has a `run_all.sh` (or the commands) that reproduces it, and most have their own README with the full tables and findings:

| Archive | What it holds | Run your own |
| --- | --- | --- |
| [`benchmark_scratch_botorch/`](../../benchmark_results/benchmark_scratch_botorch/) | Scratch BO (`bo_scratch_grid`, `bo_scratch_multistart`) vs BoTorch and random search, with (`benchmark_BO_BoTorch_with_noise/`) and without (`benchmark_BO_BoTorch_no_noise/`) output noise, plus search-behaviour plots (`visual_search_behaviors_with_noise/`) | [Single-function benchmarks](single-function.md), [Families](families.md) |
| [`benchmark_taf/`](../../benchmark_results/benchmark_taf/) | `benchmark_taf_bo_{ackley,branin,hartmann6}/`: TAF-M / TAF-R family comparisons, plus 2D search-path plots for TAF and BoTorch | [TAF](taf.md) |
| [`conbo_branin_quick/`](../../benchmark_results/conbo_branin_quick/) | ConBO vs standard BO, Test 1, one seed ([README](../../benchmark_results/conbo_branin_quick/README.md)) | [ConBO benchmark](conbo.md#benchmark-and-plot) |
| [`conbo_branin_5seeds/`](../../benchmark_results/conbo_branin_5seeds/) | ConBO vs standard BO, TAF-M, TAF-R and a single GP, Test 1, 5 seeds; about 47 minutes on 8 cores ([README](../../benchmark_results/conbo_branin_5seeds/README.md)) | [ConBO benchmark](conbo.md#benchmark-and-plot) |
| [`mobo_2obj/`](../../benchmark_results/mobo_2obj/) | OKA2 and DTLZ1a, 5 methods × 3 seeds × 40 iterations | [MOBO benchmark](mobo.md#benchmark-and-plot) |
| [`taf_plus_sims/`](../../benchmark_results/taf_plus_sims/) | CHI 2024 App. B Simulations 1 and 2, one seed, with cached populations and decay searches, plus TAF-EHVI and plain qLogNEHVI on the Simulation 2 users (2026-10-21); about 3 hours on 8 cores, plus about 1¾ hours for TAF-EHVI ([README](../../benchmark_results/taf_plus_sims/README.md)) | [TAF+ benchmark](taf-plus.md#benchmark-and-plot) |
| [`naf_branin/`](../../benchmark_results/naf_branin/) | The trained Branin AF (`training/branin_metabo50_v1/`, 236 iterations) and 5 methods × 100 test tasks ([README](../../benchmark_results/naf_branin/README.md)) | [Train and benchmark](naf.md#train-a-neural-af) |
| [`naf_goldstein_price/`](../../benchmark_results/naf_goldstein_price/) | The same for Goldstein-Price, 600 iterations ([README](../../benchmark_results/naf_goldstein_price/README.md)) | [Train and benchmark](naf.md#train-a-neural-af) |
| [`naf_hartmann3/`](../../benchmark_results/naf_hartmann3/) | The same for Hartmann-3, 600 and 2000 iterations ([README](../../benchmark_results/naf_hartmann3/README.md)) | [Train and benchmark](naf.md#train-a-neural-af) |

Saved TAF source runs live under `meta-bo-training/taf-gps/` (`branin_train_v1`
from `train_taf.py`, `branin_server_end_to_end` from the [end-to-end TAF workflow](taf.md#end-to-end-taf-workflow-family-split--scratch-gps--taf) server
workflow). All 12 tasks of `branin_train_v1` are loaded (earlier, 11 of them
sat in a nested folder the loader did not read; see [TAF workflow](taf.md#transfer-acquisition-function-taf-workflow)).

**Multi-objective BO** (`mobo_2obj/`, no README of its own). The budget is
reduced: 5 initial points plus 40 iterations, 3 seeds (the paper uses 100
iterations and 50 repeats). On OKA2, CEIPV-SLF, IEIPV and BoTorch end within
noise of each other (BoTorch rises fastest early); CEIPV-MT stalls, as its
single shared kernel suits OKA2 poorly. On DTLZ1a the NumPy methods reach a
final log10 hypervolume regret of about 1.3–1.7 while BoTorch plateaus near
3.2. With 3 seeds, the differences between CEIPV-SLF and IEIPV are not
significant. The strongly correlated SLF-GP problems, where the paper reports
the clearest gains from modelling correlation, and VLMOP3 have not been run yet.

**NAF on Branin, trained longer.** These runs are not in `naf_branin/`, whose
README covers only the archived 236-iteration model. Median log10 simple
regret on the same 100 test tasks:

| Method | @1 | @5 | @10 | @15 | @20 | @30 |
| --- | --- | --- | --- | --- | --- | --- |
| NAF, archived (236 iterations) | -0.66 | -1.60 | -3.31 | -3.35 | -3.36 | -3.36 |
| NAF, 313 iterations | -0.66 | -1.60 | -3.82 | -3.88 | -3.88 | -3.88 |
| NAF, 2000 iterations | -0.75 | -1.57 | **-4.97** | **-5.62** | **-5.62** | **-5.62** |
| EI | -0.15 | -0.87 | -1.53 | -2.76 | -3.37 | **-4.98** |

- With 313 iterations (`meta-bo-training/naf/branin_metabo50_v1_long`, best at
  307; `test_results/naf/trajectories/v2_long313_naf_branin.json`) the plateau
  is -3.88 instead of -3.36, and the mean over evaluations 1–30 (-2.95) beats
  EI (-2.77) and both TAFs.
- With 2000 iterations, the reference's length
  (`meta-bo-training/naf/branin_metabo50_v2_2000`, best checkpoint at iteration
  1585; `test_results/naf/trajectories/v3_2000_naf_branin.json`), the mean over
  evaluations 1–30 is -4.62. From 10 evaluations on it is below EI at every
  budget, and close to the paper's Figure 2(a) (about -5). It still repeats
  some points (on average 19.6 of the 30 evaluations are distinct, against
  24.4 for the archived model).
- **Read the 2000-iteration row with care.** It comes from **one training run
  (seed 0)** on one set of 100 test tasks, with no confidence interval, and
  from the checkpoint with the best training return (iteration 1585). The
  training curve has an abrupt setback right after that iteration: the mean
  final log10 regret was about -5.2 around iteration 1585, about -3.7 around
  1620 and about -4.4 at the end (`test_results/naf/plots/v3_2000_training.png`).
  The model of the **last** iteration (`af.pt`) reaches a median of -4.88 at 30
  evaluations (-4.49 at 10) and a mean over evaluations 1–30 of -3.51, so it is
  slightly behind EI (-4.98) at 30 evaluations. The table is therefore what the
  best checkpoint of this run reaches, an upper estimate of what a training
  gives, not an average. The regret curves against the baselines (the archived
  runs) are in `test_results/naf/plots/v3_2000_branin.png`.

### Code changes since the archives

Each archive was made with the code of the commit that adds it; check out that commit to reproduce it exactly. These later changes (2026-10-21, from PR #1) are bug fixes that change some numbers:

| Change | Archives affected |
| --- | --- |
| `GPScratch` and `MultiOutputGP` no longer collapse to near-zero variance on one observation or all-equal values (the std is set to 1 instead of 1e-12) | `conbo_branin_5seeds/` (TAF baselines: TAF-R in seed 0 repeats one design for two users, so its target GP saw only identical values; checked by counting, 2026-10-21) |
| Output noise has its own random stream instead of sharing `--seed` with the optimizer | `benchmark_scratch_botorch/` (noisy runs) |
| The BoTorch optimizers seed torch per `suggest()` call instead of once globally, and the MOBO one models the objectives as one batched `SingleTaskGP` | `benchmark_scratch_botorch/`, `benchmark_taf/` (`bo_botorch` runs), `mobo_2obj/` (`mobo_botorch`), `taf_plus_sims/` (population models) |
| Family variants report their reachable optimum when the input shift clips the base optimum out of `[0, 1]^d` (up to 3D) | family runs in `benchmark_scratch_botorch/` and `benchmark_taf/` whose variants clip |

The NAF archives are unaffected.
