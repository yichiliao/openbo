# ConBO Test 1 (Branin): ConBO vs BO, TAF-M, TAF-R and a single GP, 5 seeds

ConBO Phase 6 benchmark (see `docs/plans/conbo.md`), run on 2026-10-10 with
the grid refinement and the benchmark runner from the commit that adds this folder.

> **Reproducing:** later code gives slightly different numbers (the GP fix for all-equal observations changes the TAF baselines: in seed 0, TAF-R keeps proposing one design for two users, and its target GP then saw only identical values; see `docs/guide/results.md`, "Code changes since the archives"). Check out the commit that adds this folder to reproduce them exactly.

**Setup** (the paper's Test 1): for each seed, 15 simulated Branin users in a fixed
order (`generate_user_variants("branin", 15, shift_range=0.3, scale_range=0.2,
seed=seed)`), 30 evaluations each, `r0 = 16` random starts decaying by `dr = 5`
per user, `lambda = 50`, otherwise `ConBOConfig` defaults (40 x 40 candidate grid
with L-BFGS-B refinement, 1,200 replay epochs, 15 online epochs). Seeds 0-4.
All methods see the same users per seed (`benchmarks/conbo_runner.py`):

- `conbo`: `run_conbo_sequence`.
- `bo`: every user fresh, with 16 random starts, then EI on the user GP.
- `taf_m` / `taf_r`: OpenBO's TAF with all previous users as sources, ConBO's
  decaying random starts; user 1 runs as `bo`.
- `single_gp`: one GP on all previous users' data plus the current user's, EI
  against the current user's best, same random starts; hyperparameters fitted
  once per user.

Files: `trajectories/t1_<method>_branin_s<seed>.json` (per user: variant, optimum,
x, y, random starts, seconds), `logs/`, `plots/t1_branin_regret.png`,
`plots/t1_branin_summary.json`, and `run_all.sh` to reproduce everything
(25 single-threaded jobs, 7 at a time: 47 minutes on 8 cores).

**Mean log10 regret, users 4-15, mean ± std over 5 seeds** (lower is better):

| Evaluations | 1 | 3 | 5 | 10 | 20 | 30 | Time per seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ConBO | **0.69 ± 0.16** | **0.50 ± 0.20** | **0.42 ± 0.20** | 0.17 ± 0.15 | **-0.54 ± 0.18** | -1.91 ± 0.17 | 20 min |
| Standard BO | 1.46 ± 0.16 | 1.04 ± 0.18 | 0.93 ± 0.20 | 0.63 ± 0.16 | -0.15 ± 0.12 | **-2.15 ± 0.22** | 4 min |
| TAF-M | 0.94 ± 0.10 | 0.72 ± 0.14 | 0.65 ± 0.16 | 0.25 ± 0.25 | -0.34 ± 0.30 | -0.63 ± 0.57 | 11 min |
| TAF-R | 0.85 ± 0.20 | 0.66 ± 0.22 | 0.47 ± 0.21 | **0.10 ± 0.12** | -0.12 ± 0.20 | -0.16 ± 0.22 | 14 min |
| Single GP | 1.09 ± 0.17 | 0.68 ± 0.10 | 0.51 ± 0.19 | 0.28 ± 0.16 | -0.03 ± 0.17 | -0.19 ± 0.20 | 13 min |

Times are with 7 jobs sharing 8 cores, about 1.7x slower than running alone.

**Seeds (out of 5) in which ConBO beats each method** (same mean, per seed):

| Evaluations | 1 | 5 | 10 | 20 | 30 |
| --- | --- | --- | --- | --- | --- |
| vs Standard BO | 5 | 4 | 5 | 5 | 2 |
| vs TAF-M | 5 | 4 | 2 | 3 | 5 |
| vs TAF-R | 4 | 3 | 2 | 4 | 5 |
| vs Single GP | 5 | 3 | 3 | 5 | 5 |

**Reading it**
- **Early in a session ConBO is best.** At 1 evaluation it beats every method in
  at least 4 of 5 seeds. Against standard BO, which spends 16 evaluations on
  random points, the gap is about 0.5-0.8 log10 up to 10 evaluations and
  ConBO wins in nearly every seed up to 20 evaluations.
- **Around 5-10 evaluations the transfer methods catch up.** TAF-R and the
  single GP are within one standard deviation of ConBO at 5 evaluations, and
  TAF-R is slightly ahead at 10 (0.10 vs 0.17; ConBO wins only 2 of 5 seeds
  against either TAF).
- **At the end, standard BO is slightly better** (-2.15 vs -1.91, ConBO ahead in
  2 of 5 seeds). With 30 evaluations on a 2D function, plain BO catches up by
  itself.
- **The other transfer methods stall.** TAF-M (-0.63), TAF-R (-0.16) and the
  single GP (-0.19) end far above ConBO and BO. TAF keeps re-suggesting source
  users' optima once the target GP is confident: OpenBO's TAF compares each
  source's prediction with a fixed quantile of that source's own data, so the
  pull never fades (the original TAF uses the best source prediction over the
  target's evaluated points; see `docs/plans/conbo.md`, Phase 6). The single GP
  pools shifted users, so near the optimum its mean is biased towards the
  population.
- The per-user panels average over seeds by arrival position. Each seed draws
  its own users, so position 11 in seed 0 is a different user from position 11
  in seed 1; single positions (e.g. ConBO's weaker users 4, 10, 11 and 14) are
  not comparable across seeds.
