# TAF+ simulations (CHI 2024 App. B, Simulations 1 and 2), seed 0

TAF+ Phase 4 benchmark (see `docs/plans/taf_plus.md`), run on 2026-10-11 with
the workflow, runner and scripts from the commit that adds this folder.

> **Reproducing:** later code gives slightly different numbers (the BoTorch MOBO rewrite and its per-call seeding change the population models, which are cached here, so only a rebuild is affected; see `docs/guide/results.md`, "Code changes since the archives"). Check out the commit that adds this folder to reproduce them exactly.

**Setup** (`benchmarks/taf_plus_runner.py`): the App. B base function (three
2D spheres on a 4D input, 3 maximized objectives, noise std 0.05). Each
setting samples 20 users with shifts `U(-r/2, r/2)` and output scales
`U(1 - r/2, 1 + r/2)`: 10 population users, modeled with 20 random points plus
20 qLogNEHVI iterations and one GP per objective, and 10 new users. Every method
gets 20 evaluations per new user and is scored on the noise-free weighted sum
`f(x) @ alpha`.

- `taf_plus`: TAF+ with TAF-PoE weights, RBF GPs, a fresh 256-point Sobol pool
  each step plus L-BFGS-B polish, and the decay `(d1, d2)` from the
  leave-one-out search. The first point maximizes the population's
  product-of-experts mean.
- `taf`: the same without decay (the paper's "TAF (multi-objective)").
- `bo`: scratch EI (RBF) on the noisy weighted sum, 5 random starts.
- `random`: uniform random points.

Simulation 2: ranges `r` = 0.05, 0.1, 0.2, 0.3 with `alpha = (0.3, 0.5, 0.2)`.
Simulation 1: `r = 0.01`, six `alpha` vectors, no decay (TAF+ is `taf`).
Deviations from the paper are listed in the runner's docstring and the plan.

Files: `populations/v1_<sim>_s0/` (population GPs and data, plus
`population_modeling.json` with the users and hypervolumes),
`decay/v1_<sim>_a030-050-020_s0.json` (decay search scores and curves),
`trajectories/v1_<method>_<sim>_<alpha>_s0.json` (per new user: shift, scale,
optimum, x, y, noise-free weighted values, per-iteration and best-so-far
regret), `plots/` (`sim2_best.png`, `sim2_instant.png`, `sim2_hv.png`, `sim1_best.png`,
`sim1_instant.png`, `sim1_hv.png`, `decay.png`, `pareto_hypervolume.json`), `run_all.log` and `run_all.sh`. With the
populations cached, `run_all.sh` takes about 2 hours on 8 cores (each decay
search 17-18 minutes with 7 workers, each TAF+/TAF run about 5.5 minutes);
building the five populations took another 48 minutes (7 workers).

## Decay search (Simulation 2)

Score: mean weighted value of each evaluated design, over the 10 leave-one-out
users and 10 iterations (higher is better).

| Range | Chosen `(d1, d2)` | Score | No decay | Paper's choice, our score | Worst pair |
| --- | --- | --- | --- | --- | --- |
| 0.05 | (6, 0.1) | -0.331 | -0.353 | (7, 0.1): -0.341 | -0.739 |
| 0.1 | (8, 0.1) | -0.332 | -0.359 | (4, 0.1): -0.432 | -0.756 |
| 0.2 | (8, 0.1) | -0.184 | -0.183 | (4, 0.2): -0.396 | -0.598 |
| 0.3 | (8, 0.1) | 0.023 | -0.008 | (3, 0.2): -0.440 | -0.635 |

Late, slow decay wins everywhere, within about 0.03 of no decay. The paper found
that more diverse users want earlier, faster decay; here every earlier or faster
decay does worse.

## Simulation 2: mean log10 regret over the 10 new users (lower is better)

Best design so far:

| Range | Method | 1 | 2 | 3 | 5 | 10 | 20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | TAF+ | **-2.47** | **-2.47** | **-2.47** | **-2.47** | **-2.47** | **-2.47** |
| | TAF (no decay) | -2.47 | -2.47 | -2.47 | -2.47 | -2.47 | -2.47 |
| | BO | 0.01 | -0.05 | -0.19 | -0.24 | -0.55 | -1.49 |
| | Random | 0.01 | -0.05 | -0.19 | -0.24 | -0.59 | -0.75 |
| 0.1 | TAF+ | **-1.87** | **-1.87** | **-1.87** | **-1.87** | **-1.87** | **-1.87** |
| | TAF (no decay) | -1.87 | -1.87 | -1.87 | -1.87 | -1.87 | -1.87 |
| | BO | -0.01 | -0.07 | -0.20 | -0.25 | -0.35 | -1.15 |
| | Random | -0.01 | -0.07 | -0.20 | -0.25 | -0.58 | -0.77 |
| 0.2 | TAF+ | **-1.26** | **-1.27** | **-1.27** | **-1.27** | -1.29 | **-1.29** |
| | TAF (no decay) | -1.26 | -1.27 | -1.27 | -1.27 | -1.30 | -1.30 |
| | BO | -0.03 | -0.11 | -0.22 | -0.26 | -0.47 | -1.12 |
| | Random | -0.03 | -0.11 | -0.22 | -0.26 | -0.55 | -0.78 |
| 0.3 | TAF+ | **-0.89** | **-0.91** | **-0.92** | **-0.98** | **-1.06** | -1.07 |
| | TAF (no decay) | -0.89 | -0.91 | -0.92 | -0.98 | -0.99 | -1.03 |
| | BO | -0.05 | -0.15 | -0.25 | -0.28 | -0.47 | **-1.18** |
| | Random | -0.05 | -0.15 | -0.25 | -0.28 | -0.51 | -0.73 |

Each evaluated design (what a user experiences):

| Range | Method | 1 | 2 | 3 | 5 | 10 | 20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | TAF+ | -2.47 | -0.09 | -0.13 | 0.16 | 0.19 | 0.24 |
| | BO | 0.01 | 0.23 | -0.08 | 0.08 | 0.06 | 0.14 |
| 0.1 | TAF+ | -1.87 | -0.39 | -0.42 | 0.29 | 0.07 | 0.30 |
| | BO | -0.01 | 0.22 | -0.07 | 0.08 | -0.09 | -0.16 |
| 0.2 | TAF+ | -1.26 | -0.56 | -0.75 | 0.05 | -0.26 | 0.22 |
| | BO | -0.03 | 0.20 | -0.05 | 0.08 | 0.12 | -0.33 |
| 0.3 | TAF+ | -0.89 | -0.45 | -0.52 | -0.34 | -0.34 | 0.14 |
| | BO | -0.05 | 0.18 | -0.02 | 0.09 | -0.07 | -0.34 |

## Simulation 1: near-identical users, six objective weight vectors

Best design so far, at evaluations 1 and 20:

| alpha | TAF @1 | TAF @20 | BO @20 | Random @20 |
| --- | --- | --- | --- | --- |
| (1, 0, 0) | **-3.81** | **-3.99** | -2.34 | -1.07 |
| (0, 1, 0) | **-3.54** | **-3.59** | -2.16 | -1.13 |
| (0, 0, 1) | **-3.44** | **-3.57** | -2.58 | -1.16 |
| (0.33, 0.33, 0.34) | **-3.47** | **-3.47** | -0.97 | -0.67 |
| (0.5, 0.3, 0.2) | **-3.52** | **-3.52** | -1.19 | -0.70 |
| (0.3, 0.5, 0.2) | **-3.56** | **-3.56** | -1.17 | -0.72 |

Each evaluated design at evaluations 2, 5 and 20 (TAF only): one-hot `alpha`
-2.72 / -1.58 / -1.34, -2.67 / -0.61 / -1.25 and -2.56 / -0.66 / -0.91;
mixed `alpha` -0.04 / 0.25 / 0.20, -0.03 / -0.04 / 0.17 and -0.19 / 0.14 / 0.25
(random search is around 0.0 to 0.1).

## Findings

1. **Transfer works, and fades with user diversity, as the paper reports.** The
   first point alone (the population's product-of-experts mean) reaches a
   regret of 0.003 at range 0.05, 0.013 at 0.1, 0.055 at 0.2 and 0.13 at 0.3,
   and about 0.0003 in Simulation 1, for every `alpha`. That is better than 20
   evaluations of BO up to range 0.2. Simulation 1 confirms the paper's claim
   that TAF+ is robust to the objective weights.
2. **TAF+ barely improves after step 1.** At range 0.3, BO overtakes it by
   evaluation 20. The paper reports convergence to the global maximum within
   3-4 iterations; here the first point is already that good, and nothing after
   it adds much.
3. **After step 1, the designs TAF+ shows are no better than random.** Once
   the first point sits at the population's optimum, each source's predicted
   improvement over its reference (the original's eq. 47) is close to zero, so
   the target's EI takes over and explores. Plan, open question 7.
4. **The objective weighting (CHI eq. 4) makes this much worse.** With one-hot
   `alpha` (a single objective), TAF keeps showing fairly good designs after step
   1 (log10 regret between about -1 and -2). With any mixed `alpha`, the designs jump to random-search
   level from step 2. Eq. 4 sums per-objective clipped improvements, so each
   objective's own optimum looks attractive even when it is poor for the
   weighted sum. Plan, open question 5.
5. **Decay barely matters here.** TAF+ and TAF without decay are nearly
   identical, and the decay search prefers late, slow decay in every range,
   unlike the paper. Decay only acts after step `d1`, by which time the first
   point has already done the work, and handing control to the target sooner
   only starts the exploration sooner.

One seed (10 new users) per setting, as in the paper; the 95% bands in the
plots are over new users.

## TAF-EHVI and plain qLogNEHVI (added 2026-10-21)

Two multi-objective methods run on the same Simulation 2 users, with the same
cached populations and noise streams as the methods above (code: the commit that
adds these trajectories):

- `taf_ehvi`: TAF-EHVI (`optimizers/mobo_taf.py`, contributed in PR #1), with
  the 10 population users as sources (their per-objective GPs, converted by
  `write_population_sources`). It optimizes hypervolume and never sees `alpha`.
  Defaults: objective-wise TAF-R weights, no warmup, no decay.
- `mobo`: plain qLogNEHVI with the same settings, i.e. TAF-EHVI without sources.

Both need one observation before their first suggestion, so they start from one
random point (the same point BO and random search start from), then run 19
qLogNEHVI steps with 256 raw samples and 4 restarts, the population modelling's
settings. On one thread a TAF-EHVI step took about 32 s and a plain qLogNEHVI
step about 23 s (8 runs in parallel on 8 cores: about 103 and 73 minutes per
range).

Besides the weighted-sum regret, every method is also scored on **hypervolume**:
`log10(HV* - HV)` of the noise-free objectives of the designs evaluated so far,
above `v_ref = (-1, -1, -1)`. `HV*` is the user's true Pareto hypervolume (about
7.7), traced by the maximizers of 861 weighted sums; it is accurate to about
0.003, so values are floored at log10(0.01) = -2. (`plots/pareto_hypervolume.json`
caches `HV*` per user.)

Best design so far on the weighted sum, mean log10 regret (the other methods'
rows are in the tables above):

| Range | Method | 1 | 2 | 3 | 5 | 10 | 20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | TAF-EHVI | 0.01 | **-0.44** | **-0.61** | **-0.79** | **-1.25** | **-1.79** |
| | qLogNEHVI | 0.01 | 0.01 | 0.01 | -0.34 | -0.99 | -1.62 |
| 0.1 | TAF-EHVI | -0.01 | **-0.38** | **-0.38** | **-0.59** | **-1.07** | **-1.91** |
| | qLogNEHVI | -0.01 | -0.01 | -0.01 | -0.39 | -1.04 | -1.71 |
| 0.2 | TAF-EHVI | -0.03 | **-0.29** | **-0.29** | **-0.42** | **-1.11** | -1.76 |
| | qLogNEHVI | -0.03 | -0.03 | -0.03 | -0.32 | -1.00 | **-1.80** |
| 0.3 | TAF-EHVI | -0.05 | **-0.20** | **-0.20** | -0.33 | -1.11 | -1.63 |
| | qLogNEHVI | -0.05 | -0.05 | -0.05 | **-0.43** | **-1.14** | **-1.77** |

Hypervolume, mean log10 hypervolume regret, all methods:

| Range | Method | 1 | 2 | 3 | 5 | 10 | 20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | TAF+ | **0.12** | **-0.04** | **-0.08** | **-0.10** | -0.12 | -0.13 |
| | TAF-EHVI | 0.84 | 0.70 | 0.64 | 0.49 | -0.03 | **-0.62** |
| | qLogNEHVI | 0.84 | 0.84 | 0.84 | 0.64 | 0.18 | -0.58 |
| | BO | 0.84 | 0.83 | 0.80 | 0.79 | 0.61 | 0.13 |
| | Random | 0.84 | 0.83 | 0.80 | 0.79 | 0.57 | 0.43 |
| 0.1 | TAF+ | **0.15** | **0.01** | **-0.13** | **-0.16** | **-0.17** | -0.19 |
| | TAF-EHVI | 0.83 | 0.70 | 0.70 | 0.54 | 0.10 | **-0.58** |
| | qLogNEHVI | 0.83 | 0.83 | 0.83 | 0.63 | 0.18 | -0.55 |
| | BO | 0.83 | 0.83 | 0.79 | 0.78 | 0.69 | 0.31 |
| | Random | 0.83 | 0.83 | 0.79 | 0.78 | 0.57 | 0.41 |
| 0.2 | TAF+ | **0.27** | **0.10** | **-0.02** | **-0.03** | **-0.08** | -0.10 |
| | TAF-EHVI | 0.82 | 0.72 | 0.72 | 0.59 | -0.01 | **-0.58** |
| | qLogNEHVI | 0.82 | 0.82 | 0.82 | 0.68 | 0.17 | **-0.58** |
| | BO | 0.82 | 0.80 | 0.77 | 0.76 | 0.64 | 0.34 |
| | Random | 0.82 | 0.80 | 0.77 | 0.76 | 0.56 | 0.38 |
| 0.3 | TAF+ | **0.40** | **0.34** | **0.25** | **0.17** | 0.06 | -0.00 |
| | TAF-EHVI | 0.81 | 0.76 | 0.76 | 0.66 | **0.01** | -0.56 |
| | qLogNEHVI | 0.81 | 0.81 | 0.81 | 0.59 | 0.07 | **-0.58** |
| | BO | 0.81 | 0.78 | 0.74 | 0.73 | 0.66 | 0.30 |
| | Random | 0.81 | 0.78 | 0.74 | 0.73 | 0.55 | 0.38 |

TAF (no decay) is within 0.07 of TAF+ everywhere (see `plots/sim2_hv.png`).

Findings:

1. **Transfer helps TAF-EHVI early, then fades.** From the same random first
   point, its second and third designs are better than plain qLogNEHVI's on
   both metrics in every range (qLogNEHVI makes no progress on those steps). For
   ranges 0.05 to 0.2 it stays ahead to about evaluation 10; at range 0.3, the
   least similar users, qLogNEHVI is ahead from evaluation 5. By evaluation 20
   the two are level: the differences there (at most 0.2 on the weighted sum,
   0.04 on hypervolume) are within the 95% bands of `plots/sim2_best.png` and
   `sim2_hv.png`.
2. **TAF+'s first design is still unmatched early.** Starting from the
   population's product-of-experts optimum, TAF+ covers most of the front's
   hypervolume at once, which TAF-EHVI, starting from one random point, needs
   about 10 evaluations to reach.
3. **The multi-objective methods win late.** By evaluation 20 TAF-EHVI and
   qLogNEHVI have 2.5 to 3.6 times less hypervolume regret than TAF+ (which
   optimizes the weighted sum, not hypervolume, and stalls after its first
   design; finding 2 above) and 5 to 8 times less than
   BO, and on the weighted
   sum they beat TAF+ for the less similar users (ranges 0.2 and 0.3) without
   ever seeing `alpha`. Unlike TAF+ and BO, they also keep proposing good
   designs late (`plots/sim2_instant.png`).
4. **Caveats.** One seed (10 new users), as above. TAF-EHVI ran with its
   defaults only (no warmup, no decay, TAF-R weights); a warmup or a start from
   the sources alone (which it does not support yet) might narrow the early gap
   to TAF+. Its source GPs are the TAF+ population's own (RBF, fitted by
   `GPScratch`), replayed in BoTorch; the means match TAF+'s exactly (tested).

To reproduce: the `taf_ehvi` and `mobo` lines in `run_all.sh`, then
`scripts/plot_taf_plus_results.py --results-dir benchmark_results/taf_plus_sims
--test-id v1 --metric hv` (and `--metric best`, `--metric instant`).

## TAF-EIPV and plain IEIPV (added 2026-10-21)

TAF-EIPV (`optimizers/mobo_taf_eipv.py`, plan `docs/plans/taf_eipv.md`) is the
original TAF on OpenBO's own multi-objective BO: the target's IEIPV (independent
GPs) blended with the hypervolume each source's posterior mean adds to the front
of what that source predicts at the designs already evaluated. That reference
fades as the user explores, as in the original TAF; TAF-EHVI's reference (the
source's own data front) does not. TAF-EIPV uses the 10 population users as
sources directly, with unnormalized TAF-R weights and target weight 3/4, and
**starts from the sources alone** (no random start). `ieipv` is the same
optimizer without sources, starting from one random point. Both use the default
1024-point Sobol pool; a TAF-EIPV step took about 1.3 s on one thread (about 4.5
minutes per range), a plain IEIPV step about 0.4 s.

**Which metric suits which method.** TAF+, TAF and BO optimize the weighted sum
with a known `alpha`; TAF-EIPV, TAF-EHVI, IEIPV and qLogNEHVI optimize
hypervolume and never see `alpha`. Every method is scored on both, from the
designs it evaluated, so each table is home ground for one group: the weighted
sum for TAF+, hypervolume for the multi-objective methods.

Mean log10 hypervolume regret (lower is better):

| Range | Method | 1 | 2 | 3 | 5 | 10 | 20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | TAF-EIPV | **0.08** | **-0.14** | **-0.55** | **-0.67** | **-0.77** | **-0.95** |
| | IEIPV | 0.84 | 0.84 | 0.84 | 0.63 | 0.29 | -0.41 |
| | TAF-EHVI | 0.84 | 0.70 | 0.64 | 0.49 | -0.03 | -0.62 |
| | TAF+ | 0.12 | -0.04 | -0.08 | -0.10 | -0.12 | -0.13 |
| 0.1 | TAF-EIPV | **0.13** | **-0.08** | **-0.39** | **-0.49** | **-0.59** | **-0.82** |
| | IEIPV | 0.83 | 0.83 | 0.83 | 0.62 | 0.23 | -0.40 |
| | TAF-EHVI | 0.83 | 0.70 | 0.70 | 0.54 | 0.10 | -0.58 |
| | TAF+ | 0.15 | 0.01 | -0.13 | -0.16 | -0.17 | -0.19 |
| 0.2 | TAF-EIPV | **0.26** | **0.10** | **-0.09** | **-0.21** | **-0.45** | **-0.64** |
| | IEIPV | 0.82 | 0.82 | 0.82 | 0.63 | 0.14 | -0.50 |
| | TAF-EHVI | 0.82 | 0.72 | 0.72 | 0.59 | -0.01 | -0.58 |
| | TAF+ | 0.27 | **0.10** | -0.02 | -0.03 | -0.08 | -0.10 |
| 0.3 | TAF-EIPV | 0.41 | **0.28** | **0.14** | **0.07** | **-0.17** | -0.45 |
| | IEIPV | 0.81 | 0.81 | 0.80 | 0.62 | 0.20 | -0.49 |
| | TAF-EHVI | 0.81 | 0.76 | 0.76 | 0.66 | 0.01 | -0.56 |
| | TAF+ | **0.40** | 0.34 | 0.25 | 0.17 | 0.06 | -0.00 |

At range 0.3, evaluation 20, qLogNEHVI (-0.58) is best.

Best design so far on the weighted sum, mean log10 regret:

| Range | Method | 1 | 2 | 3 | 5 | 10 | 20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | TAF-EIPV | -1.88 | -1.89 | -1.91 | -1.91 | -2.01 | -2.13 |
| | IEIPV | 0.01 | 0.01 | 0.01 | -0.38 | -0.81 | -1.34 |
| | TAF+ | **-2.47** | **-2.47** | **-2.47** | **-2.47** | **-2.47** | **-2.47** |
| 0.1 | TAF-EIPV | -1.58 | -1.68 | -1.74 | -1.74 | -1.80 | -1.90 |
| | IEIPV | -0.01 | -0.01 | -0.01 | -0.38 | -0.80 | -1.40 |
| | TAF+ | **-1.87** | **-1.87** | **-1.87** | **-1.87** | **-1.87** | -1.87 |
| 0.2 | TAF-EIPV | -1.15 | -1.20 | **-1.45** | **-1.56** | **-1.65** | -1.66 |
| | IEIPV | -0.03 | -0.03 | -0.03 | -0.34 | -0.90 | -1.53 |
| | TAF+ | **-1.26** | **-1.27** | -1.27 | -1.27 | -1.29 | -1.29 |
| 0.3 | TAF-EIPV | -0.84 | **-0.91** | **-1.05** | **-1.10** | **-1.45** | -1.54 |
| | IEIPV | -0.05 | -0.05 | -0.05 | -0.38 | -0.90 | -1.47 |
| | TAF+ | **-0.89** | **-0.91** | -0.92 | -0.98 | -1.06 | -1.07 |

At evaluation 20 the best are TAF-EHVI (-1.91) at range 0.1, and qLogNEHVI at
ranges 0.2 (-1.80) and 0.3 (-1.77); TAF-EIPV leads range 0.05 among the methods
that do not see `alpha`.

Findings:

1. **On hypervolume, TAF-EIPV is the best method at almost every evaluation for
   ranges 0.05 to 0.2.** Its first design, from the sources alone, matches
   TAF+'s (the population's best guess), and unlike TAF+ it keeps improving: at
   range 0.05 it ends at -0.95, against -0.62 for TAF-EHVI, -0.58 for qLogNEHVI,
   -0.41 for IEIPV and -0.13 for TAF+. For the least similar users (range 0.3) it
   leads through about evaluation 13, and the other multi-objective methods
   finish slightly ahead (-0.45 against -0.49 to -0.58).
2. **The fading reference is what makes the difference.** TAF-EHVI transfers
   from the same sources but measures their improvement against their own data
   fronts, which never change, and it cannot start from the sources alone; it
   ends level with qLogNEHVI. TAF-EIPV's sources stop pulling once the user has
   tried what they promised, and the target's IEIPV takes over.
3. **On the weighted sum, TAF+ keeps its edge for similar users.** Knowing
   `alpha`, its first design is unmatched at range 0.05 throughout and at 0.1
   until evaluation 20. For more varied users (ranges 0.2 and 0.3) TAF-EIPV is ahead from
   evaluation 3, without knowing `alpha`.
4. **OpenBO's IEIPV is a reasonable stand-in for qLogNEHVI** at a fraction of the
   cost: level at evaluation 5 and a little behind afterwards (log hypervolume
   regret -0.40 to -0.50 at evaluation 20, against -0.55 to -0.58; both cover
   about 95% of the front), at about 0.4 s per step against about 23 s.
5. **What a user sees** (`plots/sim2_instant.png`): TAF-EIPV's first designs are
   good on the weighted sum too, and its later ones spread along the front, as a
   hypervolume method's should.
6. **Caveats.** One seed and 10 new users per range, simulated users, and
   TAF-EIPV ran with its defaults only (TAF-R weights, no decay).

To reproduce: the `taf_eipv` and `ieipv` lines in `run_all.sh`, then the plot
commands above.
