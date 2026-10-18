# TAF+ simulations (CHI 2024 App. B, Simulations 1 and 2), seed 0

TAF+ Phase 4 benchmark (see `docs/plans/taf_plus.md`), run on 2026-10-11 with
the workflow, runner and scripts from the commit that adds this folder.

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
regret), `plots/` (`sim2_best.png`, `sim2_instant.png`, `sim1_best.png`,
`sim1_instant.png`, `decay.png`), `run_all.log` and `run_all.sh`. With the
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
