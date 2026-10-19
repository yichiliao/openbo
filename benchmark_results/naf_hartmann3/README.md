# NAF (MetaBO) on the Hartmann-3 family

One run of MetaBO (Volpp et al., ICLR 2020) as implemented in OpenBO, made on
2026-10-15 on 8 CPU cores. The user guide is `docs/guide/naf.md`; the design
record, with the settings read from the reference code, is `docs/plans/naf.md`
(Phase 5). `run_all.sh` lists the commands (about 3.2 hours for `v1`, 9.5 hours for `v2`).

**Read this first.** There are two runs of the same training, one continuing the
other (the second is the first, resumed from iteration 600 to 2000):

- **`v1`, 600 iterations**: does **not** reproduce the paper's Hartmann-3
  result. NAF leads for about the first 15 evaluations, then stops improving and
  TAF-ME and EI pass it.
- **`v2`, 2000 iterations** (the reference's length): **does** reproduce it. Its
  best checkpoint beats every baseline from the sixth evaluation on (median log10
  regret -3.27 at 30; TAF-ME -3.06, EI -2.61). See "2000 iterations (`v2`)" below.

The gap between the two was training length. Both are **one training run (seed
0)**, tested on one set of 100 tasks without a confidence interval, and `v2` uses
the checkpoint with the best training return, which sits just before an unstable
stretch of the training curve (see below).

## Contents

- `training/hartmann3_metabo50_v1/`: the 600-iteration neural AF (`v1`). `best_af.pt` is the
  model the benchmark uses (iteration 591, best mean training return 58.0);
  `af.pt` is the last iteration (599); `training_state.pt` resumes training
  (`scripts/train_naf.py --family hartmann3 --resume`, with `--output-dir`
  pointing at `training/` or a copy). `training/hartmann3_metabo50_v1.log` has
  one line per PPO iteration.
- `training/hartmann3_metabo50_v2_2000/`: the same run resumed to 2000
  iterations (`v2`): `best_af.pt` is iteration 1454 (best mean training return
  79.2), `af.pt` the last iteration (1999). `training/hartmann3_metabo50_v2_2000.log`
  joins the log of iterations 0–599 and of the resumed session.
- `trajectories/v1_<method>_hartmann3.json`: for each of the 100 test tasks,
  the evaluated points, values and simple regret per step (`v1`, all five
  methods). `trajectories/v2_naf_hartmann3.json` is NAF with the 2000-iteration
  model; the baselines do not depend on the NAF model, so `v2` reuses the `v1`
  ones.
- `plots/v1_hartmann3.png` and `plots/v2_hartmann3.png` (median log10 regret
  with 30–70% bands) and `plots/v1_training.png` and `plots/v2_training.png`
  (mean final log10 regret per PPO iteration).
- There is no `benchmark.log`: the console output of the benchmark was not
  saved. The run was made with the commands in `run_all.sh`, but that script
  has not been run as a whole.

## Setup

- Family: `s · g(x − t)` with `g` the reference's normalized Hartmann-3,
  `-(f + 0.93) / 0.95`, `t ~ U[−0.1, 0.1]³`, `s ~ U[0.9, 1.1]`.
- GP: fixed hyperparameters, zero mean, no output standardization, as in the
  reference. These are OpenBO's own type-2 ML fit (median over 10 tasks; the
  reference's training script also gives the authors' values, which were noticed
  only after this run, see the diagnosis below): lengthscales 0.709, 0.305 and
  0.173, variance 0.694.
- Candidate set: a 12 × 12 × 12 grid (`floor(2000^(1/3))` per dimension, as in
  the reference code) plus 5 local maxima from 2000-point local clouds
  (`N_MS = N_LS = 2000`, `k = 5`): 1733 candidates per step.
- Training: MetaBO-50 (the 50 fixed Sobol-grid tasks), PPO with the paper's
  Table 4 settings, 7 rollout workers, 600 iterations of about 16 s.
- Test: 100 unseen tasks (test seed 100), 30 evaluations each, the same
  fixed-hyperparameter GP and candidate set for every method. TAF uses the same
  50 tasks as sources (100 Sobol points each).

## Results

Median log10 simple regret over the 100 test tasks (lower is better):

| Method | @1 | @3 | @5 | @10 | @15 | @20 | @30 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NAF | 0.56 | -0.07 | -0.81 | **-2.03** | **-2.12** | -2.12 | -2.12 |
| TAF-ME-50 | **-0.46** | -0.85 | -1.11 | -1.81 | -2.06 | **-2.64** | **-3.06** |
| TAF-R-50 | **-0.46** | **-0.96** | **-1.29** | -1.44 | -1.61 | -1.80 | -2.05 |
| EI | 0.53 | 0.50 | 0.48 | -0.18 | -1.33 | -2.05 | -2.61 |
| Random | 0.55 | 0.41 | 0.30 | 0.07 | -0.04 | -0.13 | -0.24 |

Mean log10 regret over evaluations 1–10 / 1–15 / 1–30: NAF -0.81 / -1.26 /
-1.72, TAF-ME **-1.23 / -1.49 / -2.10**, TAF-R -1.18 / -1.33 / -1.63, EI 0.05 /
-0.33 / -1.23, random 0.23 / 0.13 / -0.03.

Share of tasks where NAF's best-so-far value is better than the other method's:

| vs | @1 | @5 | @10 | @15 | @20 | @30 |
| --- | --- | --- | --- | --- | --- | --- |
| TAF-R-50 | 0% | 17% | 82% | 82% | 69% | 51% |
| TAF-ME-50 | 0% | 28% | 57% | 53% | 33% | 14% |
| EI | 38% | 91% | 89% | 72% | 56% | 22% |

## Findings

- **NAF leads only in the middle of the budget.** It is the best method at 10
  and 15 evaluations (by a small margin over TAF-ME), then its median stays at
  -2.12. TAF-ME passes it at evaluation 17 and ends at -3.06; EI
  passes it at evaluation 22 and ends at -2.61. Over all 30 evaluations NAF's
  mean regret (-1.72) is worse than TAF-ME's (-2.10), and it beats TAF-R's
  (-1.63) only narrowly.
- **NAF stops exploring.** In evaluations 11–30, 86% of its points exactly
  repeat an earlier point (EI and TAF-ME: 0%; the archived Branin run: 24%, the
  Goldstein-Price run: 13%), and for 86% of the tasks the best value is the same
  at evaluation 30 as at evaluation 12.
- **It probes first, in a nearly fixed way.** Its first point is the corner
  (0, 1, 1) on every task (median regret 3.6, about EI's 3.4), and its second
  point takes only 14 distinct values over the 100 tasks (Branin: 88,
  Goldstein-Price: 34). TAF's first point, the averaged sources' optimum, has a
  median regret of 0.35.
- **The training curve was nearly flat at the end** (mean return per 100
  iterations: 32, 43, 49, 52, 54, 54; `plots/v1_training.png`), with a plateau
  near -1.8 from iteration 40 to about 160 and a slow fall to about -2.25 after
  400. On Branin the curve also had long plateaus and training to 2000
  iterations helped a lot, so more iterations may help here; this was not tried.
- **Against the paper's Figure 2(c)**, read by eye from a log-scale figure, so
  only approximate: at 30 evaluations the paper's MetaBO-50 and EI are close,
  around -2.4, with TAF-ME about -2.2. This run gives NAF -2.12 (a little
  worse), EI -2.61 (a little better) and TAF-ME -3.06 (much better). The paper
  shows MetaBO as the best method; this run does not.
- **Diagnosis (2026-10-16, after this archive was made; details in
  `docs/plans/naf.md`, "Hartmann-3 diagnosis").** *Ruled out:* the candidate set's
  resolution (the best reachable median log10 regret is -3.59, NAF stops at
  -2.12), greedy versus sampled actions (sampling gives -2.15) and a train/test
  gap (-2.18 on test, -2.24 in training). *What NAF does:* it evaluates about 12
  distinct points, then repeats one point, usually not the best one so far,
  for the rest of the budget; it is in the right basin but not refined (its best
  point is 0.072 from the optimum, TAF-ME's 0.023). Banning repeats only gives
  -2.20. *The GP hyperparameters matter* (TAF-ME ends at -2.27 with lengthscales
  doubled and -3.06 with ours) *but no single scaling matches the paper for both
  EI and TAF-ME.* *The decisive test:* the authors' own pretrained Hartmann-3
  weights, run in OpenBO's environment, reach -3.14 at 30 evaluations with 10%
  repeats (-3.21 with OpenBO's hyperparameters), against -2.12 and 86% here. So the
  environment, the evaluation and the hyperparameters are fine and the gap is in
  this training. *Still open:* training length (600 of about 2000 iterations) is
  the leading explanation, untested.

## 2000 iterations (`v2`)

The 600-iteration run was resumed to 2000 iterations on 2026-10-17 (about 6.3
hours at 16 s per iteration, uninterrupted).
Same 100 test tasks, same GP and candidate set. Median log10 simple regret:

| Method | @1 | @3 | @5 | @10 | @15 | @20 | @30 | mean 1–30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NAF, 2000 iterations (checkpoint 1454) | 0.25 | 0.21 | -0.74 | **-2.61** | **-3.22** | **-3.26** | **-3.27** | **-2.61** |
| NAF, 2000 iterations, last iteration (1999) | 0.56 | 0.24 | -0.19 | -0.76 | -2.18 | -3.26 | -3.30 | -1.91 |
| NAF, 600 iterations (`v1`) | 0.56 | -0.07 | -0.81 | -2.03 | -2.12 | -2.12 | -2.12 | -1.72 |
| TAF-ME-50 | **-0.46** | **-0.85** | -1.11 | -1.81 | -2.06 | -2.64 | -3.06 | -2.10 |
| TAF-R-50 | **-0.46** | **-0.96** | **-1.29** | -1.44 | -1.61 | -1.80 | -2.05 | -1.63 |
| EI | 0.53 | 0.50 | 0.48 | -0.18 | -1.33 | -2.05 | -2.61 | -1.23 |

The last-iteration row was measured for this README and its trajectories are not
archived (`af.pt` in `training/hartmann3_metabo50_v2_2000/` reproduces it).

Share of tasks where the 2000-iteration NAF has a better best-so-far value:

| vs | @1 | @5 | @10 | @15 | @20 | @30 |
| --- | --- | --- | --- | --- | --- | --- |
| TAF-ME-50 | 6% | 26% | 83% | 95% | 83% | 63% |
| TAF-R-50 | 6% | 19% | 97% | 99% | 99% | 96% |
| EI | 86% | 90% | 94% | 96% | 92% | 85% |
| NAF, 600 iterations | 100% | 51% | 90% | 97% | 98% | 98% |

- **The paper's result is reproduced.** The best checkpoint is the best method
  from the sixth evaluation on and flat at -3.27 from about evaluation 15. At 30
  evaluations TAF-ME is at -3.06 (0.2 worse, 63% of tasks go to NAF) and EI at -2.61
  (0.7 worse). Over all 30 evaluations NAF's mean regret (-2.61) is the lowest.
  TAF still starts better, and NAF is behind both TAFs until evaluation 5.
- **The 600-iteration stall was training length.** Repeats remain: in
  evaluations 11–30, 56% of the points are exact repeats (86% for the 600-iteration
  model; 25% for the last-iteration model). They now follow a much better search
  (regret -3.27 against -2.12 at that point), which is my reading, not a separate
  test. The model evaluates a median of 18.5 distinct points per task (13 for the
  600-iteration model).
- **Against the paper's Figure 2(c)** (by eye, approximate): MetaBO about -2.4
  and EI about -2.4 at 30 evaluations; here -3.27 and -2.61. The authors'
  own pretrained Hartmann-3 weights, run in OpenBO's environment (see
  `docs/plans/naf.md`, "Hartmann-3 diagnosis"), give -3.14, so this run is at the
  level of the authors' model, and slightly better.
- **The candidate set limits the regret.** The best median log10 regret the
  candidate set can reach is -3.59, so the flat part near -3.27 is close to that
  floor.
- **The training curve is not stable.** The mean final log10 regret per 200
  iterations was -1.7, -2.1, -2.2, -2.3, -2.8, -3.0, -3.1, -3.1, -3.0 and -3.0:
  it fell in steps, with a clear one near iteration 800, reached its best near
  1450 (-3.3, where the checkpoint was taken) and then showed spikes around 1520,
  1620, 1760 and 1960 that it recovered from (`plots/v2_training.png`). The last
  iteration's model is worse early (-0.76 at 10 evaluations) and equal late
  (-3.30 at 30), so taking the best-return checkpoint matters mostly for the
  early evaluations, which the training reward weights. The cause of the
  spikes is not known.
- **Limits.** One run, one seed, no confidence interval. The checkpoint was chosen
  by training return, so the `v2` row is the best case of this run. The
  baselines are the `v1` ones with OpenBO's own GP hyperparameters, which are
  close to the authors' (Hartmann-3: lengthscales 0.709, 0.305, 0.173 and
  variance 0.694 against 0.716, 0.298, 0.186 and 0.83).
