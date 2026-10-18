# NAF (MetaBO) on the Goldstein-Price family

One run of MetaBO (Volpp et al., ICLR 2020) as implemented in OpenBO, made on
2026-10-15 on 8 CPU cores. The user guide is README Section 10; the design
record, with the settings read from the reference code, is `docs/plans/naf.md`
(Phase 5, "Goldstein-Price and Hartmann-3: settings"). `run_all.sh` lists the
commands (about 1.6 hours).

**Read this first.** The results below are for **one training run (seed 0)**,
evaluated on one set of 100 test tasks without a confidence interval. The
model the benchmark uses is the checkpoint with the best training return,
iteration 499. The same run was later continued to 2000 iterations (the
reference's length) and **did not improve**: the best-return checkpoint stayed at
iteration 499, and the model of iteration 1999 is worse (see "2000 iterations
without improvement" below). So 600 iterations was enough here, unlike on Branin
and Hartmann-3. The comparison with the paper is made by eye from its Figure
2(b), which is plotted on a log scale.

## Contents

- `training/goldstein_price_metabo50_v1/`: the trained neural AF. `best_af.pt`
  is the model the benchmark uses (iteration 499, best mean training return
  55.5); `af.pt` is the last iteration (599); `training_state.pt` resumes
  training (`scripts/train_naf.py --family goldstein_price --resume`, with
  `--output-dir` pointing at `training/` or a copy).
  `training/goldstein_price_metabo50_v1.log` has one line per PPO iteration.
- `trajectories/v1_<method>_goldstein_price.json`: for each of the 100 test
  tasks, the evaluated points, values and simple regret per step.
- `plots/v1_goldstein_price.png` (median log10 regret with 30–70% bands) and
  `plots/v1_training.png` (mean final log10 regret per PPO iteration).
- There is no `benchmark.log`: the console output of the benchmark was not
  saved. The run was made with the commands in `run_all.sh`, but that script
  has not been run as a whole.

## Setup

- Family: `s · g(x − t)` with `g` the reference's log-normalized
  Goldstein-Price, `-(log f − 8.693) / 2.427` on `[-2, 2]²` mapped to the unit
  square, `t ~ U[−0.1, 0.1]²`, `s ~ U[0.9, 1.1]`.
- GP: fixed hyperparameters, zero mean, no output standardization, as in the
  reference. These are OpenBO's own type-2 ML fit (median over 10 tasks; the
  reference's training script also gives the authors' values, which were noticed
  only after this run, see the diagnosis below): lengthscales 0.109 and 0.087,
  variance 0.652. The short lengthscales reflect the function's sharp peak.
- Candidate set: a 31 × 31 grid plus 5 local maxima from 1000-point local
  clouds (`N_MS = N_LS = 1000`, `k = 5`), the same as for Branin.
- Training: MetaBO-50 (the 50 fixed Sobol-grid tasks), PPO with the paper's
  Table 4 settings, 7 rollout workers, 600 iterations of about 8.6 s.
- Test: 100 unseen tasks (test seed 100), 30 evaluations each, the same
  fixed-hyperparameter GP and candidate set for every method. TAF uses the same
  50 tasks as sources (100 Sobol points each).

## Results

Median log10 simple regret over the 100 test tasks (lower is better):

| Method | @1 | @3 | @5 | @10 | @15 | @20 | @30 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NAF | 0.56 | 0.07 | **-0.50** | **-1.59** | **-1.99** | **-2.17** | **-2.37** |
| TAF-ME-50 | **0.05** | -0.13 | -0.27 | -0.40 | -1.29 | -1.56 | -1.68 |
| TAF-R-50 | **0.05** | **-0.15** | -0.30 | -0.46 | -0.99 | -1.14 | -1.32 |
| EI | 0.52 | 0.37 | 0.26 | 0.03 | -0.02 | -0.12 | -1.28 |
| Random | 0.49 | 0.37 | 0.28 | 0.19 | 0.12 | 0.11 | 0.04 |

Mean log10 regret over evaluations 1–10 / 1–15 / 1–30: NAF **-0.66 / -1.07 /
-1.72**, TAF-ME -0.35 / -0.58 / -1.13, TAF-R -0.39 / -0.58 / -0.92, EI 0.19 /
0.04 / -0.40, random 0.26 / 0.21 / 0.11.

Share of tasks where NAF's best-so-far value is better than the other method's:

| vs | @1 | @5 | @10 | @20 | @30 |
| --- | --- | --- | --- | --- | --- |
| TAF-R-50 | 0% | 52% | 79% | 83% | 82% |
| TAF-ME-50 | 0% | 52% | 87% | 68% | 65% |
| EI | 35% | 94% | 93% | 88% | 85% |

## Findings

- **NAF is the best method from the fifth evaluation on**, at every budget up
  to 30, and its mean regret over 30 evaluations is the lowest of all methods.
  Unlike on Branin, EI does not catch up: it stays near regret 1 until about
  evaluation 20 and reaches -1.28 at 30, because the function's peak is sharp.
- **TAF starts better.** NAF's first point is the same point on every task,
  (0.702, 0.0), a probe (median first-step regret 3.6, log10 0.56); its second
  point differs per task (34 distinct of 100). TAF's first point is the
  averaged sources' optimum (log10 regret about 0.05). NAF is still
  behind both TAFs at evaluation 4 (-0.19 against -0.25 and -0.29) and passes them at 5. Both TAF variants improve sharply around evaluations
  10–13 (the plot), which this run does not explain.
- **NAF repeats points less than on Branin.** In evaluations 11–30, 13% of its
  points exactly repeat an earlier point and 14% are within 1e-3 of one (EI:
  0% and 0%; the archived Branin run: 24% and 46%).
- **The training curve has a long plateau.** The mean final log10 regret falls
  from about 0 to -2.1 in the first 80 iterations, stays at -2.1 to -2.2 until
  about iteration 450, then dips to about -2.4 to -2.5 around iterations 450 to
  600 (`plots/v1_training.png`), where the best checkpoint was taken. I first
  read the dip as the start of a longer improvement; continuing to 2000
  iterations showed it was not (below).
- **Against the paper's Figure 2(b)** (read by eye from a log-scale plot, so only
  approximate): at 30 evaluations the paper's MetaBO-50 is near -2.5,
  TAF-ME about -2, TAF-R about -1.3, EI about -1 and random about 0. This
  run gives -2.37, -1.68, -1.32, -1.28 and 0.04.

## 2000 iterations without improvement (`v2`)

The run was resumed from iteration 600 to 2000 on 2026-10-17 (3.3 hours at 8.4 s
per iteration, uninterrupted).

- **No better model came out of it.** The best mean training return of the
  whole run is still iteration 499 (55.5); in iterations 600–1999 it never
  exceeded 48.9 (iteration 1836). `best_af.pt` is byte-identical to the 600-
  iteration model, so the benchmark above (`v1`) is also the benchmark of the
  2000-iteration run's best checkpoint.
- **The training curve did not keep improving.** Mean training return per 200
  iterations: 25.7, 36.3, 44.2, 36.2, 35.1, 37.3, 38.3, 38.7, 40.9, 38.2; mean
  final log10 regret about -2.1 to -2.4 from iteration 80 to 2000
  (`plots/v2_training.png`). The peak near iterations 450–550 faded again by
  iteration 650. Part of the return of the best iteration is luck: the best
  single iteration (55.5) is well above the 50-iteration mean around it (45 to
  47), and a checkpoint chosen as the maximum of 2000 noisy returns is picked
  partly for that. The test results above are not affected, since the test tasks
  are independent of the choice.
- **The last iteration's model is worse than the 600-iteration one.** Median log10
  regret on the same 100 test tasks (`trajectories/v2_last_naf_goldstein_price.json`,
  model `training/goldstein_price_metabo50_v2_2000/af.pt`; load it with
  `NAFSequentialOptimizer.from_directory(dir, name="af")`):

  | NAF model | @1 | @5 | @10 | @15 | @20 | @30 | mean 1–30 | repeats 11–30 |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | 600 iterations, best checkpoint 499 (`v1`) | 0.56 | **-0.50** | **-1.59** | **-1.99** | **-2.17** | **-2.37** | **-1.72** | 13% |
  | 2000 iterations, last iteration 1999 | 0.67 | 0.40 | -0.36 | -1.51 | -1.75 | -1.97 | -1.09 | 1% |

  It has a better value than the 600-iteration model on 1%, 12%, 33% and 41% of
  tasks at evaluations 5, 10, 20 and 30. It still beats EI at 30 evaluations
  (-1.97 against -1.28) and TAF-R (-1.32), but not TAF-ME clearly (-1.68), and its
  mean over 30 evaluations (-1.09) is about TAF-ME's (-1.13).
- **What it means.** At 600 iterations NAF is already at the paper's level on this
  family (-2.37 against about -2.5 read by eye), and further training with this
  setup does not move it. It is not a general rule: Branin and Hartmann-3 needed
  2000 iterations and improved a lot. Why Goldstein-Price differs is not known.
  One run, one seed.
- **Files.** `training/goldstein_price_metabo50_v2_2000/` holds `af.pt` (the last
  iteration), `naf_config.json` and `training_history.json` (all 2000 iterations);
  `best_af.pt` (same as `v1`) and `training_state.pt` are not kept again, so that run
  cannot be resumed beyond 2000 from this folder.
  `training/goldstein_price_metabo50_v2_2000.log` joins the logs of the two sessions.
