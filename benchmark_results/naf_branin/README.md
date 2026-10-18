# NAF (MetaBO) on the Branin family

One run of MetaBO (Volpp et al., ICLR 2020) as implemented in OpenBO, made on
2026-10-11/12 on 8 CPU cores. The user guide is README Section 10; the design
record, with every deviation and open question, is `docs/plans/naf.md`.
`run_all.sh` reproduces everything (about 7 hours).

## Contents

- `training/branin_metabo50_v1/`: the trained neural AF. `best_af.pt` is the
  model the benchmark uses (iteration 193, best mean training return 90.6);
  `af.pt` is the last iteration (235); `training_state.pt` resumes training
  (`scripts/train_naf.py --resume`, with `--output-dir` pointing here or a copy).
  `training/branin_metabo50_v1.log` has one line per PPO iteration.
- `trajectories/v1_<method>_branin.json`: for each of the 100 test tasks, the
  evaluated points, values and simple regret per step.
- `plots/v1_branin.png` (median log10 regret with 30–70% bands) and
  `plots/v1_training.png` (mean final log10 regret per PPO iteration).
- `benchmark.log`: the benchmark's console output.

## Setup

- Family: `s · g(x − t)` with `g` the reference's normalized Branin,
  `t ~ U[−0.1, 0.1]²`, `s ~ U[0.9, 1.1]`.
- Training: MetaBO-50 (the 50 fixed Sobol-grid tasks), PPO with the paper's
  Table 4 settings, 7 rollout workers. Started with `--n-iterations 300` and
  stopped by hand after 236 iterations because the return had stopped
  improving and iterations had slowed from about 52 s to 140 s.
- Test: 100 unseen tasks (test seed 100), 30 evaluations each, the same
  fixed-hyperparameter GP and candidate set for every method.

## Results

Median log10 simple regret over the 100 test tasks (lower is better):

| Method | @1 | @3 | @5 | @10 | @15 | @20 | @30 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NAF | -0.66 | -1.35 | -1.60 | **-3.31** | **-3.35** | -3.36 | -3.36 |
| TAF-ME-50 | **-1.19** | -1.74 | **-1.93** | -2.26 | -2.36 | -2.48 | -2.86 |
| TAF-R-50 | **-1.19** | **-1.77** | -1.80 | -1.95 | -2.20 | -2.54 | -2.97 |
| EI | -0.15 | -0.65 | -0.87 | -1.53 | -2.76 | **-3.37** | **-4.98** |
| Random | -0.19 | -0.75 | -0.86 | -1.13 | -1.25 | -1.31 | -1.58 |

Mean log10 regret over evaluations 1–10 / 1–15 / 1–30: NAF -1.92 / **-2.28** /
-2.67, TAF-ME -1.96 / -2.10 / -2.39, TAF-R -1.84 / -1.95 / -2.37, EI -0.98 /
-1.46 / **-2.77**, random -0.90 / -1.04 / -1.30.

## Findings

- **NAF is the best method from about 10 to 15 evaluations.** It has a better
  best-so-far value than TAF-R on 77% of tasks at evaluation 10.
- **TAF starts better.** NAF's first point is the corner (1, 0) on every task,
  a probe; its second point differs per task (88 distinct of 100). TAF's
  first point is the averaged sources' optimum. MetaBO's reward sums the log
  regret over all 30 steps, so a probe that helps later steps pays off.
- **NAF plateaus after about 8 evaluations.** In evaluations 11–30, 24% of
  its points exactly repeat an earlier point and 46% are within 1e-3 of one
  (EI: 3% and 3%). EI catches up at 20 and wins at 30.
- **Short training.** 236 of the reference's roughly 2000 iterations. The
  training curve improved in jumps (iterations 0–100 and 160–190) with a
  flat stretch between, then got slightly worse after 200.

Next steps are in README Section 10.6 and `docs/plans/naf.md` (open
questions 10–12).
