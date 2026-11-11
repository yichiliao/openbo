# ConBO paper simulations (CHI 2025, Appendix A), 5 seeds

Reproduction of the three simulation tests of Liao et al., CHI 2025, Appendix A with the paper's settings, run on 2026-11-03. Not reproducible without participants: the 12-person VR keyboard study. Settings per test are in `docs/plans/paper_reproduction.md`; progress log there too. Reproduce with `run_all.sh` (`SEEDS`, `JOBS`, `TESTS` variables); about 1 hour on 11 cores for all three tests.

Eight methods per test (Test 3: two): `conbo`, `conbo_no_gp`, `conbo_no_filter`, `bnn_no_replay`, `bnn_direct_replay`, `bo`, `taf_paper` (TAF with decayed source weights, d1 = 0, d2 = 0.05 / 0.15), `single_gp`. Error bars in the figures are one standard deviation over the (seed, user) pairs.

| Paper item | Test | Files (test id `v1`) |
| --- | --- | --- |
| Figure 4 | `branin_paper` (Test 1: 15 users, 30 evaluations, alpha 16 / 0.1) | `plots/v1_branin_paper_regret.png` |
| Figure 5 | predicted variance after the last user (seed 0) | `plots/v1_branin_paper_variance.png` |
| Figure 6 | `mccormick_paper` (Test 2: 15 users, 7 evaluations, alpha 3 / 0.15) | `plots/v1_mccormick_paper_regret.png` |
| Figure 7 | `branin_revisit` (Test 3: 10 users, then the same users again) | `plots/v1_branin_revisit_revisit.png` |

Panel d of Figures 4 and 6 is the computation time of one iteration per user. Numbers: `plots/*_summary.json`.

## Test 1 (Branin), mean log10 regret of the best design so far, all users

| Method | k = 1 | 5 | 10 | 20 | 30 |
| --- | --- | --- | --- | --- | --- |
| ConBO | 1.01 | 0.53 | 0.43 | 0.03 | -0.87 |
| ConBO-wo-GP | 1.16 | 0.69 | 0.53 | 0.33 | 0.09 |
| ConBO-wo-Filter | 0.90 | 0.49 | 0.31 | -0.07 | -0.70 |
| BNN-wo-Replay | 1.12 | 0.49 | 0.33 | -0.13 | -1.24 |
| BNN-Direct-Replay | 1.18 | 0.48 | 0.18 | -0.27 | -1.55 |
| Single GP | 1.11 | 0.55 | 0.27 | -0.05 | -0.33 |
| TAF (decayed) | 1.08 | 0.41 | 0.14 | -1.01 | -2.48 |
| Standard BO | 1.47 | 0.92 | 0.61 | -0.16 | -2.21 |

### Test 1 with OpenBO's default schedule (alpha 5 / 0.2), test id `v1alpha5`

Same users and seeds, only the schedule changed (`conbo_no_gp`, `bo` and `taf_paper` do not depend on alpha). Run with `--test branin --test-id v1alpha5` for the four methods that use the user GP.

| Method | alpha | k = 1 | 5 | 10 | 20 | 30 |
| --- | --- | --- | --- | --- | --- | --- |
| ConBO | 16 / 0.1 | 1.01 | 0.53 | 0.43 | 0.03 | -0.87 |
| ConBO | 5 / 0.2 | 0.85 | 0.53 | 0.25 | -0.74 | **-2.17** |
| ConBO-wo-Filter | 16 / 0.1 | 0.90 | 0.49 | 0.31 | -0.07 | -0.70 |
| ConBO-wo-Filter | 5 / 0.2 | 0.84 | 0.54 | 0.19 | -0.67 | -2.02 |
| BNN-wo-Replay | 16 / 0.1 | 1.12 | 0.49 | 0.33 | -0.13 | -1.24 |
| BNN-wo-Replay | 5 / 0.2 | 1.03 | 0.55 | -0.01 | -1.05 | -2.47 |
| BNN-Direct-Replay | 16 / 0.1 | 1.18 | 0.48 | 0.18 | -0.27 | -1.55 |
| BNN-Direct-Replay | 5 / 0.2 | 1.18 | 0.45 | 0.07 | -1.07 | **-2.97** |

The schedule explains ConBO's late gap: with alpha 5 / 0.2 ConBO reaches -2.17 at 30 evaluations, level with standard BO (-2.21), and is far ahead of it at 20 (-0.74 against -0.16). With alpha 16 / 0.1 the population model keeps most of the weight for the first 16 evaluations and ConBO is slow to hand over. But with alpha 5 / 0.2 the two BNN ablations without replay beat ConBO at 20 and 30 evaluations, which the paper does not show (it reports ConBO ahead of them); the variance maps, Test 3 and the regret of ConBO-wo-GP are the supports for replay that we do reproduce.

### Why replay does not win in Test 1 (investigation, 2026-11-04)

With alpha 5 / 0.2 the user GP takes over completely from the 10th evaluation, so only the first ten points depend on the population model. Paired over the 75 (seed, user) runs, ConBO is worse than BNN-Direct-Replay by 0.8 log10 at 30 evaluations (95% bootstrap interval 0.5 to 1.1; ConBO is worse in 71% of runs) and worse than BNN-wo-Replay by 0.3 (0.01 to 0.62). The difference starts at user 3, the first with a population.

**Cause: the replayed BNN is overconfident about a new user.** `replay_targets` sets the variance target to the average of the stored users' own GP variances (as the paper's description reads). It leaves out the spread between the users' means, which is most of the uncertainty about a user who has not been seen. Checked on 30 held-out users after the 15-user sequence (seed 0, `diagnostics/`, `scripts/diagnose_conbo_prior.py`):

| Population model | argmax log10 regret | Spearman with the user's function | 2 sd coverage | mean sd / mean abs error |
| --- | --- | --- | --- | --- |
| ConBO (replay) | 0.39 | 0.87 | 0.45 | 5.6 / 19.4 |
| ConBO-wo-Filter | 0.41 | 0.79 | 0.76 | 18.8 / 27.0 |
| BNN-wo-Replay | 0.74 | 0.80 | 0.54 | 9.1 / 23.9 |
| BNN-Direct-Replay | 0.86 | 0.85 | 0.80 | 16.2 / 19.9 |
| ConBO + between-user variance | 0.42 | 0.88 | 0.90 | 20.6 / 19.4 |

The replay BNN has the best mean (lowest argmax regret) but the worst calibration, so its EI under-explores: 57% of a ConBO user's first ten points are within 0.05 of an earlier point, against 37% for BNN-wo-Replay and 16% for BNN-Direct-Replay (users 4 to 15, mean pairwise distance of the first ten points 0.35, 0.47 and 0.57).

**Test of the cause:** `conbo_total_var` (a variation of ours, not in the paper: `ConBOConfig.replay_between_users=True` adds the variance of the users' means to the variance target) fixes the calibration (coverage 0.45 to 0.90, same mean quality), explores more (37% repeats) and improves ConBO at 30 evaluations from -2.17 to -2.52 (paired difference 0.35, interval 0.04 to 0.65), level with BNN-wo-Replay (-2.47). It does not reach BNN-Direct-Replay (-2.97), which explores even more (16% repeats), so calibration explains roughly half of the gap; the rest is not explained. A possible reason is that ConBO is never penalized for repeating the population's argmax, whereas the raw-data BNN's mean is rougher and sends points elsewhere, but this was not tested.

## Test 2 (McCormick), same metric

| Method | k = 1 | 3 | 5 | 7 |
| --- | --- | --- | --- | --- |
| ConBO | 0.34 | 0.12 | -0.03 | -0.20 |
| ConBO-wo-GP | 0.55 | 0.18 | 0.07 | -0.02 |
| BNN-wo-Replay | 0.37 | 0.26 | 0.23 | 0.21 |
| BNN-Direct-Replay | 0.37 | 0.07 | -0.06 | -0.28 |
| TAF (decayed) | 0.23 | -0.10 | -0.25 | -0.48 |
| Standard BO | 0.74 | 0.51 | 0.30 | -0.06 |

## Test 3 (re-optimization from the final population), log10 regret at k = 1, 5, 10, 20, 30

| Users | ConBO | BNN-wo-Replay |
| --- | --- | --- |
| 1-5 | 0.65, 0.34, 0.13, -0.06, -0.41 | 0.99, 0.61, 0.48, 0.24, -0.36 |
| 6-10 | 0.55, 0.38, 0.25, 0.13, -0.62 | 0.77, 0.40, 0.29, -0.06, -0.56 |

## Compared with the paper

- **Reproduced:** BNN-wo-Replay is the worst BNN variant in Test 2; the variance filter regulates the variance (ConBO-wo-GP 2.6-13, no filter 525-1550, no replay up to 1700, direct replay up to 3700); ConBO-wo-GP beats standard BO early; Test 3: ConBO re-optimizes users 1-5 better than BNN-wo-Replay (0.65 against 0.99 at k = 1) and the two are alike on users 6-10; the user GP makes no clear difference on McCormick.
- **Not reproduced:**
  - In Test 1 BNN-Direct-Replay is better than ConBO-wo-GP, which the paper does not show.
  - The user GP helps ConBO (-0.87 against 0.09 at 30 evaluations) but ConBO ends behind standard BO (-2.21) and decayed TAF (-2.48) at 30 evaluations. This is mostly the schedule: with alpha 5 / 0.2 ConBO ends at -2.17 (see above).
  - With alpha 5 / 0.2 the BNN ablations without replay end ahead of ConBO; the investigation above ties this to the replayed variance leaving out the spread between users.
  - The computation time of TAF does not rise with the number of users here (about 0.1 s per iteration throughout), so the paper's crossover with the BNN methods does not appear. The single GP rises from 0.003 s to 0.25 s per iteration over 15 users.
- Errors are large (one standard deviation over 75 user runs), so only the ordering of well-separated curves should be read from the figures.

## Choices the paper leaves open

- alpha1 = 16 is taken literally as Eq. 4's start of the handover (`weight_clock="evaluations"`); the paper's text also says the user model takes over after the fifth iteration, which does not fit alpha1 = 16. With this schedule ConBO uses only the population model for the first 16 evaluations of each user.
- BNN-wo-Replay trains a first BNN on user 1's data (1200 epochs) and then only keeps each user's online updates. BNN-Direct-Replay resets the BNN and trains it on all stored observations with the NLL after each user.
- The paper's TAF baseline is one-objective TAF+ (product-of-experts weights, Matern-5/2 GPs, decay delta(t) with d1 = 0), with the same decaying random starts as ConBO.
- Computation time is measured per suggest + observe step, without the retraining after the user; absolute values depend on the machine.
