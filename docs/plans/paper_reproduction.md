# Reproducing the TAF+ and ConBO paper benchmarks

Goal (set 2026-11-03): reproduce every simulation benchmark in the TAF+ paper (Liao et al., CHI 2024) and the ConBO paper (Liao et al., CHI 2025), archive the results in `benchmark_results/` and upload them. Tick each box when the item is done, and add a line under "Log" for each decision or measurement.

Not reproducible (need real participants): TAF+ user study (14 people; Tables 10–11, Figure 8) and ConBO's 12-person VR keyboard study. Compare with the paper's numbers only.

## Decisions

- Seeds: start with 5 per setting (the ConBO paper does not state its repeat count).
- ConBO: use the paper's settings (alpha1 = 16, alpha2 = 0.1) and re-run Test 1 with them.
- Archive in `benchmark_results/taf_plus_paper/` and `benchmark_results/conbo_paper/`, each with a `run_all.sh`.
- The user pushes; do not commit without the user's approval.

## TAF+ (CHI 2024, Appendix B–D)

- [x] Simulation 1: objective weights (done earlier, seed 0, `benchmark_results/taf_plus_sims`)
- [x] Simulation 2: user similarity (done earlier, seed 0)
- [x] Simulation 1 and 2 with 5 seeds
- [x] Code for Simulations 3 and 4 (done 2026-11-03: base functions in `taf_plus_users.py`, `settings_for`, `--n-models`, tests); seed-0 run started with `benchmark_results/taf_plus_paper/run_sim34.sh`
- [x] Simulation 3: Relocated Spheres (Table 7), seed 0 (5 seeds)
- [x] Simulation 3: Spheres + Branin (Table 8), seed 0
- [x] Simulation 3: Sphere + Branin + Hartmann 3D (Table 9), seed 0
- [x] Simulation 4: number of population models, seed 0 (5 seeds)
- [x] Appendix C: `population_hv.png` written by `plot_taf_plus_results.py` (check once all seeds are in)
- [x] Appendix D (analog: Sim 2 range 0.2, 60 evaluations); NOT reproduced: TAF+ stalls, BO overtakes
- [x] Computation time per iteration (paper: 6.17 s on a ThinkPad i7): about 0.5 s median per TAF+ iteration here, from the run logs (wall time of a method run / (10 users x evaluations), runs sharing the CPU; not a clean measurement)
- [x] README and CLAUDE.md updated, archive and run scripts written

## ConBO (CHI 2025)

- [x] Test 2 and Test 3 settings written into `CONBO_TESTS` (`mccormick_paper`, `branin_revisit`); ablations `conbo_no_gp/no_filter`, `bnn_no_replay`, `bnn_direct_replay`, `taf_paper` added with tests; `scripts/plot_conbo_paper_figures.py` and `benchmark_results/conbo_paper/run_all.sh` written (2026-11-03)
- [x] Test 1 with the paper's alpha (16, 0.1), 5 seeds, 8 methods
- [x] Test 2, 5 seeds, 8 methods
- [x] Test 3 (ConBO vs BNN-wo-Replay, 5 seeds)
- [x] Ablations: conbo_no_gp, conbo_no_filter, bnn_no_replay, bnn_direct_replay
- [x] Computation time plot (panel d of the regret figure)
- [x] Variance map figure
- [x] README and CLAUDE.md updated, archive and run scripts written

## ConBO paper settings (Appendix A, read 2026-11-03)

Eight methods: ConBO, ConBO-wo-GP (alpha1 = total evals, so only the population EI), ConBO-wo-Filter (no variance filter), BNN-wo-Replay (BNN keeps updating on new data, no replay), BNN-Direct-Replay (after each user retrain on all past observations), Single GP, TAF (d1 = 0, d2 as below), BO. Population BNN trained 1200 epochs, 15 epochs per new observation; replay inputs 30x30 grid + 50 random points.

| | Test 1 | Test 2 | Test 3 |
| --- | --- | --- | --- |
| Function | Branin | McCormick | Branin |
| Users | 15 | 15 | 10, then the same 10 again |
| Evals per user | 30 | 7 | 30 |
| shift / scale range | 0.3 / 0.2 | 0.5 / 0.2 | 0.4 / 0.4 |
| r0, dr | 16, 5 | 3, 3 | 16, 5 |
| alpha1, alpha2 | 16, 0.1 | 3, 0.15 | 16, 0.1 |
| lambda | 50 | 10 | 50 |
| wo-GP alpha1 | 30 | 7 | - |
| TAF d2 | 0.05 | 0.15 | - |
| Methods | all 8 | all 8 | ConBO vs BNN-wo-Replay |

Outputs: Figures 4 and 6 (regret curves in three groups: BNN-based, GP-based, with/without user GP; panel d computation time per iteration), Figure 5 (predicted variance map after 15 users for ConBO-wo-GP, BNN-wo-Replay, BNN-Direct-Replay), Figure 7 (re-optimization of users 1-5 and 6-10). All error bars are one standard deviation.

## Log

- 2026-11-03: Figure 18-20 legends give the sizes: Sim 3 population 40/50/60 evaluations (20 random + 20/30/40 iterations), 40 adaptation evaluations; Sim 4 uses 2/4/6/8 models, 10 evaluations. The paper does not give the scaling of Branin and Hartmann-3, so MetaBO's unit-scale normalizations are used and `v_ref` stays (-1, -1, -1). Compute: one seed of Sims 1+2 took about 3 hours on 8 cores, so 5 seeds of everything is roughly a day and a half; run seed 0 first.
- 2026-11-03 (results, seed 0 for TAF+, 5 seeds for ConBO): ConBO Test 3 matches the paper (ConBO beats BNN-wo-Replay when re-optimizing users 1-5, equal on 6-10); the variance maps match its ordering (the filter keeps ConBO-wo-GP at 2.6-13, no filter at 525-1550, no replay up to 1700, direct replay up to 3700). Test 1: BNN-Direct-Replay beats ConBO-wo-GP, which the paper does not show; the GP-based baselines (TAF, standard BO) end better than ConBO at 30 evaluations (ConBO -0.9, TAF -2.5, BO -2.2). TAF's computation time stays flat in our implementation, unlike the paper's linear growth. TAF+ Sim 3/4: with the best-so-far metric TAF+ is at its final regret from evaluation 1 (the population mean's argmax), so the paper's 'converges in 3-4 iterations' holds; with the per-design metric the designs after the first are worse than random (the known stall, `docs/plans/taf_plus.md`).
- 2026-11-04: Test 1 with alpha 5 / 0.2 (`v1alpha5`): ConBO -2.17 at 30 evaluations (BO -2.21), so alpha 16 / 0.1 caused the late gap; but BNN-wo-Replay (-2.47) and BNN-Direct-Replay (-2.97) then beat ConBO, against the paper. Investigated the same day: the replayed variance is the average of the users' own variances and leaves out the spread between users, so the BNN is overconfident about new users (2 sd coverage 0.45; first-10 repeats 57%). Adding that spread (`replay_between_users`, `conbo_total_var`, not in the paper) gives coverage 0.90 and -2.52 at 30 evaluations, about half of the gap to BNN-Direct-Replay (-2.97). The rest is open. Details in `benchmark_results/conbo_paper/README.md`.
- 2026-11-03: plan created.
