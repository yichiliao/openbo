# TAF+ paper simulations (CHI 2024, Appendix B and D), 5 seeds

Reproduction of every simulation in Liao et al., CHI 2024, Appendix B (Simulations 1-4) and D, plus the population-modeling hypervolume curves (Appendix C). Run on 2026-11-03 to 2026-11-04. Not reproducible without participants: the 14-person user study (Tables 10-11, Figure 8). Progress log: `docs/plans/paper_reproduction.md`.

**Setup** (`benchmarks/taf_plus_runner.py`, same as `benchmark_results/taf_plus_sims`): 10 population users and 10 new users per seed, noise std 0.05, population modeling with 20 random points plus 20 qLogNEHVI iterations (Simulation 3: 20/30/40 iterations, read off Figure 18), objective weights (0.3, 0.5, 0.2) unless stated. Methods: `taf_plus` (decay searched per setting), `taf` (no decay), `bo` (weighted-sum EI, 5 random starts), `random`. Seeds 0-4 give different users; curves and tables pool the 10 new users of every seed (50 per setting).

| Paper item | Setting here | Files (test id `v1`) |
| --- | --- | --- |
| Simulation 1 (weights) | range 0.01, six weight vectors, no decay | `sim1_{instant,best}.png` |
| Simulation 2 (similarity) | ranges 0.05, 0.1, 0.2, 0.3, decay searched per range | `sim2_*.png`, `decay.png` |
| Simulation 3 (base functions) | Relocated Spheres, Spheres + Branin, Sphere + Branin + Hartmann 3D (Tables 7-9), range 0.1, 40 evaluations | `sim3_*.png` |
| Simulation 4 (population size) | 3 Spheres, range 0.2, first 2/4/6/8 of 8 population users, decay (4, 0.2), 10 evaluations | `sim4_*.png` |
| Appendix C | hypervolume per iteration while modeling each population | `population_hv.png` |
| Appendix D (extended) | range 0.2, 60 evaluations, TAF+/TAF/BO, test id `v1ext` | `appD_*.png` |

Tables: `table_best.txt`, `table_instant.txt`, `table_appD_*.txt` (mean log10 regret of the noise-free weighted sum). `best` is the best design so far, `instant` the design evaluated at each step. Reproduce with `run_sim34.sh` (Simulations 3 and 4, `SEEDS="0 1 2 3 4"`) and `run_sim12_appD.sh` (Simulations 1, 2, D; one process per seed), then `scripts/plot_taf_plus_results.py --seeds 0 1 2 3 4`.

## Findings (mean log10 regret of the best design so far)

- **Simulations 1-3: the paper's claim holds.** TAF+ (= TAF without decay in Simulation 1) is near the optimum at its first design, because the first point maximizes the population's product-of-experts mean: regret about 10^-3.5 in Simulation 1, 10^-2.5 to 10^-0.9 in Simulation 2 (worse as users get more diverse), 10^-1.4 to 10^-1.9 in Simulation 3. Standard BO needs 20 to 40 evaluations to reach 10^-1.2 to 10^-2.2, and random search stays near 10^-0.6.
- **Simulation 4:** the 2-model population starts worst (-1.21 at evaluation 1); 4, 6 and 8 models are all at -1.27 to -1.28. The paper's trend (more models, better start) holds only between 2 and 4.
- **Decay:** TAF+ and TAF are within 0.02 of each other everywhere; the decay does not help here. The searched decays differ from the paper's (see `decay.png`; the paper chose (7, 0.1), (4, 0.1), (4, 0.2), (3, 0.2) for ranges 0.05-0.3).
- **Appendix D is not reproduced.** The paper says TAF+, TAF and BO all converge near the global maximum. Here TAF+ and TAF stay at their first-design level (-1.39 and -1.37 at 60 evaluations) while BO passes them after about 20 evaluations (-2.10 at 60). This is the known stall after the first design (`docs/plans/taf_plus.md`); by the per-design metric, designs after the first are on average worse than random.

## Deviations and guesses

- Branin and Hartmann-3 in Simulation 3 use MetaBO's unit-scale normalizations (`test_functions/metabo_families.py`); the paper does not give its scaling. The hypervolume reference point stays (-1, -1, -1) for every base function.
- Appendix D: the paper's Figure 22 uses the user-study populations (absolute and relative pointing); we use Simulation 2 at range 0.2 instead.
- The other deviations are those of `benchmark_results/taf_plus_sims/README.md` (qLogNEHVI instead of qEHVI, a refined suggestion, 5 BO starts).
- Computation time: the run logs give a median of about 0.5 s per TAF+ iteration (0.2 to 3 s; several runs shared the CPU), against the paper's 6.17 s on a ThinkPad X1 Carbon. Rough: it is wall time of a method run divided by 10 users times the evaluations.
