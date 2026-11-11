# Reproducing the TAF+ and ConBO papers

[← Guide index](README.md) · [README](../../README.md)

Two archives reproduce every simulation benchmark of the TAF+ paper (Liao et al., CHI 2024) and the ConBO paper (Liao et al., CHI 2025), 5 seeds each. The user studies with participants cannot be reproduced. Each archive has a README with the tables, the comparison with the paper and the guesses made, and the scripts that rerun it. The plan and log are in [`docs/plans/paper_reproduction.md`](../plans/paper_reproduction.md).

These runs were made before the fixes of 2026-10-21 (a GP fitted to one value or equal values no longer collapses its variance, and TAF's warmup counts observed points). They are archived as they ran and were not rerun with those fixes.

| Archive | What it holds |
| --- | --- |
| [`benchmark_results/taf_plus_paper/`](../../benchmark_results/taf_plus_paper/README.md) | TAF+, CHI 2024: Simulations 1 to 4, the Appendix C hypervolume curves and an extended Appendix D run |
| [`benchmark_results/conbo_paper/`](../../benchmark_results/conbo_paper/README.md) | ConBO, CHI 2025: Tests 1 to 3 with the paper's settings, 8 methods including the four ablations, the variance maps and computation times |

In short: TAF+ reproduces Simulations 1 to 4 under the best-so-far metric (it is at its final regret at the first design) but not Appendix D, where TAF+ stalls and BO overtakes it. ConBO reproduces Test 3 and the variance-filter effect, but ends behind standard BO and TAF at 30 evaluations in Test 1.

## TAF+ simulations

The [TAF+ benchmark](taf-plus.md#benchmark-and-plot) covers all four simulations. Simulation 3 changes the base function (`--sim sim3_relocated`, `sim3_branin`, `sim3_branin_hartmann`; Tables 7–9, 40 evaluations), and Simulation 4 the population size (`--sim sim4 --n-models k`, the first `k` of 8 population users). `settings_for(sim)` in `benchmarks/taf_plus_runner.py` holds each simulation's sizes, and `--n-evals` overrides the evaluations per new user.

```bash
# Simulation 3, one base function, one seed
uv run python scripts/run_taf_plus_benchmark.py --method decay_search --sim sim3_branin --seed 0 --workers 7 --test-id demo
uv run python scripts/run_taf_plus_benchmark.py --method taf_plus --sim sim3_branin --seed 0 --test-id demo
# Simulation 4 with 4 population models
uv run python scripts/run_taf_plus_benchmark.py --method taf_plus --sim sim4 --n-models 4 --seed 0 --d1 4 --d2 0.2 --test-id demo
# Pool the new users of several seeds in the plots
uv run python scripts/plot_taf_plus_results.py --test-id demo --seeds 0 1 2 3 4
```

The full runs are `run_sim34.sh` and `run_sim12_appD.sh` in the archive.

## ConBO tests

The [ConBO benchmark](conbo.md#benchmark-and-plot) has the paper's three tests and its ablations (`--test branin_paper`, `mccormick_paper`, `branin_revisit`). `scripts/plot_conbo_paper_figures.py` draws the paper's Figures 4–7 from those runs:

```bash
uv run python scripts/run_conbo_benchmark.py --method conbo --test mccormick_paper --seeds 0 1 2 3 4 --test-id demo
uv run python scripts/plot_conbo_paper_figures.py --test-id demo --test mccormick_paper
```

The full run is the archive's `run_all.sh`. `scripts/diagnose_conbo_prior.py` judges a final population BNN as a prior for 30 held-out users (regret of its argmax, rank correlation, 2 sd coverage); it was used to find why replay does not win in Test 1 (the archive README, "Why replay does not win in Test 1").
