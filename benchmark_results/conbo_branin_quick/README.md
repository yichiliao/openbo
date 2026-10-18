# ConBO vs standard BO: first look (Branin, one seed)

A quick check run after ConBO phase 5 (see `docs/plans/conbo.md`), before the full
Phase 6 benchmark. Run on 2026-10-10.

**Setup** (the paper's Test 1): 15 simulated Branin users in a fixed order
(`generate_user_variants("branin", 15, shift_range=0.3, scale_range=0.2, seed=0)`),
30 evaluations each, `r0 = 16` random starts decaying by `dr = 5` per user,
`lambda = 50`, otherwise `ConBOConfig` defaults (40 x 40 candidate grid, 1,200
replay epochs, 15 online epochs).

- `bo`: standard BO with the same machinery but no population, so every user is
  "user 1" (16 random starts, then EI on the user GP).
- `conbo_eval`: ConBO with the schedule's `t` counting every evaluation (default).
- `conbo_bo_steps`: ConBO with `t` counting BO steps after the random starts.

Files: `results/<method>.json` (per-user log10 best-so-far regret, 15 x 30),
`logs/`, `plots/conbo_vs_bo_branin.png`, `run_comparison.py`, `plot_comparison.py`
and `run_all.sh` to reproduce everything.

**Mean log10 regret, users 4-15** (lower is better):

| Evaluations | 1 | 3 | 5 | 10 | 15 | 20 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Standard BO | 1.64 | 1.20 | 1.00 | 0.32 | 0.21 | -0.22 | **-1.64** |
| ConBO (`conbo_eval`) | **1.00** | **0.43** | **0.35** | **0.22** | **-0.32** | **-0.55** | -1.29 |
| ConBO (`conbo_bo_steps`) | 0.98 | 0.81 | 0.72 | 0.36 | -0.16 | -0.48 | -1.15 |

**Reading it**
- ConBO is clearly ahead early in a session: after 3-5 evaluations its regret is
  about 0.65-0.8 lower (log10), and it stays ahead up to about 20 evaluations.
  Standard BO spends its first 16 evaluations on random points.
- The end-of-session numbers mostly reflect the 40 x 40 candidate grid. For 9 of
  15 users both methods finish exactly at the best grid point. Standard BO's
  better mean at 30 evaluations comes from random points that land between grid
  points (user 9 reached -4.38 this way; the grid's best is also -4.38 there,
  while ConBO stopped at -1.09).
- ConBO ended poorly for users 11 and 14 (about -0.1). All three Branin basins
  are nearly equally good for them, so this is not a wrong-basin effect.
- The two `t` options are identical for users 2-3, and from user 5 onward there
  are no random starts, so they are the same rule. Their differences from user 4
  on come from runs diverging, which shows how large single-seed variation is.

**Caveats:** one seed only; no error bars across seeds. The full Phase 6 benchmark
should use several seeds and polish or refine the final grid suggestion so the
end-of-session comparison measures the methods rather than the grid.
