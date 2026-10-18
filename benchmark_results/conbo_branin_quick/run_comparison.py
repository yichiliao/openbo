"""Quick look: ConBO vs standard BO on the paper's Branin Test 1 (one seed)."""
import json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from openbo.optimizers.conbo import ConBOConfig, ConBOPopulation, run_conbo_sequence
from openbo.test_functions.families import generate_user_variants, make_user_sequence

method, seed, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
users = make_user_sequence("branin", generate_user_variants("branin", 15, 0.3, 0.2, seed=seed))
cfg = ConBOConfig(input_dim=2, r0=16, dr=5, replay_max_variance=50.0, seed=seed)
N = 30
t0 = time.time()
rows = []
if method == "bo":
    for i, u in enumerate(users):
        opt = ConBOPopulation(replace(cfg, seed=seed + 7919 * i)).new_user(u.name)  # fresh: plain BO, 16 random
        for _ in range(N):
            x = opt.suggest(); opt.observe(x, u.objective(x))
        rows.append(opt.y_obs)
else:
    c = replace(cfg, weight_clock="evaluations" if method == "conbo_eval" else "bo_steps")
    r = run_conbo_sequence([u.objective for u in users], c, N, names=[u.name for u in users])
    rows = [u.y_obs for u in r.users]
regret = [np.log10(np.maximum(u.optimum - np.maximum.accumulate(y), 1e-12)).tolist() for u, y in zip(users, rows)]
out.write_text(json.dumps({"method": method, "seed": seed, "seconds": time.time() - t0, "log_regret": regret}))
print(method, seed, "done", round(time.time() - t0))
