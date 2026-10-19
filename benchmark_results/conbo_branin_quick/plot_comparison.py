"""Plot the quick ConBO vs standard BO comparison stored next to this script."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
labels = {
    "bo": "Standard BO (16 random + EI)",
    "conbo_eval": "ConBO (t counts evaluations)",
    "conbo_bo_steps": "ConBO (t counts BO steps)",
}
results = {m: np.array(json.loads((root / "results" / f"{m}.json").read_text())["log_regret"]) for m in labels}
evals = np.arange(1, 31)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for method, regret in results.items():
    later = regret[3:]  # users 4-15: at most one random start under ConBO
    mean, std = later.mean(axis=0), later.std(axis=0)
    axes[0].plot(evals, mean, label=labels[method])
    axes[0].fill_between(evals, mean - std, mean + std, alpha=0.15)
axes[0].set_xlabel("Evaluations for the user")
axes[0].set_ylabel("log10 regret (best so far)")
axes[0].set_title("Users 4-15, mean ± std")
axes[0].legend()

users = np.arange(1, 16)
width = 0.27
for k, (method, regret) in enumerate(results.items()):
    axes[1].bar(users + (k - 1) * width, regret[:, 4], width, label=labels[method])
axes[1].set_xlabel("User (order of arrival)")
axes[1].set_ylabel("log10 regret after 5 evaluations")
axes[1].set_title("Early regret per user")
axes[1].set_xticks(users)
fig.suptitle("ConBO vs standard BO on Branin users (paper Test 1 settings, one seed)")
fig.tight_layout()
out = root / "plots" / "conbo_vs_bo_branin.png"
fig.savefig(out, dpi=150)
print(f"saved_plot={out}")
