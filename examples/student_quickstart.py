"""Compare random search and BO with the same small evaluation budget.

Run from the repository root: uv run python examples/student_quickstart.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openbo.optimizers.bo_scratch import run_bo_scratch
from openbo.test_functions.registry import get_function_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-evals", type=int, default=12, help="Total evaluations per method (>= 3).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("test_results/student_quickstart"))
    args = parser.parse_args()
    if args.n_evals < 3:
        parser.error("--n-evals must be at least 3 (the initial design).")

    spec = get_function_spec("branin")
    rng = np.random.default_rng(args.seed)
    random_x = rng.random((args.n_evals, spec.dim))
    random_y = spec.objective(random_x)
    bo = run_bo_scratch(
        spec.objective, bounds=spec.bounds, n_init=3, n_iter=args.n_evals - 3,
        n_candidates=128, n_starts=2, seed=args.seed,
    )
    runs = {"random": (random_x, random_y), "bo_scratch": (bo.x_obs, bo.y_obs)}
    fig, ax = plt.subplots(figsize=(7, 4))
    records = {}
    for name, (x, y) in runs.items():
        best = np.maximum.accumulate(y)
        ax.plot(np.arange(1, args.n_evals + 1), best, marker="o", label=name)
        records[name] = {"x_values": x.tolist(), "y_values": y.tolist(), "best_so_far": best.tolist()}
        print(f"{name}: {args.n_evals} evaluations, best y={best[-1]:.4f}")
    ax.axhline(spec.optimum, color="black", linestyle="--", label="Known optimum")
    ax.set(xlabel="Objective evaluations (including the 3 initial points)", ylabel="Best objective value so far")
    ax.legend()
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "comparison.png", dpi=150)
    plt.close(fig)
    payload = {"function": spec.name, "seed": args.seed, "n_evals": args.n_evals, "runs": records}
    (args.output_dir / "trajectories.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results: {args.output_dir.resolve()}")
    print("A short run can favor either method. Compare several seeds before drawing conclusions.")


if __name__ == "__main__":
    main()
