"""Run the TAF+ simulations (Liao et al., CHI 2024, App. B) on the sphere users.

Steps, each with its own `--method`:
- `population`: build and cache the simulation's population models (MOBO per population user).
- `decay_search`: leave-one-out grid search for `(d1, d2)` on the population (workflow step 3).
- `taf_plus`, `taf`, `bo`, `random`: run one method on the 10 new users.

The population is built on first use and cached under
`<results-dir>/populations/<test-id>_<sim>_s<seed>/`, so every method sees the
same population. `taf_plus` takes `--d1/--d2`, or else the result of an earlier
`decay_search` with the same test id, simulation, weights and seed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from openbo.benchmarks.taf_plus_runner import (
    SIM2_OBJECTIVE_WEIGHTS,
    TAF_PLUS_METHODS,
    TAF_PLUS_SIMULATIONS,
    SimulationSettings,
    build_population,
    run_taf_plus_method,
)
from openbo.optimizers.bo_taf_plus import TAFPlusConfig
from openbo.optimizers.taf_plus_workflow import search_decay
from openbo.test_functions.taf_plus_users import SPHERE_INPUT_DIM


def alpha_tag(alpha: tuple[float, ...]) -> str:
    """Return a filename tag such as `a030-050-020`."""
    return "a" + "-".join(f"{round(a * 100):03d}" for a in alpha)


def main() -> None:
    """Parse arguments and run the requested step."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--method", choices=("population", "decay_search", *TAF_PLUS_METHODS), required=True, help="Step to run."
    )
    parser.add_argument("--sim", choices=sorted(TAF_PLUS_SIMULATIONS), required=True, help="Similarity setting.")
    parser.add_argument(
        "--alpha", type=float, nargs=3, default=list(SIM2_OBJECTIVE_WEIGHTS), help="Objective weights (sum to 1)."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for users, noise and optimizers.")
    parser.add_argument("--d1", type=int, default=None, help="Decay start (taf_plus).")
    parser.add_argument("--d2", type=float, default=None, help="Decay rate (taf_plus).")
    parser.add_argument("--workers", type=int, default=1, help="Processes for population modeling and decay search.")
    parser.add_argument("--n-evals", type=int, default=20, help="Evaluations per new user.")
    parser.add_argument("--results-dir", default="test_results/taf_plus", help="Output root folder.")
    parser.add_argument("--test-id", default="default", help="Experiment id used in filenames.")
    args = parser.parse_args()

    alpha = tuple(float(a) for a in args.alpha)
    settings = SimulationSettings(n_evals=args.n_evals)
    root = Path(args.results_dir)
    tag = f"{args.sim}_{alpha_tag(alpha)}_s{args.seed}"
    decay_path = root / "decay" / f"{args.test_id}_{tag}.json"
    start = time.perf_counter()

    population = build_population(
        args.sim,
        args.seed,
        settings,
        root / "populations" / f"{args.test_id}_{args.sim}_s{args.seed}",
        n_workers=args.workers,
    )
    if args.method == "population":
        print(f"sim={args.sim} seed={args.seed} users={population.n_users} seconds={time.perf_counter() - start:.1f}")
        return

    if args.method == "decay_search":
        config = TAFPlusConfig(
            input_dim=SPHERE_INPUT_DIM,
            objective_weights=alpha,
            n_candidates=settings.n_candidates,
            refine_suggestion=True,
            seed=args.seed,
        )
        result = search_decay(population, config, n_iter=10, n_workers=args.workers)
        decay_path.parent.mkdir(parents=True, exist_ok=True)
        decay_path.write_text(
            json.dumps(
                {
                    "sim": args.sim,
                    "objective_weights": list(alpha),
                    "seed": args.seed,
                    "best": list(result.best),
                    "d1_values": list(result.d1_values),
                    "d2_values": list(result.d2_values),
                    "scores": result.scores.tolist(),
                    "no_decay_score": float(result.no_decay_curves.mean()),
                    "curves": result.curves.tolist(),
                    "no_decay_curves": result.no_decay_curves.tolist(),
                }
            ),
            encoding="utf-8",
        )
        print(
            f"sim={args.sim} best=(d1={result.best[0]}, d2={result.best[1]}) "
            f"seconds={time.perf_counter() - start:.1f} saved={decay_path}"
        )
        return

    d1, d2 = args.d1, args.d2
    if args.method == "taf_plus" and (d1 is None or d2 is None):
        if not decay_path.exists():
            parser.error(f"taf_plus needs --d1 and --d2, or a decay search result at {decay_path}.")
        best = json.loads(decay_path.read_text(encoding="utf-8"))["best"]
        d1, d2 = int(best[0]), float(best[1])
    result = run_taf_plus_method(
        args.method, args.sim, alpha, args.seed, population, settings, d1=d1 or 0, d2=d2 or 1.0
    )
    out = root / "trajectories" / f"{args.test_id}_{args.method}_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"method={args.method} {tag} seconds={time.perf_counter() - start:.1f} saved={out}")


if __name__ == "__main__":
    main()
