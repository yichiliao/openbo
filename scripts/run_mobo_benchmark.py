"""Run one multi-objective BO method on one problem over several seeds."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from openbo.benchmarks.mobo_runner import MO_PROBLEMS, MOBO_METHODS, get_mo_problem, run_mobo_method


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a multi-objective BO benchmark.")
    parser.add_argument("--method", choices=MOBO_METHODS, required=True, help="Method to run.")
    parser.add_argument("--problem", choices=MO_PROBLEMS, default="oka2", help="Test problem.")
    parser.add_argument("--rho", type=float, default=0.0, help="Output correlation (slf_gp only).")
    parser.add_argument(
        "--problem-seed", type=int, default=0, help="Seed of the SLF-GP draw (slf_gp only)."
    )
    parser.add_argument("--n-init", type=int, default=5, help="Random initial points.")
    parser.add_argument("--n-iter", type=int, default=100, help="BO iterations after init.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Run seeds.")
    parser.add_argument(
        "--n-candidates", type=int, default=None, help="Sobol pool size (NumPy MOBO only)."
    )
    parser.add_argument(
        "--n-rescore",
        type=int,
        default=None,
        help="Score the pool with IEIPV and rescore the top N with CEIPV (CEIPV only).",
    )
    parser.add_argument("--results-dir", default="test_results/mobo", help="Output root folder.")
    parser.add_argument("--test-id", default="default", help="Experiment id used in filenames.")
    return parser.parse_args()


def problem_label(problem: str, rho: float, problem_seed: int) -> str:
    """Return the problem name used in output filenames."""
    if problem == "slf_gp":
        return f"slf_gp_rho{rho:+.2f}_p{problem_seed}"
    return problem


def main() -> None:
    """Run every seed and save one JSON file with all trajectories."""
    args = parse_args()
    spec = get_mo_problem(args.problem, rho=args.rho, problem_seed=args.problem_seed)
    options: dict[str, object] = {}
    if args.method.startswith("mobo_") and args.method != "mobo_botorch":
        if args.n_candidates is not None:
            options["n_candidates"] = args.n_candidates
        if args.n_rescore is not None and "ceipv" in args.method:
            options["n_rescore"] = args.n_rescore

    label = problem_label(args.problem, args.rho, args.problem_seed)
    runs = []
    for seed in args.seeds:
        start = time.perf_counter()
        result = run_mobo_method(
            args.method, spec, n_init=args.n_init, n_iter=args.n_iter, seed=seed, **options
        )
        elapsed = time.perf_counter() - start
        print(
            f"method={args.method} problem={label} seed={seed} "
            f"hypervolume={result.hv_history[-1]:.6g} pareto_size={result.pareto_y.shape[0]} "
            f"seconds={elapsed:.1f}",
            flush=True,
        )
        runs.append(
            {
                "seed": seed,
                "seconds": elapsed,
                "hv_history": result.hv_history.tolist(),
                "x_values": result.x_obs.tolist(),
                "y_values": result.y_obs.tolist(),
                "final_model_state": result.final_model_state,
            }
        )

    payload = {
        "test_id": args.test_id,
        "method": args.method,
        "problem": label,
        "n_objectives": spec.n_objectives,
        "dim": spec.dim,
        "n_init": args.n_init,
        "n_iter": args.n_iter,
        "v_ref": list(spec.v_ref),
        "w_ref": list(spec.w_ref),
        "true_hypervolume": spec.true_hypervolume,
        "options": options,
        "mean_final_hypervolume": float(np.mean([r["hv_history"][-1] for r in runs])),
        "runs": runs,
    }
    out_dir = Path(args.results_dir) / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{args.test_id}_{args.method}_{label}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved_trajectory={output_path}")


if __name__ == "__main__":
    main()
