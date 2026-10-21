"""Run fixed-input numerical comparisons against the current Python checkout.

From the repository root: uv run python rust/tools/validate_python_parity.py
Python is a validation dependency only; the Rust library and CLI are native.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from openbo.acquisition.ei import expected_improvement_maximization
from openbo.acquisition.taf import compute_taf_m_weights
from openbo.benchmarks.hypervolume import hypervolume
from openbo.models.gp_scratch import GPScratch
from openbo.test_functions.metabo_families import (
    normalized_branin,
    normalized_goldstein_price,
    normalized_hartmann3,
)
from openbo.test_functions.multi_objective import get_mo_function_spec
from openbo.test_functions.registry import get_function_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the locked Rust release binary first.",
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "test_results/rust_parity.json"
    )
    args = parser.parse_args()
    executable = "openbo.exe" if sys.platform == "win32" else "openbo"
    if args.build:
        subprocess.run(
            [
                "cargo",
                "build",
                "--locked",
                "--release",
                "--manifest-path",
                str(ROOT / "rust/Cargo.toml"),
            ],
            check=True,
        )
    binary = args.binary or ROOT / "rust/target/release" / executable
    if not binary.exists():
        binary = ROOT / "rust/target/debug" / executable
    if not binary.exists():
        parser.error(
            "Build the binary first with cargo build --manifest-path rust/Cargo.toml, or use --build."
        )
    checks = []

    def native(command: list[str], data: dict):
        result = subprocess.run(
            [str(binary), *command],
            input=json.dumps(data, allow_nan=False),
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(result.stdout)

    def compare(name, actual, expected, *, atol=1e-9, rtol=1e-8):
        actual = np.asarray(actual, dtype=float)
        expected = np.asarray(expected, dtype=float)
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=name)
        checks.append(
            {
                "name": name,
                "values": int(actual.size),
                "max_absolute_error": float(np.max(np.abs(actual - expected)))
                if actual.size
                else 0.0,
            }
        )

    rng = np.random.default_rng(20261021)
    for name in [
        "branin",
        "sphere",
        "ackley",
        "rastrigin",
        "rosenbrock",
        "hartmann6",
        "mccormick",
    ]:
        spec = get_function_spec(name)
        x = rng.random((17, spec.dim))
        compare(
            f"objective/{name}",
            native(["evaluate", "--function", name], {"x": x.tolist()}),
            spec.objective(x),
        )
    for name, dim, objective in [
        ("normalized_branin", 2, normalized_branin),
        ("normalized_goldstein_price", 2, normalized_goldstein_price),
        ("normalized_hartmann3", 3, normalized_hartmann3),
    ]:
        x = rng.random((17, dim))
        compare(
            f"objective/{name}",
            native(["evaluate", "--function", name], {"x": x.tolist()}),
            objective(x),
        )
    for name in ["oka2", "vlmop3", "dtlz1a"]:
        spec = get_mo_function_spec(name)
        x = rng.random((17, spec.dim))
        compare(
            f"objective/{name}",
            native(["evaluate", "--function", name, "--multi"], {"x": x.tolist()}),
            spec.objective(x),
        )

    for kernel in ["rbf", "matern52"]:
        for standardize in [False, True]:
            for constant in [False, True]:
                for noise in [0.0, 1e-4]:
                    x = rng.random((7, 2))
                    y = np.full(7, 2.5) if constant else np.sin(5 * x[:, 0]) + x[:, 1]
                    query = np.vstack([rng.random((9, 2)), x[:2]])
                    config = {
                        "kernel_type": kernel,
                        "lengthscale": [0.2, 0.6],
                        "variance": 1.7,
                        "noise": noise,
                        "standardize_targets": standardize,
                        "optimize_hyperparameters": False,
                        "mean_constant": 0.3,
                    }
                    gp = GPScratch(**config)
                    gp.fit(x, y)
                    mean, var = gp.posterior(query)
                    result = native(
                        ["gp"],
                        {
                            "config": config,
                            "x_train": x.tolist(),
                            "y_train": y.tolist(),
                            "x_test": query.tolist(),
                        },
                    )
                    name = f"gp/{kernel}/standardize={standardize}/constant={constant}/noise={noise}"
                    compare(name + "/mean", result["mean"], mean, atol=2e-7)
                    compare(name + "/variance", result["variance"], var, atol=2e-8)
                    cross = gp._kernel(x, query)
                    v = np.linalg.solve(gp.l_chol, cross)
                    covariance = (gp._kernel(query, query) - v.T @ v) * gp.y_std**2
                    compare(
                        name + "/covariance",
                        result["covariance"],
                        covariance,
                        atol=2e-8,
                    )

    means = np.linspace(-10, 10, 41)
    variances = np.geomspace(1e-12, 5.0, len(means))
    compare(
        "acquisition/ei",
        native(
            ["acquisition"],
            {
                "operation": "ei",
                "mean": means.tolist(),
                "variance": variances.tolist(),
                "best": 0.2,
            },
        ),
        expected_improvement_maximization(means, variances, 0.2),
        atol=1e-12,
    )
    for dim in [2, 3, 4]:
        points = rng.uniform(-0.5, 1.5, (12, dim))
        reference = np.full(dim, -1.0)
        compare(
            f"geometry/hypervolume{dim}d",
            native(
                ["acquisition"],
                {
                    "operation": "hypervolume",
                    "y": points.tolist(),
                    "reference": reference.tolist(),
                },
            ),
            hypervolume(points, reference),
            atol=1e-9,
        )
    source_meta = rng.normal(size=(6, 5))
    target_meta = rng.normal(size=5)
    for rho in [0.1, 1.0, 10.0]:
        compare(
            f"transfer/meta/{rho}",
            native(
                ["acquisition"],
                {
                    "operation": "meta",
                    "sources": source_meta.tolist(),
                    "target": target_meta.tolist(),
                    "rho": rho,
                },
            ),
            compute_taf_m_weights(source_meta, target_meta, rho),
            atol=1e-12,
        )

    report = {
        "status": "passed",
        "groups": len(checks),
        "values_compared": sum(c["values"] for c in checks),
        "binary": str(binary),
        "python": sys.version,
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"PASS: {len(checks)} numerical groups; {report['values_compared']} values compared"
    )
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
