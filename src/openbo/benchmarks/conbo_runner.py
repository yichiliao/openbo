"""Method dispatcher for ConBO benchmarks over sequences of simulated users.

Each method runs the same ordered users (Liao et al., CHI 2025, Appendix A) and
returns one trajectory per user. Kept separate from ``runner.py`` because a run
is a whole user sequence, not one task.

Methods:
- ``conbo``: :func:`run_conbo_sequence` with the test's settings.
- ``bo``: standard BO; every user starts fresh with ``r0`` random points, then EI
  on the user GP (the same machinery as ConBO's first user).
- ``taf_m`` / ``taf_r``: TAF with all previous users as sources. User 1 has no
  sources and runs as ``bo``. Random starts follow the same decaying schedule
  as ConBO.
- ``single_gp``: one GP over all previous users' data plus the current user's,
  with EI against the current user's best. Same random-start schedule.
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from dataclasses import dataclass, replace
from typing import Any
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from openbo.acquisition.ei import expected_improvement_maximization
from openbo.models.conbo_replay import replay_inputs
from openbo.optimizers.bo_taf import TAFConfig, TAFSequentialOptimizer
from openbo.optimizers.conbo import (
    ConBOConfig,
    ConBOPopulation,
    argmax_on_grid,
    gp_state_dict,
    new_user_gp,
    run_conbo_sequence,
)
from openbo.test_functions.families import generate_user_variants, make_user_sequence
from openbo.test_functions.registry import FunctionSpec

CONBO_METHODS = ("conbo", "bo", "taf_m", "taf_r", "single_gp")


@dataclass(frozen=True)
class ConBOTest:
    """One simulated-user setting from the paper's appendix."""

    base: str
    n_users: int
    shift_range: float
    scale_range: float
    n_evals: int
    r0: int
    dr: int
    max_variance: float


CONBO_TESTS: dict[str, ConBOTest] = {
    # Test 1: Branin, 15 users, 30 evaluations, r0 = 16 decaying by 5, lambda = 50.
    "branin": ConBOTest("branin", 15, 0.3, 0.2, 30, 16, 5, 50.0),
}


def make_test_users(test: ConBOTest, seed: int) -> tuple[list, list[FunctionSpec]]:
    """Return the variants and specs of a test's users for one seed."""
    variants = generate_user_variants(
        test.base, test.n_users, test.shift_range, test.scale_range, seed=seed
    )
    return variants, make_user_sequence(test.base, variants)


def conbo_config(test: ConBOTest, input_dim: int, seed: int) -> ConBOConfig:
    """Return the ConBO configuration for a test."""
    return ConBOConfig(
        input_dim=input_dim,
        r0=test.r0,
        dr=test.dr,
        replay_max_variance=test.max_variance,
        seed=seed,
    )


def _random_starts(user_index: int, test: ConBOTest) -> int:
    """ConBO's decaying random starts for a 1-based user; at least 1 for user 1."""
    n = max(0, test.r0 - (user_index - 1) * test.dr)
    return max(1, n) if user_index == 1 else n


def _run_fresh_bo(spec: FunctionSpec, config: ConBOConfig, n_evals: int) -> NDArray[np.float64]:
    """Run one user as plain BO (no population) and return the evaluated x."""
    opt = ConBOPopulation(config).new_user(spec.name)
    for _ in range(n_evals):
        x = opt.suggest()
        opt.observe(x, spec.objective(x))
    return opt.x_obs


def _store_source(
    root: Path, name: str, x: NDArray[np.float64], y: NDArray[np.float64], seed: int
) -> None:
    """Fit a user GP and save it in the TAF source layout."""
    gp = new_user_gp("matern52", x.shape[1], seed)
    gp.fit(x, y)
    (root / "gp_states").mkdir(parents=True, exist_ok=True)
    (root / "trajectories").mkdir(parents=True, exist_ok=True)
    (root / "gp_states" / f"{name}.json").write_text(json.dumps({"gp_state": gp_state_dict(gp)}))
    (root / "trajectories" / f"{name}.json").write_text(
        json.dumps(
            {
                "x_values": x.tolist(),
                "y_values": y.tolist(),
                "best_so_far": np.maximum.accumulate(y).tolist(),
            }
        )
    )


def _run_taf_user(
    spec: FunctionSpec, source_dir: Path, mode: str, n_random: int, n_evals: int, seed: int
) -> NDArray[np.float64]:
    """Run one user with TAF over the stored sources; return the evaluated x."""
    rng = np.random.default_rng(seed)
    with contextlib.redirect_stdout(io.StringIO()):  # TAF prints debug lines per step
        taf = TAFSequentialOptimizer(
            TAFConfig(
                bounds=[(0.0, 1.0)] * spec.dim,
                taf_run_dir=source_dir,
                n_init=0,
                n_iter=n_evals + 1,
                taf_weight_mode=mode,
                track_acquisition=False,
                seed=seed,
            )
        )
        for k in range(n_evals):
            x = rng.random((1, spec.dim)) if k < n_random else taf.suggest()
            taf.observe(x, spec.objective(x))
    return taf.x_obs


def _run_single_gp_user(
    spec: FunctionSpec,
    x_prev: NDArray[np.float64],
    y_prev: NDArray[np.float64],
    n_random: int,
    n_evals: int,
    candidates: NDArray[np.float64],
    seed: int,
) -> NDArray[np.float64]:
    """Run one user with a GP on all previous users' data plus this user's.

    Hyperparameters are fitted once, at the user's first BO step; later steps
    keep them and only add data. Refitting on every step costs about 3 hours
    per 15-user sequence, because the pooled data grows to about 450 points.
    """
    rng = np.random.default_rng(seed)
    gp = new_user_gp("matern52", spec.dim, seed)
    x_obs = np.empty((0, spec.dim))
    y_obs = np.empty(0)
    for k in range(n_evals):
        if k < n_random:
            x = rng.random((1, spec.dim))
        else:
            gp.fit(np.vstack([x_prev, x_obs]), np.concatenate([y_prev, y_obs]))
            gp.optimize_hyperparameters = False
            best = float(np.max(y_obs)) if y_obs.size else float(np.max(y_prev))

            def acquisition(z: NDArray[np.float64], best: float = best) -> NDArray[np.float64]:
                return expected_improvement_maximization(*gp.posterior(z), best)

            x = argmax_on_grid(acquisition, candidates, refine=True)
        x_obs = np.vstack([x_obs, x])
        y_obs = np.concatenate([y_obs, spec.objective(x)])
    return x_obs


def run_conbo_method(
    method: str,
    test_name: str,
    seed: int,
    work_dir: str | Path,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Run one method on one test for one seed; return a JSON-friendly result.

    `work_dir` holds per-run scratch files (TAF sources). `config_overrides`
    replaces `ConBOConfig` fields (used by tests to shrink training budgets).
    """
    if method not in CONBO_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {CONBO_METHODS}.")
    test = CONBO_TESTS[test_name]
    variants, users = make_test_users(test, seed)
    config = replace(conbo_config(test, users[0].dim, seed), **(config_overrides or {}))
    xs: list[NDArray[np.float64]] = []
    seconds: list[float] = []
    n_random: list[int] = []

    if method == "conbo":
        result = run_conbo_sequence(
            [u.objective for u in users], config, test.n_evals, names=[u.name for u in users]
        )
        xs = [u.x_obs for u in result.users]
        seconds = [u.seconds for u in result.users]
        n_random = [u.n_random for u in result.users]
    else:
        source_dir = Path(work_dir) / f"{method}_{test_name}_s{seed}"
        candidates = replay_inputs(config.input_dim, config.candidate_grid_points, 0, seed=seed)
        x_prev = np.empty((0, config.input_dim))
        y_prev = np.empty(0)
        for i, spec in enumerate(users):
            start = time.perf_counter()
            user_seed = seed + 1009 * (i + 1)
            if method == "bo" or (method.startswith("taf") and i == 0):
                x = _run_fresh_bo(spec, replace(config, seed=user_seed), test.n_evals)
                count = max(1, test.r0)
            elif method.startswith("taf"):
                count = _random_starts(i + 1, test)
                x = _run_taf_user(spec, source_dir, method, count, test.n_evals, user_seed)
            else:
                count = _random_starts(i + 1, test)
                x = _run_single_gp_user(
                    spec, x_prev, y_prev, count, test.n_evals, candidates, user_seed
                )
            y = spec.objective(x)
            if method.startswith("taf"):
                _store_source(source_dir, spec.name, x, y, user_seed)
            x_prev = np.vstack([x_prev, x])
            y_prev = np.concatenate([y_prev, y])
            xs.append(x)
            seconds.append(time.perf_counter() - start)
            n_random.append(min(count, test.n_evals))

    return {
        "method": method,
        "test": test_name,
        "seed": seed,
        "users": [
            {
                "name": spec.name,
                "variant": variant.to_dict(),
                "optimum": spec.optimum,
                "x": x.tolist(),
                "y": spec.objective(x).tolist(),
                "n_random": count,
                "seconds": sec,
            }
            for spec, variant, x, count, sec in zip(users, variants, xs, n_random, seconds)
        ],
    }
