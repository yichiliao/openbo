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

The paper's ablations (Appendix A.4), all ConBO with one switch changed:
``conbo_no_gp`` (population EI only), ``conbo_no_filter`` (no variance filter),
``bnn_no_replay`` (the BNN keeps its online updates) and ``bnn_direct_replay``
(retrain on all past observations after each user).
``taf_paper`` is the paper's TAF baseline: TAF with decayed source weights
(``d1 = 0``, ``d2`` per test), i.e. one-objective TAF+ with Matern-5/2 GPs.

Every result holds the time of each iteration per user (suggest and observe, without
the retraining after the user) and, for BNN methods, the BNN's predicted mean and
variance on a 50 x 50 grid after the last user (the paper's Figure 5). A test with
``revisit=True`` (Test 3) re-optimizes every user from the final population.
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
from openbo.optimizers.bo_taf_plus import TAFPlusConfig, TAFPlusPopulation, TAFPlusSequentialOptimizer
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

# ConBO and its ablations: one switch of ConBOConfig each.
CONBO_VARIANTS: dict[str, dict[str, object]] = {
    "conbo": {},
    "conbo_no_gp": {"use_user_gp": False},
    "conbo_no_filter": {"replay_filter": False},
    "bnn_no_replay": {"memory": "none"},
    "bnn_direct_replay": {"memory": "direct"},
    # Not in the paper: ConBO whose replayed variance includes the spread between users.
    "conbo_total_var": {"replay_between_users": True},
}
CONBO_METHODS = (*CONBO_VARIANTS, "bo", "taf_m", "taf_r", "taf_paper", "single_gp")
VARIANCE_MAP_POINTS = 50


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
    # Schedule of the mixed EI; None keeps ConBOConfig's defaults (5, 0.2).
    alpha1: float | None = None
    alpha2: float | None = None
    # Source-weight decay of the paper's TAF baseline.
    taf_d1: int = 0
    taf_d2: float = 0.05
    # Test 3: after the sequence, re-optimize every user from the final population.
    revisit: bool = False


CONBO_TESTS: dict[str, ConBOTest] = {
    # Test 1: Branin, 15 users, 30 evaluations, r0 = 16 decaying by 5, lambda = 50.
    # With OpenBO's default schedule (alpha1 = 5, alpha2 = 0.2): the first archived run.
    "branin": ConBOTest("branin", 15, 0.3, 0.2, 30, 16, 5, 50.0),
    # The same with the paper's schedule (App. A.4: alpha1 = 16, alpha2 = 0.1).
    "branin_paper": ConBOTest("branin", 15, 0.3, 0.2, 30, 16, 5, 50.0, 16.0, 0.1, 0, 0.05),
    # Test 2: McCormick, 15 users, 7 evaluations, r0 = 3, dr = 3, lambda = 10 (App. A.6).
    "mccormick_paper": ConBOTest("mccormick", 15, 0.5, 0.2, 7, 3, 3, 10.0, 3.0, 0.15, 0, 0.15),
    # Test 3: Branin, 10 users, ranges 0.4 and 0.4, then the same users again (App. A.8).
    "branin_revisit": ConBOTest(
        "branin", 10, 0.4, 0.4, 30, 16, 5, 50.0, 16.0, 0.1, 0, 0.05, revisit=True
    ),
}


def make_test_users(test: ConBOTest, seed: int) -> tuple[list, list[FunctionSpec]]:
    """Return the variants and specs of a test's users for one seed."""
    variants = generate_user_variants(
        test.base, test.n_users, test.shift_range, test.scale_range, seed=seed
    )
    return variants, make_user_sequence(test.base, variants)


def conbo_config(test: ConBOTest, input_dim: int, seed: int) -> ConBOConfig:
    """Return the ConBO configuration for a test."""
    extra: dict[str, object] = {}
    if test.alpha1 is not None:
        extra["alpha1"] = test.alpha1
    if test.alpha2 is not None:
        extra["alpha2"] = test.alpha2
    return ConBOConfig(
        input_dim=input_dim,
        r0=test.r0,
        dr=test.dr,
        replay_max_variance=test.max_variance,
        seed=seed,
        **extra,  # type: ignore[arg-type]
    )


def _random_starts(user_index: int, test: ConBOTest) -> int:
    """ConBO's decaying random starts for a 1-based user; at least 1 for user 1."""
    n = max(0, test.r0 - (user_index - 1) * test.dr)
    return max(1, n) if user_index == 1 else n


def _run_fresh_bo(
    spec: FunctionSpec, config: ConBOConfig, n_evals: int
) -> tuple[NDArray[np.float64], list[float]]:
    """Run one user as plain BO (no population); return the evaluated x and each step's seconds."""
    opt = ConBOPopulation(config).new_user(spec.name)
    steps: list[float] = []
    for _ in range(n_evals):
        start = time.perf_counter()
        x = opt.suggest()
        opt.observe(x, spec.objective(x))
        steps.append(time.perf_counter() - start)
    return opt.x_obs, steps


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
) -> tuple[NDArray[np.float64], list[float]]:
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
        steps: list[float] = []
        for k in range(n_evals):
            start = time.perf_counter()
            x = rng.random((1, spec.dim)) if k < n_random else taf.suggest()
            taf.observe(x, spec.objective(x))
            steps.append(time.perf_counter() - start)
    return taf.x_obs, steps


def _run_taf_paper_user(
    spec: FunctionSpec,
    population: TAFPlusPopulation,
    test: ConBOTest,
    n_random: int,
    n_evals: int,
    seed: int,
) -> tuple[NDArray[np.float64], list[float]]:
    """Run one user with the paper's TAF: decayed source weights over all earlier users."""
    rng = np.random.default_rng(seed)
    opt = TAFPlusSequentialOptimizer(
        TAFPlusConfig(
            input_dim=spec.dim,
            objective_weights=(1.0,),
            decay=True,
            d1=test.taf_d1,
            d2=test.taf_d2,
            kernel_type="matern52",
            seed=seed,
        ),
        population,
    )
    steps: list[float] = []
    for k in range(n_evals):
        start = time.perf_counter()
        x = rng.random((1, spec.dim)) if k < n_random else opt.suggest()
        opt.observe(x, spec.objective(x)[:, None])
        steps.append(time.perf_counter() - start)
    return opt.x_obs, steps


def _run_single_gp_user(
    spec: FunctionSpec,
    x_prev: NDArray[np.float64],
    y_prev: NDArray[np.float64],
    n_random: int,
    n_evals: int,
    candidates: NDArray[np.float64],
    seed: int,
) -> tuple[NDArray[np.float64], list[float]]:
    """Run one user with a GP on all previous users' data plus this user's.

    Hyperparameters are fitted once, at the user's first BO step; later steps
    keep them and only add data. Refitting on every step costs about 3 hours
    per 15-user sequence, because the pooled data grows to about 450 points.
    """
    rng = np.random.default_rng(seed)
    gp = new_user_gp("matern52", spec.dim, seed)
    x_obs = np.empty((0, spec.dim))
    y_obs = np.empty(0)
    steps: list[float] = []
    for k in range(n_evals):
        start = time.perf_counter()
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
        steps.append(time.perf_counter() - start)
    return x_obs, steps


def _variance_map(population: ConBOPopulation) -> dict[str, object] | None:
    """Return the BNN's mean and variance on a 50 x 50 grid (2D problems only), or None."""
    if population.bnn is None or population.config.input_dim != 2:
        return None
    axis = np.linspace(0.0, 1.0, VARIANCE_MAP_POINTS)
    grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    mean, var = population.bnn.predict(grid)
    return {"points": VARIANCE_MAP_POINTS, "mean": mean.tolist(), "var": var.tolist()}


def _reoptimize_users(
    population: ConBOPopulation, users: list[FunctionSpec], n_evals: int
) -> list[dict[str, object]]:
    """Test 3: re-optimize each user from the final population, which stays unchanged.

    Each user starts as the next one after the whole sequence, so by the random-start
    schedule they begin without random points. Nothing is stored (no ``finish_user``).
    """
    records = []
    for spec in users:
        opt = population.new_user(f"{spec.name}_reopt")
        steps: list[float] = []
        for _ in range(n_evals):
            start = time.perf_counter()
            x = opt.suggest()
            opt.observe(x, spec.objective(x))
            steps.append(time.perf_counter() - start)
        records.append(
            {
                "name": spec.name,
                "optimum": spec.optimum,
                "x": opt.x_obs.tolist(),
                "y": opt.y_obs.tolist(),
                "n_random": min(opt.n_random, n_evals),
                "iteration_seconds": steps,
            }
        )
    return records


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
    if test_name not in CONBO_TESTS:
        raise ValueError(f"Unknown test '{test_name}'. Choose from {sorted(CONBO_TESTS)}.")
    test = CONBO_TESTS[test_name]
    variants, users = make_test_users(test, seed)
    config = replace(conbo_config(test, users[0].dim, seed), **(config_overrides or {}))
    xs: list[NDArray[np.float64]] = []
    seconds: list[float] = []
    iteration_seconds: list[list[float]] = []
    n_random: list[int] = []
    extra: dict[str, object] = {}

    if method in CONBO_VARIANTS:
        variant_config = replace(config, **CONBO_VARIANTS[method])  # type: ignore[arg-type]
        result = run_conbo_sequence(
            [u.objective for u in users], variant_config, test.n_evals, names=[u.name for u in users]
        )
        xs = [u.x_obs for u in result.users]
        seconds = [u.seconds for u in result.users]
        iteration_seconds = [u.iteration_seconds for u in result.users]
        n_random = [u.n_random for u in result.users]
        assert result.population is not None
        extra["variance_map"] = _variance_map(result.population)
        extra["replay_seconds"] = [u.replay.seconds for u in result.users]
        if test.revisit:
            extra["reoptimization"] = _reoptimize_users(result.population, users, test.n_evals)
    elif test.revisit:
        raise ValueError("A revisit test (Test 3) compares only the ConBO variants.")
    else:
        source_dir = Path(work_dir) / f"{method}_{test_name}_s{seed}"
        candidates = replay_inputs(config.input_dim, config.candidate_grid_points, 0, seed=seed)
        x_prev = np.empty((0, config.input_dim))
        y_prev = np.empty(0)
        paper_population = TAFPlusPopulation(config.input_dim, 1, "matern52")
        for i, spec in enumerate(users):
            start = time.perf_counter()
            user_seed = seed + 1009 * (i + 1)
            if method == "bo" or (method.startswith("taf") and i == 0):
                x, steps = _run_fresh_bo(spec, replace(config, seed=user_seed), test.n_evals)
                count = max(1, test.r0)
            elif method == "taf_paper":
                count = _random_starts(i + 1, test)
                x, steps = _run_taf_paper_user(
                    spec, paper_population, test, count, test.n_evals, user_seed
                )
            elif method.startswith("taf"):
                count = _random_starts(i + 1, test)
                x, steps = _run_taf_user(spec, source_dir, method, count, test.n_evals, user_seed)
            else:
                count = _random_starts(i + 1, test)
                x, steps = _run_single_gp_user(
                    spec, x_prev, y_prev, count, test.n_evals, candidates, user_seed
                )
            y = spec.objective(x)
            if method == "taf_paper":
                paper_population.add_user(spec.name, x, y[:, None], seed=user_seed)
            elif method.startswith("taf"):
                _store_source(source_dir, spec.name, x, y, user_seed)
            x_prev = np.vstack([x_prev, x])
            y_prev = np.concatenate([y_prev, y])
            xs.append(x)
            seconds.append(time.perf_counter() - start)
            iteration_seconds.append(steps)
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
                "iteration_seconds": steps,
            }
            for spec, variant, x, count, sec, steps in zip(
                users, variants, xs, n_random, seconds, iteration_seconds
            )
        ],
        **extra,
    }
