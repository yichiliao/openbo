"""Regression tests for benchmark-runner budget resolution and TAF routing."""

from __future__ import annotations

import json

import numpy as np
import pytest

from openbo.benchmarks.runner import run_simple_benchmark
from openbo.models.gp_scratch import GPScratch


def test_bo_budget_rejects_nonpositive_n_iter() -> None:
    """--n-iter 0 must raise, not silently degrade BO to random search."""
    with pytest.raises(ValueError, match="n_iter must be positive"):
        run_simple_benchmark("branin", n_evals=20, method="bo_scratch", n_iter=0)


def test_bo_budget_rejects_nonpositive_n_init() -> None:
    with pytest.raises(ValueError, match="n_init must be positive"):
        run_simple_benchmark("branin", n_evals=20, method="bo_scratch", n_init=0)


def test_bo_budget_rejects_single_eval() -> None:
    """A BO method with n_evals==1 fails clearly instead of overrunning to 2 evals."""
    with pytest.raises(ValueError, match="n_evals >= 2"):
        run_simple_benchmark("branin", n_evals=1, method="bo_scratch")


def test_bo_budget_uses_exactly_n_evals() -> None:
    result = run_simple_benchmark("branin", n_evals=6, method="bo_scratch", seed=0)
    assert len(result.y_values) == 6


def test_bo_budget_rejects_overrunning_n_init() -> None:
    """An n_init that leaves no room for a BO step must raise, not silently
    overrun the budget (which would make method comparisons unfair)."""
    with pytest.raises(ValueError, match="n_init must be < n_evals"):
        run_simple_benchmark("branin", n_evals=10, method="bo_scratch", n_init=20)


def test_bo_budget_rejects_overrunning_n_iter() -> None:
    with pytest.raises(ValueError, match="n_iter must be < n_evals"):
        run_simple_benchmark("branin", n_evals=10, method="bo_scratch", n_iter=15)


@pytest.mark.parametrize("kwargs", [{"n_init": 3}, {"n_iter": 4}, {}])
def test_bo_budget_derived_split_totals_n_evals(kwargs) -> None:
    """Whenever the split is derived from n_evals, the totals match exactly."""
    result = run_simple_benchmark("branin", n_evals=10, method="bo_scratch", seed=0, **kwargs)
    assert len(result.y_values) == 10


def _make_taf_run_dir(tmp_path):
    run_dir = tmp_path / "taf_run"
    (run_dir / "gp_states").mkdir(parents=True)
    (run_dir / "trajectories").mkdir(parents=True)
    x = np.array([[0.1, 0.1], [0.6, 0.4], [0.9, 0.8]], dtype=np.float64)
    y = np.array([-1.0, 0.3, 0.2], dtype=np.float64)
    gp = GPScratch(optimize_hyperparameters=False)
    gp.fit(x, y)
    ls = np.asarray(gp.lengthscale, dtype=np.float64).reshape(-1).tolist()
    (run_dir / "trajectories" / "train_task_000.json").write_text(
        json.dumps({"x_values": x.tolist(), "y_values": y.tolist()}), encoding="utf-8"
    )
    (run_dir / "gp_states" / "train_task_000.json").write_text(
        json.dumps({"gp_state": {
            "kernel_type": gp.kernel_type, "lengthscale": ls,
            "variance": float(gp.variance), "noise": float(gp.noise),
        }}), encoding="utf-8",
    )
    return run_dir


def test_taf_default_uses_n_evals(tmp_path) -> None:
    run_dir = _make_taf_run_dir(tmp_path)
    result = run_simple_benchmark(
        "branin", n_evals=4, method="bo_taf", taf_run_dir=str(run_dir), seed=0
    )
    assert len(result.y_values) == 4


def test_taf_honors_user_n_init(tmp_path) -> None:
    """A user-provided n_init is respected (not silently ignored) and totals n_evals.

    The total (5) alone does not discriminate the fix: the pre-fix code ignored
    n_init (0 init + 5 BO) and also totalled 5. The BO-step count does: honoring
    n_init=2 leaves n_evals-n_init=3 BO steps (one acquisition-trace entry each),
    versus 5 when n_init is ignored.
    """
    run_dir = _make_taf_run_dir(tmp_path)
    result = run_simple_benchmark(
        "branin", n_evals=5, method="bo_taf", taf_run_dir=str(run_dir), n_init=2, seed=0
    )
    assert len(result.y_values) == 5
    assert result.metadata is not None
    assert len(result.metadata["taf_acquisition_trace"]) == 3  # 3 BO steps, not 5


def test_taf_rejects_nonpositive_n_iter(tmp_path) -> None:
    """The TAF path validates n_iter (it does not go through resolve_bo_budget)."""
    run_dir = _make_taf_run_dir(tmp_path)
    with pytest.raises(ValueError, match="n_iter must be positive"):
        run_simple_benchmark(
            "branin", n_evals=5, method="bo_taf", taf_run_dir=str(run_dir), n_iter=0, seed=0
        )


def test_bo_budget_explicit_pair_allows_small_n_evals() -> None:
    """An explicit n_init/n_iter pair is honored regardless of n_evals (the
    n_evals>=2 guard only applies when the split is derived from n_evals)."""
    result = run_simple_benchmark(
        "branin", n_evals=1, method="bo_scratch", n_init=2, n_iter=1, seed=0
    )
    assert len(result.y_values) == 3  # 2 init + 1 BO step


def test_taf_n_init_ge_n_evals_raises(tmp_path) -> None:
    """A TAF n_init that leaves no room for a BO step is rejected (no overrun)."""
    run_dir = _make_taf_run_dir(tmp_path)
    with pytest.raises(ValueError, match="n_init must be < n_evals"):
        run_simple_benchmark(
            "branin", n_evals=3, method="bo_taf", taf_run_dir=str(run_dir), n_init=3, seed=0
        )
