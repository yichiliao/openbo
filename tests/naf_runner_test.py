"""Tests for the NAF optimizer and the MetaBO benchmark runner (NAF, Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest

from openbo.benchmarks.naf_runner import NAF_METHODS, run_naf_method, run_naf_task, naf_test_tasks
from openbo.optimizers.naf import (
    BRANIN_GP_HYPERPARAMETERS,
    NAFConfig,
    NAFSequentialOptimizer,
    run_naf,
    save_naf,
)

TASKS = naf_test_tasks("branin", n_tasks=2, seed=100)


@pytest.fixture(scope="module")
def af_dir(tmp_path_factory) -> str:
    """An untrained 2D AF with a small candidate set, saved like a training run."""
    config = NAFConfig(dim=2, budget=3, hidden=(16, 16), points_per_dim=11, n_local=20)
    directory = tmp_path_factory.mktemp("naf")
    save_naf(directory, config.new_af(), config, BRANIN_GP_HYPERPARAMETERS, name="best_af")
    return str(directory)


@pytest.mark.parametrize("method", NAF_METHODS)
def test_every_method_runs_and_reports_simple_regret(method: str, af_dir: str) -> None:
    record = run_naf_task(method, TASKS[0], 0, af_dir=af_dir, budget=3)
    x, y, regret = np.asarray(record["x"]), np.asarray(record["y"]), np.asarray(record["regret"])
    assert x.shape == (3, 2) and y.shape == (3,) and regret.shape == (3,)
    np.testing.assert_allclose(y, TASKS[0](x))
    np.testing.assert_allclose(regret, TASKS[0].optimum - np.maximum.accumulate(y))
    assert np.all(regret >= -1e-12) and np.all(np.diff(regret) <= 1e-12)
    assert np.all((x >= 0.0) & (x <= 1.0))


def test_taf_starts_near_the_family_optimum() -> None:
    """With no data, TAF's first point maximizes the 50 sources' mean, close to every task's optimum."""
    for method in ("taf_me_50", "taf_r_50"):
        record = run_naf_task(method, TASKS[1], 1, budget=1)
        assert record["regret"][0] < 0.2


def test_naf_budget_must_match_training(af_dir: str) -> None:
    with pytest.raises(ValueError, match="trained with budget 3"):
        run_naf_task("naf", TASKS[0], 0, af_dir=af_dir, budget=5)
    with pytest.raises(ValueError, match="af_dir"):
        run_naf_task("naf", TASKS[0], 0, budget=3)


def test_optimizer_is_ask_tell_and_stops_at_its_budget(af_dir: str) -> None:
    optimizer = NAFSequentialOptimizer.from_directory(af_dir)
    x, y = run_naf(TASKS[0], optimizer)
    assert x.shape == (3, 2) and optimizer.t == 3
    np.testing.assert_allclose(y, TASKS[0](x))
    with pytest.raises(RuntimeError, match="budget"):
        optimizer.suggest()
    with pytest.raises(ValueError, match="shape"):
        optimizer.observe(np.zeros((2, 2)), np.zeros(3))


def test_workers_give_the_same_results() -> None:
    serial = run_naf_method("ei", n_tasks=2, budget=2, n_workers=1)
    parallel = run_naf_method("ei", n_tasks=2, budget=2, n_workers=2)
    assert serial["tasks"] == parallel["tasks"]
    assert [t["task_index"] for t in serial["tasks"]] == [0, 1]
