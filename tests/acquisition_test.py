"""Tests for scratch acquisition and BO loops."""

from __future__ import annotations

import json
import numpy as np

from openbo.benchmarks.runner import run_simple_benchmark
from openbo.acquisition.ei import expected_improvement_maximization
from openbo.acquisition.taf import (
    SourceTaskSurrogate,
    compute_taf_m_weights,
    compute_taf_r_weights,
    epanechnikov_weight,
    taf_m_acquisition,
)
from openbo.models.gp_scratch import GPScratch
from openbo.optimizers.bo_botorch import (
    BoTorchConfig,
    BoTorchSequentialOptimizer,
    run_bo_botorch,
)
from openbo.optimizers.bo_scratch import (
    ScratchConfig,
    ScratchSequentialOptimizer,
    run_bo_scratch,
)
from openbo.optimizers.bo_taf import run_bo_taf
from openbo.test_functions.registry import get_function_spec


def test_scratch_ei_shape() -> None:
    """EI output shape should match mean/variance shape."""
    mean = np.array([0.1, -0.2, 0.0], dtype=np.float64)
    variance = np.array([0.5, 0.2, 1.0], dtype=np.float64)
    ei = expected_improvement_maximization(mean, variance, best_y=0.3)
    assert ei.shape == (3,)


def test_scratch_bo_runs_small_loop() -> None:
    """Scratch BO should run and produce expected trajectory length."""
    spec = get_function_spec("branin")
    result = run_bo_scratch(spec.objective, spec.bounds, n_init=3, n_iter=2, seed=0)
    assert result.x_obs.shape == (5, 2)
    assert result.y_obs.shape == (5,)
    assert result.best_y_history.shape == (3,)


def test_scratch_bo_runs_small_loop_with_matern52() -> None:
    """Scratch BO should run with Matern-5/2 ARD kernel and hyperparameter learning."""
    spec = get_function_spec("branin")
    result = run_bo_scratch(
        spec.objective,
        spec.bounds,
        n_init=3,
        n_iter=2,
        kernel_type="matern52",
        optimize_hyperparameters=True,
        seed=0,
    )
    assert result.x_obs.shape == (5, 2)
    assert result.y_obs.shape == (5,)
    assert result.best_y_history.shape == (3,)


def test_scratch_sequential_optimizer_ask_tell() -> None:
    """Scratch sequential optimizer should support ask/tell style updates."""
    spec = get_function_spec("branin")
    optimizer = ScratchSequentialOptimizer(
        ScratchConfig(bounds=spec.bounds, n_init=3, search_strategy="multistart", seed=0)
    )
    optimizer.bootstrap(spec.objective)
    assert optimizer.x_obs.shape == (3, 2)
    assert optimizer.y_obs.shape == (3,)

    x_next = optimizer.suggest()
    assert x_next.shape == (1, 2)
    y_next = np.asarray(spec.objective(x_next), dtype=np.float64)
    optimizer.observe(x_next, y_next)

    result = optimizer.result()
    assert result.x_obs.shape == (4, 2)
    assert result.y_obs.shape == (4,)
    assert result.best_y_history.shape == (2,)


def test_botorch_bo_runs_small_loop() -> None:
    """BoTorch BO should run for a tiny setup."""
    spec = get_function_spec("branin")
    result = run_bo_botorch(spec.objective, spec.bounds, n_init=3, n_iter=2, seed=0)
    assert result.x_obs.shape == (5, 2)
    assert result.y_obs.shape == (5,)
    assert result.best_y_history.shape == (3,)


def test_botorch_sequential_optimizer_ask_tell() -> None:
    """BoTorch sequential optimizer should support ask/tell style updates."""
    spec = get_function_spec("branin")
    optimizer = BoTorchSequentialOptimizer(
        BoTorchConfig(bounds=spec.bounds, n_init=3, seed=0)
    )
    optimizer.bootstrap(spec.objective)
    assert optimizer.x_obs.shape == (3, 2)
    assert optimizer.y_obs.shape == (3,)

    x_next = optimizer.suggest()
    assert x_next.shape == (1, 2)
    y_next = np.asarray(spec.objective(x_next), dtype=np.float64)
    optimizer.observe(x_next, y_next)

    result = optimizer.result()
    assert result.x_obs.shape == (4, 2)
    assert result.y_obs.shape == (4,)
    assert result.best_y_history.shape == (2,)


def test_epanechnikov_weight_behavior() -> None:
    """Epanechnikov should be positive inside radius and zero outside."""
    assert np.isclose(epanechnikov_weight(0.0, rho=1.0), 0.75)
    assert epanechnikov_weight(2.0, rho=1.0) == 0.0


def test_compute_taf_m_weights_shape() -> None:
    """TAF-M weights should return one scalar per source task."""
    source = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    target = np.array([0.0, 1.0], dtype=np.float64)
    weights = compute_taf_m_weights(source, target, rho=1.5)
    assert weights.shape == (2,)
    assert weights[0] >= weights[1]


def test_compute_taf_r_weights_shape_and_range() -> None:
    """TAF-R weights should be one scalar per source in [0, 0.75]."""
    x_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
    y_train = np.array([-1.0, 0.2, 0.1], dtype=np.float64)
    gp = GPScratch(optimize_hyperparameters=False)
    gp.fit(x_train, y_train)
    source = SourceTaskSurrogate(
        name="src0",
        gp=gp,
        best_y=float(np.max(y_train)),
        meta_features=np.array([0.0, 0.0], dtype=np.float64),
    )
    weights = compute_taf_r_weights([source], x_train, y_train, rho=1.0)
    assert weights.shape == (1,)
    assert 0.0 <= float(weights[0]) <= 0.75


def test_taf_m_acquisition_single_and_batch() -> None:
    """TAF-M should support both single-point and batched inputs."""
    x_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
    y_train = np.array([-1.0, 0.2, 0.1], dtype=np.float64)

    target_gp = GPScratch(optimize_hyperparameters=False)
    target_gp.fit(x_train, y_train)
    source_gp = GPScratch(optimize_hyperparameters=False)
    source_gp.fit(x_train, y_train)

    source = SourceTaskSurrogate(
        name="src0",
        gp=source_gp,
        best_y=float(np.max(y_train)),
        meta_features=np.array([0.0, 0.0], dtype=np.float64),
    )
    weights = np.array([0.5], dtype=np.float64)

    val_single = taf_m_acquisition(
        x=np.array([0.2, 0.8], dtype=np.float64),
        target_gp=target_gp,
        target_best_y=float(np.max(y_train)),
        source_surrogates=[source],
        source_weights=weights,
    )
    val_batch = taf_m_acquisition(
        x=np.array([[0.2, 0.8], [0.1, 0.9]], dtype=np.float64),
        target_gp=target_gp,
        target_best_y=float(np.max(y_train)),
        source_surrogates=[source],
        source_weights=weights,
    )
    assert isinstance(val_single, float)
    assert isinstance(val_batch, np.ndarray)
    assert val_batch.shape == (2,)


def test_bo_taf_runs_small_loop(tmp_path) -> None:
    """TAF BO should run by loading source GP states from disk."""
    # Build a tiny fake TAF source run.
    run_dir = tmp_path / "taf_run"
    gp_states_dir = run_dir / "gp_states"
    trajectories_dir = run_dir / "trajectories"
    gp_states_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    x = np.array([[0.1, 0.1], [0.6, 0.4], [0.9, 0.8]], dtype=np.float64)
    y = np.array([-1.0, 0.3, 0.2], dtype=np.float64)
    gp = GPScratch(optimize_hyperparameters=False)
    gp.fit(x, y)
    lengthscale = np.asarray(gp.lengthscale, dtype=np.float64).reshape(-1)

    traj_payload = {
        "task_name": "train_task_000",
        "x_values": [[float(v) for v in row] for row in x],
        "y_values": [float(v) for v in y],
    }
    gp_payload = {
        "task_name": "train_task_000",
        "gp_state": {
            "kernel_type": gp.kernel_type,
            "lengthscale": [float(v) for v in lengthscale],
            "variance": float(gp.variance),
            "noise": float(gp.noise),
            "standardize_targets": bool(gp.standardize_targets),
            "optimize_noise": bool(gp.optimize_noise),
        },
    }
    (trajectories_dir / "train_task_000.json").write_text(
        json.dumps(traj_payload), encoding="utf-8"
    )
    (gp_states_dir / "train_task_000.json").write_text(
        json.dumps(gp_payload), encoding="utf-8"
    )

    spec = get_function_spec("branin")
    result = run_bo_taf(
        objective=spec.objective,
        bounds=spec.bounds,
        taf_run_dir=run_dir,
        n_init=0,
        n_iter=2,
        source_meta_features={"train_task_000": np.array([0.1, 0.2, 0.3])},
        target_meta_features=np.array([0.1, 0.2, 0.3]),
        seed=0,
    )
    assert result.x_obs.shape == (2, 2)
    assert result.y_obs.shape == (2,)
    assert result.best_y_history.shape == (2,)


def test_run_simple_benchmark_supports_bo_taf(tmp_path) -> None:
    """Benchmark runner should route bo_taf with source run directory."""
    run_dir = tmp_path / "taf_run"
    gp_states_dir = run_dir / "gp_states"
    trajectories_dir = run_dir / "trajectories"
    gp_states_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    x = np.array([[0.1, 0.1], [0.6, 0.4], [0.9, 0.8]], dtype=np.float64)
    y = np.array([-1.0, 0.3, 0.2], dtype=np.float64)
    gp = GPScratch(optimize_hyperparameters=False)
    gp.fit(x, y)
    lengthscale = np.asarray(gp.lengthscale, dtype=np.float64).reshape(-1)

    (trajectories_dir / "train_task_000.json").write_text(
        json.dumps(
            {
                "task_name": "train_task_000",
                "x_values": [[float(v) for v in row] for row in x],
                "y_values": [float(v) for v in y],
            }
        ),
        encoding="utf-8",
    )
    (gp_states_dir / "train_task_000.json").write_text(
        json.dumps(
            {
                "task_name": "train_task_000",
                "gp_state": {
                    "kernel_type": gp.kernel_type,
                    "lengthscale": [float(v) for v in lengthscale],
                    "variance": float(gp.variance),
                    "noise": float(gp.noise),
                    "standardize_targets": bool(gp.standardize_targets),
                    "optimize_noise": bool(gp.optimize_noise),
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_simple_benchmark(
        function_name="branin",
        n_evals=3,
        method="bo_taf",
        taf_run_dir=str(run_dir),
        seed=0,
    )
    assert len(result.x_values) == 3
    assert len(result.y_values) == 3
    assert result.metadata is not None
    trace = result.metadata.get("taf_acquisition_trace", [])
    assert isinstance(trace, list)
    assert len(trace) > 0
