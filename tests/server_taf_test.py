"""Tests for dedicated websocket-session TAF server logic."""

from __future__ import annotations

import json

import numpy as np

from openbo.models.gp_scratch import GPScratch
from openbo.optimizers.bo_taf import run_bo_taf
from openbo.server_optimizers.bo_taf_server import (
    BOTAFServerRuntimeConfig,
    BOTAFServerSession,
)
from openbo.test_functions.registry import get_function_spec


def _make_minimal_taf_run_dir(tmp_path) -> str:
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
    return str(run_dir)


def test_taf_server_session_runs_to_done(tmp_path) -> None:
    """Session should progress suggest/observe until done."""
    taf_run_dir = _make_minimal_taf_run_dir(tmp_path)
    runtime = BOTAFServerRuntimeConfig(
        input_dim=2,
        y_min=-100.0,
        y_max=100.0,
        taf_run_dir=taf_run_dir,
    )
    session = BOTAFServerSession.from_start_message(
        {
            "type": "start",
            "n_init": 0,
            "n_iter": 2,
            "seed": 0,
        },
        runtime_config=runtime,
    )

    msg = session.handle({"type": "suggest"})
    while msg["type"] != "done":
        x = np.asarray(msg["x"], dtype=np.float64)
        y = float(-(np.sum((x - 0.2) ** 2)))
        msg = session.handle({"type": "observe", "x": msg["x"], "y": y})

    assert msg["type"] == "done"
    assert msg["optimizer"] == "bo_taf"


def test_taf_server_session_matches_run_bo_taf_with_n_init(tmp_path) -> None:
    """Batched init observe should match run_bo_taf (same iter_count / trajectory)."""
    taf_run_dir = _make_minimal_taf_run_dir(tmp_path)
    spec = get_function_spec("branin")
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    n_init, n_iter, seed = 2, 3, 42

    def objective(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.asarray(spec.objective(x), dtype=np.float64).ravel()

    direct = run_bo_taf(
        objective=objective,
        bounds=bounds,
        taf_run_dir=taf_run_dir,
        n_init=n_init,
        n_iter=n_iter,
        seed=seed,
        n_candidates=64,
        n_starts=4,
        source_only_warmup_iters=1,
        track_acquisition=False,
    )

    runtime = BOTAFServerRuntimeConfig(
        input_dim=2,
        y_min=-400.0,
        y_max=10.0,
        taf_run_dir=taf_run_dir,
        n_candidates_default=64,
        n_starts_default=4,
        source_only_warmup_iters_default=1,
        track_acquisition_default=False,
    )
    session = BOTAFServerSession.from_start_message(
        {
            "type": "start",
            "n_init": n_init,
            "n_iter": n_iter,
            "seed": seed,
            "n_candidates": 64,
            "n_starts": 4,
            "source_only_warmup_iters": 1,
            "track_acquisition": False,
        },
        runtime_config=runtime,
    )

    msg = session.handle({"type": "suggest"})
    while msg["type"] != "done":
        x = np.asarray(msg["x"], dtype=np.float64).reshape(1, -1)
        y = float(objective(x)[0])
        msg = session.handle({"type": "observe", "x": msg["x"], "y": y})

    assert msg["type"] == "done"
    np.testing.assert_allclose(
        np.asarray(msg["x_values"], dtype=np.float64),
        direct.x_obs,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(msg["y_values"], dtype=np.float64),
        direct.y_obs,
        atol=1e-12,
    )


def test_taf_server_runtime_requires_taf_run_dir(tmp_path) -> None:
    """Runtime config parser should require taf_run_dir."""
    config_path = tmp_path / "missing_taf_run_dir.yaml"
    config_path.write_text(
        "\n".join(
            [
                "input_dim: 2",
                "y_range: [-10.0, 10.0]",
            ]
        ),
        encoding="utf-8",
    )
    try:
        BOTAFServerRuntimeConfig.from_yaml_file(config_path)
    except ValueError as exc:
        assert "taf_run_dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing taf_run_dir.")
