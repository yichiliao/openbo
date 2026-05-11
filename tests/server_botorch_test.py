"""Tests for generic websocket-session optimizer logic."""

from __future__ import annotations

import json
import numpy as np

from openbo.server_optimizers.bo_server import (
    BOServerRuntimeConfig,
    BOServerSession,
)


def test_server_session_runs_to_done_botorch() -> None:
    """Session should progress start -> suggest/observe -> done."""
    runtime = BOServerRuntimeConfig(
        optimizer="bo_botorch",
        input_dim=2,
        y_min=-100.0,
        y_max=100.0,
    )
    session = BOServerSession.from_start_message(
        {
            "type": "start",
            "n_init": 2,
            "n_iter": 2,
            "seed": 0,
        },
        runtime_config=runtime,
    )

    msg = session.handle({"type": "suggest"})
    assert msg["type"] == "suggest"
    assert msg["phase"] == "init"

    for _ in range(3):
        x = np.asarray(msg["x"], dtype=np.float64)
        y = float(-(np.sum((x - 0.2) ** 2)))
        msg = session.handle({"type": "observe", "x": msg["x"], "y": y})
        assert msg["type"] in {"suggest", "done"}
        if msg["type"] == "done":
            break

    if msg["type"] != "done":
        x = np.asarray(msg["x"], dtype=np.float64)
        y = float(-(np.sum((x - 0.2) ** 2)))
        msg = session.handle({"type": "observe", "x": msg["x"], "y": y})

    assert msg["type"] == "done"
    assert msg["total_observations"] == 4
    assert len(msg["x_values"]) == 4
    assert len(msg["y_values"]) == 4


def test_server_session_stop_message() -> None:
    """Client should be able to stop an in-progress session."""
    runtime = BOServerRuntimeConfig(
        optimizer="bo_botorch",
        input_dim=2,
        y_min=-100.0,
        y_max=100.0,
    )
    session = BOServerSession.from_start_message(
        {
            "type": "start",
            "n_init": 2,
            "n_iter": 5,
            "seed": 0,
        },
        runtime_config=runtime,
    )
    suggest = session.handle({"type": "suggest"})
    assert suggest["type"] == "suggest"
    stopped = session.handle({"type": "stop", "reason": "user_cancelled"})
    assert stopped["type"] == "stopped"
    assert stopped["reason"] == "user_cancelled"
    assert "x_values" in stopped
    assert "y_values" in stopped


def test_server_session_observe_checks_y_range() -> None:
    """Observe should reject y outside configured server y-range."""
    runtime = BOServerRuntimeConfig(
        optimizer="bo_botorch",
        input_dim=2,
        y_min=-1.0,
        y_max=1.0,
    )
    session = BOServerSession.from_start_message(
        {
            "type": "start",
            "n_init": 1,
            "n_iter": 0,
            "seed": 0,
        },
        runtime_config=runtime,
    )
    suggest = session.handle({"type": "suggest"})
    assert suggest["type"] == "suggest"
    try:
        session.handle({"type": "observe", "x": suggest["x"], "y": 10.0})
    except ValueError as exc:
        assert "outside configured y_range" in str(exc)
    else:
        raise AssertionError("Expected ValueError for out-of-range y.")


def test_server_session_runs_to_done_scratch() -> None:
    """Session should also run to done with scratch backend."""
    runtime = BOServerRuntimeConfig(
        optimizer="bo_scratch",
        input_dim=2,
        y_min=-100.0,
        y_max=100.0,
    )
    session = BOServerSession.from_start_message(
        {
            "type": "start",
            "n_init": 2,
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
    assert msg["optimizer"] == "bo_scratch"


def test_server_session_scratch_auto_saves_artifacts(tmp_path) -> None:
    """Scratch server can auto-save trajectories and GP states."""
    runtime = BOServerRuntimeConfig(
        optimizer="bo_scratch",
        input_dim=2,
        y_min=-100.0,
        y_max=100.0,
        auto_save_scratch_artifacts=True,
        scratch_artifacts_dir=str(tmp_path / "scratch-artifacts"),
    )
    session = BOServerSession.from_start_message(
        {
            "type": "start",
            "task_name": "train_task_000",
            "n_init": 2,
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
    traj_path = tmp_path / "scratch-artifacts" / "trajectories" / "train_task_000.json"
    gp_path = tmp_path / "scratch-artifacts" / "gp_states" / "train_task_000.json"
    assert traj_path.exists()
    assert gp_path.exists()

    traj_payload = json.loads(traj_path.read_text(encoding="utf-8"))
    gp_payload = json.loads(gp_path.read_text(encoding="utf-8"))
    assert traj_payload["task_name"] == "train_task_000"
    assert gp_payload["task_name"] == "train_task_000"
    assert "gp_state" in gp_payload
