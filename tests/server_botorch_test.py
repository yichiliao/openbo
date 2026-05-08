"""Tests for BoTorch websocket-session optimizer logic."""

from __future__ import annotations

import numpy as np

from openbo.server_optimizers.bo_botorch_server import BoTorchServerSession


def test_botorch_server_session_runs_to_done() -> None:
    """Session should progress start -> suggest/observe -> done."""
    session = BoTorchServerSession.from_start_message(
        {
            "type": "start",
            "bounds": [[0.0, 1.0], [0.0, 1.0]],
            "n_init": 2,
            "n_iter": 2,
            "seed": 0,
        }
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
