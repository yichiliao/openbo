"""Contract and WebSocket checks for MOBO server sessions."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from openbo.benchmarks.hypervolume import hypervolume
from openbo.server_optimizers.mobo_server import (
    MOBOServerRuntimeConfig,
    MOBOServerSession,
    _mobo_websocket_handler,
)
from openbo.test_functions.multi_objective import get_mo_function_spec

SPEC = get_mo_function_spec("oka2")


def _runtime(**overrides: object) -> MOBOServerRuntimeConfig:
    """Return a small OKA2 runtime config."""
    payload: dict[str, object] = {
        "input_dim": 3,
        "y_range": [[-3.2, 3.1416], [-5.1, 0.0]],
        "n_init_default": 3,
        "n_iter_default": 2,
        "n_candidates_default": 32,
    }
    payload.update(overrides)
    return MOBOServerRuntimeConfig.from_mapping(payload)


def test_session_runs_to_done_with_vector_observations() -> None:
    """A full session should alternate suggest/observe and finish with the Pareto set."""
    session = MOBOServerSession.from_start_message({"type": "start", "seed": 0}, _runtime())
    message = session.handle({"type": "suggest"})
    phases = []
    while message["type"] == "suggest":
        phases.append(message["phase"])
        y = SPEC.objective(np.asarray(message["x"])[None])[0]
        message = session.handle({"type": "observe", "x": message["x"], "y": y.tolist()})
    assert phases == ["init"] * 3 + ["bo"] * 2
    assert message["type"] == "done"
    y_values = np.asarray(message["y_values"])
    assert y_values.shape == (5, 2)
    # The server observes points one at a time, so there is one entry per observation.
    assert len(message["hv_history"]) == 5
    assert np.all(np.diff(message["hv_history"]) >= 0.0)
    assert message["hv_history"][-1] == pytest.approx(hypervolume(y_values, SPEC.v_ref), rel=1e-6)
    assert len(message["pareto_y"]) == len(message["pareto_x"]) >= 1


def test_invalid_observations_are_rejected_without_advancing() -> None:
    """Scalar, wrong-length, out-of-range and mismatched observations should all fail."""
    session = MOBOServerSession.from_start_message({"type": "start"}, _runtime())
    suggestion = session.handle({"type": "suggest"})
    bad_payloads = [
        ({"type": "observe", "y": 0.5}, "list of 2"),
        ({"type": "observe", "y": [0.0, -1.0, 2.0]}, "list of 2"),
        ({"type": "observe", "y": [0.0, 1.0]}, r"y\[1\]"),
        ({"type": "observe", "y": [0.0, float("nan")]}, "finite"),
        ({"type": "observe"}, "needs 'y'"),
        ({"type": "observe", "y": [0.0, -1.0], "x": [0.0, 0.0, 0.0]}, "does not match"),
    ]
    for payload, match in bad_payloads:
        with pytest.raises(ValueError, match=match):
            session.handle(payload)
    assert session.optimizer.x_obs.shape[0] == 0
    with pytest.raises(ValueError, match="before observe"):
        session.handle({"type": "suggest"})
    status = session.handle({"type": "status"})
    assert status["has_pending"] and status["n_observations"] == 0
    accepted = session.handle({"type": "observe", "x": suggestion["x"], "y": [0.0, -1.0]})
    assert accepted["type"] == "suggest"


def test_stop_and_start_overrides() -> None:
    """Stop should work before any data; start overrides should reach the optimizer."""
    session = MOBOServerSession.from_start_message(
        {"type": "start", "model": "independent", "acquisition": "ieipv", "n_init": 2},
        _runtime(),
    )
    assert session.optimizer.config.model == "independent"
    assert session.optimizer.config.acquisition == "ieipv"
    stopped = session.handle({"type": "stop", "reason": "test"})
    assert stopped["type"] == "stopped" and stopped["reason"] == "test"
    assert stopped["total_observations"] == 0
    with pytest.raises(ValueError, match="model"):
        MOBOServerSession.from_start_message({"type": "start", "model": "lmc"}, _runtime())
    with pytest.raises(ValueError, match="n_init"):
        MOBOServerSession.from_start_message({"type": "start", "n_init": 0}, _runtime())


def test_runtime_config_validation_and_reference_defaults() -> None:
    """y_range defines v_ref/w_ref by default; malformed configs should raise."""
    runtime = _runtime()
    assert runtime.v_ref == [-3.2, -5.1] and runtime.w_ref == [3.1416, 0.0]
    assert _runtime(v_ref=[-4.0, -6.0]).v_ref == [-4.0, -6.0]
    with pytest.raises(ValueError, match="y_range"):
        _runtime(y_range=[-1.0, 1.0])
    with pytest.raises(ValueError, match="y_max > y_min"):
        _runtime(y_range=[[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="one entry per objective"):
        _runtime(v_ref=[0.0])
    with pytest.raises(ValueError, match="acquisition"):
        _runtime(acquisition="ehvi")


def test_committed_config_loads() -> None:
    """The shipped server config should parse and match OKA2's reference box."""
    runtime = MOBOServerRuntimeConfig.from_yaml_file("configs/server_optimizers/mobo_server.yaml")
    assert runtime.input_dim == SPEC.dim
    assert len(runtime.y_range) == SPEC.n_objectives
    assert np.all(np.asarray(runtime.v_ref) <= np.asarray(SPEC.v_ref))
    assert np.all(np.asarray(runtime.w_ref) >= np.asarray(SPEC.w_ref))


def test_real_websocket_roundtrip() -> None:
    """A real socket session should complete and return vector y values."""

    async def run() -> None:
        runtime = _runtime(n_init_default=2, n_iter_default=1)

        async def handler(websocket: object) -> None:
            await _mobo_websocket_handler(websocket, runtime)

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}") as websocket:
                await websocket.send(json.dumps({"type": "observe", "y": [0.0, 0.0]}))
                assert json.loads(await websocket.recv())["type"] == "error"
                await websocket.send(json.dumps({"type": "start", "seed": 1}))
                message = json.loads(await websocket.recv())
                while message["type"] == "suggest":
                    y = SPEC.objective(np.asarray(message["x"])[None])[0]
                    await websocket.send(json.dumps({"type": "observe", "y": y.tolist()}))
                    message = json.loads(await websocket.recv())
                assert message["type"] == "done"
                assert np.asarray(message["y_values"]).shape == (3, 2)

    asyncio.run(run())
