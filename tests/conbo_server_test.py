"""Contract and WebSocket checks for the ConBO server (one session per user)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from openbo.optimizers.bo_taf import _load_source_surrogates
from openbo.server_optimizers.conbo_server import (
    ConBOServerRuntimeConfig,
    ConBOServerSession,
    ConBOServerState,
    _conbo_websocket_handler,
)
from openbo.test_functions.synthetic import branin

# Tiny training settings so the tests stay fast; the YAML holds real defaults.
TINY = {
    "r0": 3,
    "dr": 1,
    "candidate_grid_points": 8,
    "replay_grid_points": 6,
    "replay_random_points": 4,
    "population_epochs": 20,
    "online_epochs": 1,
    "hidden_units": 8,
    "n_predict_samples": 4,
    "replay_filter": False,
}


def _config(tmp_path: Path, **overrides: object) -> ConBOServerRuntimeConfig:
    settings: dict[str, object] = {
        "input_dim": 2,
        "y_min": -350.0,
        "y_max": 1.0,
        "memory_dir": str(tmp_path / "memory"),
        "n_evaluations_default": 5,
        "conbo": TINY,
    }
    settings.update(overrides)
    return ConBOServerRuntimeConfig(**settings)  # type: ignore[arg-type]


def _run_user(session: ConBOServerSession) -> dict[str, object]:
    response = session.handle({"type": "suggest"})
    while response["type"] == "suggest":
        x = np.asarray(response["x"])
        response = session.handle(
            {"type": "observe", "x": response["x"], "y": float(branin(x[None, :])[0])}
        )
    return response


def test_users_accumulate_in_memory(tmp_path: Path) -> None:
    config = _config(tmp_path)
    results = []
    for name in ("alice", "bob", "carol"):
        session = ConBOServerSession.from_start_message(
            {"type": "start", "user_name": name}, config
        )
        results.append(_run_user(session))
    assert [r["type"] for r in results] == ["done"] * 3
    assert [r["user_index"] for r in results] == [1, 2, 3]
    assert [r["n_users_in_memory"] for r in results] == [1, 2, 3]
    assert [r["n_random"] for r in results] == [3, 2, 1]
    first, third = results[0], results[2]
    # User 1 has no population model; later users get a weight after their random starts.
    assert first["population_weights"] == [None] * 3 + [0.0, 0.0]
    assert third["population_weights"][0] is None and third["population_weights"][1] == 1.0
    assert third["total_observations"] == 5 and len(third["y_values"]) == 5
    assert third["replay"]["n_users"] == 3
    memory = tmp_path / "memory"
    assert (memory / "population_bnn.pt").exists()
    # The memory directory still works as a TAF source directory.
    assert [s.name for s in _load_source_surrogates(memory)] == ["alice", "bob", "carol"]


def test_suggest_messages_report_phase_and_weight(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _run_user(ConBOServerSession.from_start_message({"type": "start"}, config))
    session = ConBOServerSession.from_start_message({"type": "start"}, config)
    phases = []
    response = session.handle({"type": "suggest"})
    while response["type"] == "suggest":
        phases.append((response["phase"], response["population_weight"]))
        assert response["user_name"] == "user_001"
        x = np.asarray(response["x"])
        response = session.handle({"type": "observe", "y": float(branin(x[None, :])[0])})
    assert phases[:2] == [("random", None), ("random", None)]
    assert phases[2] == ("bo", 1.0)


def test_stop_discards_by_default_and_store_keeps_partial_sessions(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _run_user(ConBOServerSession.from_start_message({"type": "start"}, config))
    memory = tmp_path / "memory"
    before = (memory / "conbo_state.json").read_text()

    abandoned = ConBOServerSession.from_start_message({"type": "start"}, config)
    x = np.asarray(abandoned.handle({"type": "suggest"})["x"])
    abandoned.handle({"type": "observe", "y": float(branin(x[None, :])[0])})
    stopped = abandoned.handle({"type": "stop", "reason": "participant_left"})
    assert stopped["type"] == "stopped" and stopped["stored"] is False
    assert stopped["total_observations"] == 1 and stopped["reason"] == "participant_left"
    assert (memory / "conbo_state.json").read_text() == before

    partial = ConBOServerSession.from_start_message({"type": "start"}, config)
    assert partial.optimizer.name == "user_001"  # the abandoned user left no trace
    x = np.asarray(partial.handle({"type": "suggest"})["x"])
    partial.handle({"type": "observe", "y": float(branin(x[None, :])[0])})
    kept = partial.handle({"type": "stop", "store": True})
    assert kept["stored"] is True and kept["n_users_in_memory"] == 2

    empty = ConBOServerSession.from_start_message({"type": "start"}, config)
    empty.handle({"type": "suggest"})
    nothing = empty.handle({"type": "stop", "store": True})
    assert nothing["stored"] is False and nothing["n_users_in_memory"] == 2


def test_restart_reloads_the_population(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _run_user(ConBOServerSession.from_start_message({"type": "start"}, config))
    # A new config object, as after a server restart, with different ConBO settings:
    # the saved population keeps its own settings.
    restarted = _config(tmp_path, conbo={**TINY, "r0": 9})
    session = ConBOServerSession.from_start_message({"type": "start"}, restarted)
    assert session.optimizer.user_index == 2
    assert session.optimizer.n_random == 2  # r0 = 3, dr = 1 from the saved config
    status = session.handle({"type": "status"})
    assert status["n_users_in_memory"] == 1 and status["n_random"] == 2


def test_observe_is_validated_without_advancing(tmp_path: Path) -> None:
    session = ConBOServerSession.from_start_message(
        {"type": "start", "n_evaluations": 2}, _config(tmp_path)
    )
    suggestion = session.handle({"type": "suggest"})
    with pytest.raises(ValueError, match="pending"):
        session.handle({"type": "suggest"})
    for bad, message in [
        ({}, "needs a scalar"),
        ({"y": 5.0}, "outside"),
        ({"y": float("nan")}, "outside"),
        ({"y": [1.0]}, "number"),
        ({"y": True}, "number"),
        ({"y": -1.0, "x": [0.0, 0.0]}, "does not match"),
    ]:
        with pytest.raises(ValueError, match=message):
            session.handle({"type": "observe", **bad})
    assert session.handle({"type": "status"})["n_observations"] == 0
    session.handle({"type": "observe", "x": suggestion["x"], "y": -1.0})
    assert session.handle({"type": "status"})["n_observations"] == 1


def test_start_and_config_validation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="n_evaluations"):
        ConBOServerSession.from_start_message({"n_evaluations": 0}, config)
    with pytest.raises(ValueError, match="user_name"):
        ConBOServerSession.from_start_message({"user_name": "../evil"}, config)
    _run_user(ConBOServerSession.from_start_message({"user_name": "dana"}, config))
    with pytest.raises(ValueError, match="already in memory"):
        ConBOServerSession.from_start_message({"user_name": "dana"}, config)
    with pytest.raises(ValueError, match="2D population"):
        ConBOServerSession.from_start_message({}, _config(tmp_path, input_dim=3))
    with pytest.raises(ValueError, match="y_range"):
        _config(tmp_path, y_min=1.0, y_max=0.0)
    with pytest.raises(ValueError, match="r0"):
        _config(tmp_path, conbo={"r0": -1})
    with pytest.raises(ValueError, match="top level"):
        _config(tmp_path, conbo={"input_dim": 2})


def test_yaml_config_loads() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/server_optimizers/conbo_server.yaml"
    config = ConBOServerRuntimeConfig.from_yaml_file(path)
    assert (config.input_dim, config.y_min, config.y_max) == (2, -350.0, 1.0)
    assert config.conbo_config().replay_variance_fraction == 0.05
    assert config.conbo_config().replay_max_variance is None


def test_websocket_one_user_at_a_time(tmp_path: Path) -> None:
    config = _config(tmp_path, n_evaluations_default=3)
    state = ConBOServerState()

    async def run() -> None:
        async def handler(websocket: object) -> None:
            await _conbo_websocket_handler(websocket, config, state)

        async with serve(handler, "127.0.0.1", 0) as server:
            url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            async with connect(url) as first:
                await first.send(json.dumps({"type": "start", "user_name": "erin"}))
                message = json.loads(await first.recv())
                assert message["type"] == "suggest"
                async with connect(url) as second:
                    await second.send(json.dumps({"type": "start"}))
                    refused = json.loads(await second.recv())
                    assert refused["type"] == "error"
                    assert "another user session is active" in refused["message"]
                while message["type"] == "suggest":
                    y = float(branin(np.asarray(message["x"])[None, :])[0])
                    await first.send(json.dumps({"type": "observe", "y": y}))
                    message = json.loads(await first.recv())
                assert message["type"] == "done" and message["stored"] is True
            # The lock is released, so the next user can start.
            async with connect(url) as third:
                await third.send(json.dumps({"type": "start"}))
                assert json.loads(await third.recv())["user_index"] == 2
                await third.send(json.dumps({"type": "stop"}))
                assert json.loads(await third.recv())["stored"] is False
            # A dropped connection also releases the lock and stores nothing.
            async with connect(url) as dropped:
                await dropped.send(json.dumps({"type": "start"}))
                await dropped.recv()
            await asyncio.sleep(0.1)
            async with connect(url) as fourth:
                await fourth.send(json.dumps({"type": "start"}))
                message = json.loads(await fourth.recv())
                assert message["type"] == "suggest" and message["user_index"] == 2

    asyncio.run(run())
    assert state.active_user is None
