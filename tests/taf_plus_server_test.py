"""Contract and WebSocket checks for TAF+ server sessions."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from openbo.optimizers.bo_taf_plus import TAFPlusPopulation, TAFPlusSequentialOptimizer
from openbo.server_optimizers.taf_plus_server import (
    TAFPlusServerRuntimeConfig,
    TAFPlusServerSession,
    _taf_plus_websocket_handler,
)
from openbo.test_functions.taf_plus_users import make_sphere_users

USERS = make_sphere_users(4, 0.1, 0.1, seed=0)
NEW_USER = USERS[-1]


@pytest.fixture(scope="module")
def population_dir(tmp_path_factory) -> str:
    """A small saved population: three sphere users with 15 random points each."""
    rng = np.random.default_rng(0)
    population = TAFPlusPopulation(input_dim=4, n_objectives=3)
    for i, user in enumerate(USERS[:3]):
        x = rng.random((15, 4))
        population.add_user(f"user_{i}", x, user.observe(x, rng))
    directory = tmp_path_factory.mktemp("population")
    population.save(directory)
    return str(directory)


def _runtime(population_dir: str, **overrides: object) -> TAFPlusServerRuntimeConfig:
    """Return a small runtime config for the sphere users."""
    payload: dict[str, object] = {
        "input_dim": 4,
        "y_range": [[-12.0, 2.0]] * 3,
        "population_dir": population_dir,
        "objective_weights_default": [0.3, 0.5, 0.2],
        "d1_default": 1,
        "d2_default": 0.5,
        "n_iter_default": 4,
        "n_candidates_default": 64,
        "refine_suggestion_default": False,
    }
    payload.update(overrides)
    return TAFPlusServerRuntimeConfig.from_mapping(payload)


def test_session_runs_to_done_with_vector_observations(population_dir: str) -> None:
    """A full session should start from the population, then run TAF+ with the decay."""
    session = TAFPlusServerSession.from_start_message({"type": "start"}, _runtime(population_dir))
    message = session.handle({"type": "suggest"})
    phases, decays = [], []
    rng = np.random.default_rng(1)
    while message["type"] == "suggest":
        phases.append(message["phase"])
        decays.append(message["decay"])
        y = NEW_USER.observe(np.asarray(message["x"])[None], rng)[0]
        message = session.handle({"type": "observe", "x": message["x"], "y": y.tolist()})
    assert phases == ["population"] + ["taf_plus"] * 3
    # d1 = 1, d2 = 0.5: delta(2) = 0.5, then 0.
    assert decays == [None, 0.5, 0.0, 0.0]
    assert message["type"] == "done"
    y_values = np.asarray(message["y_values"])
    assert y_values.shape == (4, 3)
    np.testing.assert_allclose(message["weighted_y"], y_values @ np.array([0.3, 0.5, 0.2]))
    np.testing.assert_allclose(message["best_so_far"], np.maximum.accumulate(message["weighted_y"]))
    assert message["decay_history"] == decays
    assert message["population_users"] == ["user_0", "user_1", "user_2"]


def test_first_design_matches_the_in_process_optimizer(population_dir: str) -> None:
    """The server should suggest exactly what TAFPlusSequentialOptimizer suggests."""
    runtime = _runtime(population_dir)
    session = TAFPlusServerSession.from_start_message({"type": "start", "seed": 3}, runtime)
    first = session.handle({"type": "suggest"})
    optimizer = TAFPlusSequentialOptimizer(
        runtime.taf_plus_config({"seed": 3}), TAFPlusPopulation.load(population_dir)
    )
    np.testing.assert_allclose(first["x"], optimizer.suggest()[0])


def test_invalid_observations_are_rejected_without_advancing(population_dir: str) -> None:
    """Scalar, wrong-length, out-of-range and mismatched observations should all fail."""
    session = TAFPlusServerSession.from_start_message({"type": "start"}, _runtime(population_dir))
    suggestion = session.handle({"type": "suggest"})
    bad_payloads = [
        ({"type": "observe", "y": 0.5}, "list of 3"),
        ({"type": "observe", "y": [0.0, 0.0]}, "list of 3"),
        ({"type": "observe", "y": [0.0, 5.0, 0.0]}, r"y\[1\]"),
        ({"type": "observe", "y": [0.0, float("nan"), 0.0]}, "finite"),
        ({"type": "observe"}, "needs 'y'"),
        ({"type": "observe", "y": [0.0, 0.0, 0.0], "x": [0.0] * 4}, "does not match"),
    ]
    for payload, match in bad_payloads:
        with pytest.raises(ValueError, match=match):
            session.handle(payload)
    assert session.optimizer.n_observations == 0
    with pytest.raises(ValueError, match="before observe"):
        session.handle({"type": "suggest"})
    status = session.handle({"type": "status"})
    assert status["has_pending"] and status["n_observations"] == 0
    assert status["n_population_users"] == 3 and status["best_weighted_y"] is None
    accepted = session.handle({"type": "observe", "x": suggestion["x"], "y": [0.0, 0.0, 0.0]})
    assert accepted["type"] == "suggest" and accepted["phase"] == "taf_plus"


def test_stop_and_start_overrides(population_dir: str) -> None:
    """Stop should work before any data; start overrides should reach the optimizer."""
    session = TAFPlusServerSession.from_start_message(
        {
            "type": "start",
            "objective_weights": [1.0, 0.0, 0.0],
            "decay": False,
            "weight_mode": "taf_r",
            "first_point": "centre",
            "n_iter": 2,
        },
        _runtime(population_dir),
    )
    config = session.optimizer.config
    assert config.objective_weights == (1.0, 0.0, 0.0)
    assert not config.decay and config.weight_mode == "taf_r" and config.first_point == "centre"
    assert session.n_iter == 2
    assert session.handle({"type": "suggest"})["x"] == [0.5] * 4
    stopped = session.handle({"type": "stop", "reason": "test"})
    assert stopped["type"] == "stopped" and stopped["reason"] == "test"
    assert stopped["total_observations"] == 0 and stopped["decay_history"] == []
    runtime = _runtime(population_dir)
    for payload, match in [
        ({"objective_weights": [0.5, 0.5]}, "3 entries"),
        ({"objective_weights": [0.5, 0.6, 0.0]}, "sum to 1"),
        ({"weight_mode": "taf_m"}, "weight_mode"),
        ({"first_point": "random"}, "first_point"),
        ({"n_iter": 0}, "n_iter"),
        ({"d2": 0.0}, "d2"),
    ]:
        with pytest.raises(ValueError, match=match):
            TAFPlusServerSession.from_start_message({"type": "start", **payload}, runtime)


def test_runtime_config_validation(population_dir: str, tmp_path) -> None:
    """Malformed configs and mismatched populations should raise."""
    runtime = TAFPlusServerRuntimeConfig.from_mapping(
        {"input_dim": 4, "y_range": [[-12.0, 2.0]] * 3, "population_dir": population_dir}
    )
    assert runtime.objective_weights_default == pytest.approx((1 / 3,) * 3)
    with pytest.raises(ValueError, match="y_range"):
        _runtime(population_dir, y_range=[-1.0, 1.0])
    with pytest.raises(ValueError, match="one entry per objective"):
        _runtime(population_dir, objective_weights_default=[0.5, 0.5])
    with pytest.raises(ValueError, match="population_dir"):
        TAFPlusServerRuntimeConfig.from_mapping({"input_dim": 4, "y_range": [[0.0, 1.0]] * 3})
    with pytest.raises(ValueError, match="No TAF\\+ population"):
        _runtime(str(tmp_path)).load_population()
    with pytest.raises(ValueError, match="input_dim"):
        _runtime(population_dir, input_dim=3).load_population()
    with pytest.raises(ValueError, match="objectives"):
        _runtime(population_dir, y_range=[[-12.0, 2.0]] * 2, objective_weights_default=[0.5, 0.5]).load_population()


def test_committed_config_loads() -> None:
    """The shipped server config should parse and match its archived population."""
    runtime = TAFPlusServerRuntimeConfig.from_yaml_file("configs/server_optimizers/taf_plus_server.yaml")
    population = runtime.load_population()
    assert population.n_users == 10
    assert runtime.input_dim == population.input_dim == 4
    assert runtime.n_objectives == population.n_objectives == 3


def test_real_websocket_roundtrip(population_dir: str) -> None:
    """A real socket session should complete and return vector y values."""

    async def run() -> None:
        runtime = _runtime(population_dir, n_iter_default=3)

        async def handler(websocket: object) -> None:
            await _taf_plus_websocket_handler(websocket, runtime)

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}") as websocket:
                await websocket.send(json.dumps({"type": "observe", "y": [0.0, 0.0, 0.0]}))
                assert json.loads(await websocket.recv())["type"] == "error"
                await websocket.send(json.dumps({"type": "start", "seed": 1}))
                message = json.loads(await websocket.recv())
                rng = np.random.default_rng(2)
                while message["type"] == "suggest":
                    y = NEW_USER.observe(np.asarray(message["x"])[None], rng)[0]
                    await websocket.send(json.dumps({"type": "observe", "y": y.tolist()}))
                    message = json.loads(await websocket.recv())
                assert message["type"] == "done"
                assert np.asarray(message["y_values"]).shape == (3, 3)

    asyncio.run(run())
