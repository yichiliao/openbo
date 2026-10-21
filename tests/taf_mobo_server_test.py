"""Contract and WebSocket checks for TAF-MOBO server sessions (TAF-EIPV and TAF-EHVI)."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from openbo.benchmarks.hypervolume import hypervolume, pareto_mask
from openbo.optimizers.bo_taf_plus import TAFPlusPopulation
from openbo.optimizers.mobo_taf import write_population_sources
from openbo.optimizers.mobo_taf_eipv import TAFEIPVConfig, TAFEIPVSequentialOptimizer
from openbo.server_optimizers.taf_mobo_server import (
    TAFMOBOServerRuntimeConfig,
    TAFMOBOServerSession,
    _taf_mobo_websocket_handler,
)
from openbo.test_functions.taf_plus_users import make_sphere_users

USERS = make_sphere_users(4, 0.1, 0.1, seed=0)
NEW_USER = USERS[-1]
V_REF, W_REF = [-1.0, -1.0, -1.0], [1.2, 1.2, 1.2]


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


def _runtime(population_dir: str | None = None, **overrides: object) -> TAFMOBOServerRuntimeConfig:
    payload: dict[str, object] = {
        "input_dim": 4,
        "y_range": [[-12.0, 2.0]] * 3,
        "v_ref": V_REF,
        "w_ref": W_REF,
        "n_iter_default": 4,
        "n_candidates_default": 64,
        "num_restarts_default": 2,
        "raw_samples_default": 16,
    }
    if population_dir is not None:
        payload["population_dir"] = population_dir
    payload.update(overrides)
    return TAFMOBOServerRuntimeConfig.from_mapping(payload)


def _run(session: TAFMOBOServerSession, seed: int = 1) -> tuple[list[dict], dict]:
    """Answer every suggestion with the new user's noisy objectives; return suggestions and the final message."""
    rng = np.random.default_rng(seed)
    suggestions = []
    message = session.handle({"type": "suggest"})
    while message["type"] == "suggest":
        suggestions.append(message)
        y = NEW_USER.observe(np.asarray(message["x"])[None], rng)[0]
        message = session.handle({"type": "observe", "x": message["x"], "y": y.tolist()})
    return suggestions, message


def test_taf_eipv_session_starts_from_the_sources(population_dir: str) -> None:
    session = TAFMOBOServerSession.from_start_message({"type": "start"}, _runtime(population_dir))
    suggestions, done = _run(session)
    assert [s["phase"] for s in suggestions] == ["sources", "taf", "taf", "taf"]
    assert all(s["method"] == "taf_eipv" and len(s["source_weights"]) == 3 for s in suggestions)
    assert done["type"] == "done" and done["method"] == "taf_eipv"
    y = np.asarray(done["y_values"])
    assert y.shape == (4, 3)
    np.testing.assert_allclose(done["hv_history"], [hypervolume(y[: t + 1], V_REF) for t in range(4)])
    np.testing.assert_allclose(done["pareto_y"], y[pareto_mask(y)])
    assert done["sources"] == ["user_0", "user_1", "user_2"]


def test_first_design_matches_the_in_process_optimizer(population_dir: str) -> None:
    session = TAFMOBOServerSession.from_start_message({"type": "start", "seed": 3}, _runtime(population_dir))
    served = session.handle({"type": "suggest"})["x"]
    config = TAFEIPVConfig(bounds=[(0.0, 1.0)] * 4, v_ref=V_REF, w_ref=W_REF, n_init=0, n_candidates=64, seed=3)
    local = TAFEIPVSequentialOptimizer(config, TAFPlusPopulation.load(population_dir)).suggest()[0]
    np.testing.assert_allclose(served, local)


def test_taf_ehvi_session_takes_one_random_start(population_dir: str) -> None:
    runtime = _runtime(population_dir)
    session = TAFMOBOServerSession.from_start_message({"type": "start", "method": "taf_ehvi", "n_iter": 3}, runtime)
    suggestions, done = _run(session)
    assert [s["phase"] for s in suggestions] == ["random", "taf", "taf"]
    assert suggestions[0]["source_weights"] is None
    assert done["method"] == "taf_ehvi" and np.asarray(done["y_values"]).shape == (3, 3)
    assert sorted(done["sources"]) == ["user_0", "user_1", "user_2"]


def test_sources_from_a_taf_run_dir(population_dir: str, tmp_path) -> None:
    run_dir = write_population_sources(TAFPlusPopulation.load(population_dir), tmp_path / "sources")
    session = TAFMOBOServerSession.from_start_message({"type": "start", "n_iter": 2}, _runtime(taf_run_dir=str(run_dir)))
    suggestions, done = _run(session)
    assert [s["phase"] for s in suggestions] == ["sources", "taf"]
    assert done["sources"] == ["user_0", "user_1", "user_2"]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({}, "exactly one of population_dir and taf_run_dir"),
        ({"population_dir": "x", "taf_run_dir": "y"}, "exactly one"),
        ({"population_dir": "x", "method_default": "qehvi"}, "method must be"),
        ({"population_dir": "x", "weight_mode_default": "uniform", "method_default": "taf_ehvi"}, "weight_mode"),
        ({"population_dir": "x", "v_ref": [2.0, 2.0, 2.0]}, "w_ref must exceed v_ref"),
        ({"population_dir": "x", "y_range": [[0.0, 1.0]]}, "at least two"),
    ],
)
def test_config_is_validated(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _runtime(**overrides)


def test_start_and_observe_are_validated(population_dir: str) -> None:
    runtime = _runtime(population_dir)
    with pytest.raises(ValueError, match="taf_ehvi needs n_init >= 1"):
        TAFMOBOServerSession.from_start_message({"type": "start", "method": "taf_ehvi", "n_init": 0}, runtime)
    with pytest.raises(ValueError, match="0 <= n_init <= n_iter"):
        TAFMOBOServerSession.from_start_message({"type": "start", "n_iter": 2, "n_init": 3}, runtime)
    session = TAFMOBOServerSession.from_start_message({"type": "start"}, runtime)
    x = session.handle({"type": "suggest"})["x"]
    with pytest.raises(ValueError, match="before observe"):
        session.handle({"type": "suggest"})
    with pytest.raises(ValueError, match="list of 3"):
        session.handle({"type": "observe", "y": [0.1, 0.2]})
    with pytest.raises(ValueError, match="outside its configured y_range"):
        session.handle({"type": "observe", "y": [0.1, 5.0, 0.2]})
    with pytest.raises(ValueError, match="does not match"):
        session.handle({"type": "observe", "x": [0.0] * 4, "y": [0.1, 0.2, 0.3]})
    session.handle({"type": "observe", "x": x, "y": [0.1, 0.2, 0.3]})
    status = session.handle({"type": "status"})
    assert status["n_observations"] == 1 and status["has_pending"] and status["n_sources"] == 3
    stopped = session.handle({"type": "stop", "reason": "test"})
    assert stopped["type"] == "stopped" and stopped["total_observations"] == 1


def test_websocket_round_trip(population_dir: str) -> None:
    runtime = _runtime(population_dir)

    async def scenario() -> dict:
        async with serve(lambda ws: _taf_mobo_websocket_handler(ws, runtime), "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}") as ws:
                await ws.send(json.dumps({"type": "observe", "y": [0, 0, 0]}))
                assert json.loads(await ws.recv())["type"] == "error"  # start first
                await ws.send(json.dumps({"type": "start", "n_iter": 2}))
                rng = np.random.default_rng(0)
                message = json.loads(await ws.recv())
                while message["type"] == "suggest":
                    y = NEW_USER.observe(np.asarray(message["x"])[None], rng)[0]
                    await ws.send(json.dumps({"type": "observe", "y": y.tolist()}))
                    message = json.loads(await ws.recv())
                return message

    done = asyncio.run(scenario())
    assert done["type"] == "done" and done["total_observations"] == 2
