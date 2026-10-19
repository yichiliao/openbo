"""Contract and WebSocket checks for NAF server sessions."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from openbo.optimizers.naf import (
    BRANIN_GP_HYPERPARAMETERS,
    NAFConfig,
    NAFSequentialOptimizer,
    run_naf,
    save_naf,
)
from openbo.server_optimizers.naf_server import (
    NAFServerRuntimeConfig,
    NAFServerSession,
    _naf_websocket_handler,
)
from openbo.test_functions.metabo_families import sample_branin_tasks

TASK = sample_branin_tasks(1, seed=7)[0]
BUDGET = 5


@pytest.fixture(scope="module")
def af_dir(tmp_path_factory) -> str:
    """A small untrained AF saved as ``best_af`` (2D, budget 5, small candidate set)."""
    config = NAFConfig(dim=2, budget=BUDGET, hidden=(16, 16), points_per_dim=11, n_local=20, steps_per_iteration=BUDGET)
    directory = tmp_path_factory.mktemp("naf")
    save_naf(directory, config.new_af(), config, BRANIN_GP_HYPERPARAMETERS, name="best_af")
    return str(directory)


def _runtime(af_dir: str, **overrides: object) -> NAFServerRuntimeConfig:
    payload: dict[str, object] = {"input_dim": 2, "y_range": [-12.0, 2.0], "af_dir": af_dir}
    payload.update(overrides)
    return NAFServerRuntimeConfig.from_mapping(payload)


def _value(x: list[float]) -> float:
    return float(TASK(np.asarray(x)[None, :])[0])


def test_session_matches_the_in_process_optimizer(af_dir: str) -> None:
    """A full session should suggest exactly what run_naf does, for the AF's whole budget."""
    session = NAFServerSession.from_start_message({"type": "start"}, _runtime(af_dir))
    message = session.handle({"type": "suggest"})
    iterations = []
    while message["type"] == "suggest":
        assert message["budget"] == BUDGET and message["n_iter"] == BUDGET
        iterations.append(message["iteration"])
        message = session.handle({"type": "observe", "x": message["x"], "y": _value(message["x"])})
    assert message["type"] == "done" and iterations == list(range(BUDGET))
    x_ref, y_ref = run_naf(TASK, NAFSequentialOptimizer.from_directory(af_dir))
    np.testing.assert_allclose(message["x_values"], x_ref)
    np.testing.assert_allclose(message["y_values"], y_ref)
    np.testing.assert_allclose(message["best_so_far"], np.maximum.accumulate(y_ref))
    # After done, a further suggest just repeats the result.
    assert session.handle({"type": "suggest"})["type"] == "done"


def test_invalid_observations_are_rejected_without_advancing(af_dir: str) -> None:
    session = NAFServerSession.from_start_message({"type": "start"}, _runtime(af_dir))
    message = session.handle({"type": "suggest"})
    with pytest.raises(ValueError, match="before observe"):
        session.handle({"type": "suggest"})
    for bad, match in [(5.0, "outside"), ([0.1], "single number"), (True, "single number"), (float("nan"), "finite")]:
        with pytest.raises(ValueError, match=match):
            session.handle({"type": "observe", "y": bad})
    with pytest.raises(ValueError, match="needs 'y'"):
        session.handle({"type": "observe"})
    with pytest.raises(ValueError, match="does not match"):
        session.handle({"type": "observe", "x": [0.123, 0.456], "y": 0.0})
    assert session.optimizer.t == 0
    session.handle({"type": "observe", "x": message["x"], "y": 0.0})
    assert session.optimizer.t == 1
    with pytest.raises(ValueError, match="Unsupported"):
        session.handle({"type": "bogus"})


def test_n_iter_status_and_stop(af_dir: str) -> None:
    session = NAFServerSession.from_start_message({"type": "start", "n_iter": 2}, _runtime(af_dir))
    message = session.handle({"type": "suggest"})
    message = session.handle({"type": "observe", "y": _value(message["x"])})
    status = session.handle({"type": "status"})
    assert status["n_observations"] == 1 and status["has_pending"] and status["n_iter"] == 2
    message = session.handle({"type": "observe", "y": _value(message["x"])})
    assert message["type"] == "done" and message["total_observations"] == 2
    stopped = NAFServerSession.from_start_message({"type": "start"}, _runtime(af_dir)).handle({"type": "stop"})
    assert stopped["type"] == "stopped" and stopped["x_values"] == [] and stopped["best_so_far"] == []
    # The config's default, and the budget limit.
    assert NAFServerSession.from_start_message({}, _runtime(af_dir, n_iter_default=3)).n_iter == 3
    with pytest.raises(ValueError, match="training budget"):
        NAFServerSession.from_start_message({"n_iter": BUDGET + 1}, _runtime(af_dir))
    with pytest.raises(ValueError, match="training budget"):
        NAFServerSession.from_start_message({"n_iter": 0}, _runtime(af_dir))


def test_runtime_config_validation(af_dir: str, tmp_path) -> None:
    with pytest.raises(ValueError, match="y_max > y_min"):
        _runtime(af_dir, y_range=[1.0, 0.0])
    with pytest.raises(ValueError, match=r"\[y_min, y_max\]"):
        _runtime(af_dir, y_range=[[0.0, 1.0]])
    with pytest.raises(ValueError, match="af_dir is required"):
        NAFServerRuntimeConfig.from_mapping({"input_dim": 2, "y_range": [0.0, 1.0]})
    with pytest.raises(ValueError, match="No trained NAF"):
        _runtime(str(tmp_path)).load_optimizer()
    with pytest.raises(ValueError, match="No trained NAF"):
        _runtime(af_dir, af_name="missing").load_optimizer()
    with pytest.raises(ValueError, match="input_dim"):
        _runtime(af_dir, input_dim=3).load_optimizer()


def test_committed_config_loads_the_archived_af() -> None:
    """The shipped config should load the archived Branin AF (budget 30)."""
    runtime = NAFServerRuntimeConfig.from_yaml_file("configs/server_optimizers/naf_server.yaml")
    optimizer = runtime.load_optimizer()
    assert runtime.input_dim == optimizer.af.dim == 2
    assert optimizer.budget == 30
    assert NAFServerSession.from_start_message({}, runtime, optimizer).n_iter == 30


def test_real_websocket_roundtrip(af_dir: str) -> None:
    """A real socket session should complete and return scalar y values."""

    async def run() -> None:
        runtime = _runtime(af_dir)

        async def handler(websocket: object) -> None:
            await _naf_websocket_handler(websocket, runtime)

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}") as websocket:
                await websocket.send(json.dumps({"type": "observe", "y": 0.0}))
                assert json.loads(await websocket.recv())["type"] == "error"
                await websocket.send(json.dumps({"type": "start", "n_iter": 3}))
                message = json.loads(await websocket.recv())
                await websocket.send(json.dumps({"type": "start"}))
                assert "already started" in json.loads(await websocket.recv())["message"]
                while message["type"] == "suggest":
                    await websocket.send(json.dumps({"type": "observe", "y": _value(message["x"])}))
                    message = json.loads(await websocket.recv())
                assert message["type"] == "done"
                assert len(message["y_values"]) == 3

    asyncio.run(run())


@pytest.mark.parametrize(
    "family, config_path",
    [
        ("goldstein_price", "configs/server_optimizers/naf_server_goldstein_price.yaml"),
        ("hartmann3", "configs/server_optimizers/naf_server_hartmann3.yaml"),
    ],
)
def test_committed_configs_for_the_other_families_load_their_archived_af(family: str, config_path: str) -> None:
    """Each shipped config loads its archived AF, and its y_range covers the family's values."""
    from openbo.optimizers.naf import get_naf_family

    spec = get_naf_family(family)
    runtime = NAFServerRuntimeConfig.from_yaml_file(config_path)
    optimizer = runtime.load_optimizer()
    assert runtime.input_dim == optimizer.af.dim == spec.dim
    assert optimizer.budget == 30
    assert NAFServerSession.from_start_message({}, runtime, optimizer).n_iter == 30
    # The AF was trained on this family: its GP hyperparameters are the family's.
    assert optimizer.gp.variance == pytest.approx(spec.gp_hyperparameters.variance)
    rng = np.random.default_rng(0)
    lo, hi = runtime.y_range
    for task in spec.sample_tasks(200, 1) + spec.grid_tasks(50):
        values = task(np.vstack([rng.random((1000, spec.dim)), np.array(np.meshgrid(*[[0.0, 1.0]] * spec.dim)).reshape(spec.dim, -1).T]))
        assert lo <= values.min() and values.max() <= hi, f"{family}: y_range {runtime.y_range} misses {values.min():.3f}..{values.max():.3f}"
