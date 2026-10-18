"""Tests for the ConBO benchmark method dispatcher."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openbo.benchmarks import conbo_runner
from openbo.benchmarks.conbo_runner import CONBO_METHODS, ConBOTest, run_conbo_method

FAST = {
    "population_epochs": 30,
    "online_epochs": 2,
    "hidden_units": 16,
    "n_predict_samples": 8,
    "candidate_grid_points": 10,
    "replay_grid_points": 8,
    "replay_random_points": 5,
}


@pytest.fixture(autouse=True)
def tiny_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        conbo_runner.CONBO_TESTS, "tiny", ConBOTest("branin", 3, 0.3, 0.2, 6, 4, 2, 50.0)
    )


@pytest.mark.parametrize("method", CONBO_METHODS)
def test_every_method_runs_the_user_sequence(method: str, tmp_path: Path) -> None:
    result = run_conbo_method(method, "tiny", seed=0, work_dir=tmp_path, config_overrides=FAST)
    users = result["users"]
    assert [u["name"] for u in users] == ["user_000", "user_001", "user_002"]
    expected_random = [4, 4, 4] if method == "bo" else [4, 2, 0]
    assert [u["n_random"] for u in users] == expected_random
    for user in users:
        x = np.asarray(user["x"])
        y = np.asarray(user["y"])
        assert x.shape == (6, 2) and y.shape == (6,)
        assert np.all((x >= 0.0) & (x <= 1.0))
        assert np.max(y) <= user["optimum"] + 1e-9
        assert user["seconds"] >= 0.0


def test_taf_uses_previous_users_as_sources(tmp_path: Path) -> None:
    run_conbo_method("taf_m", "tiny", seed=0, work_dir=tmp_path, config_overrides=FAST)
    sources = sorted(p.stem for p in (tmp_path / "taf_m_tiny_s0" / "gp_states").glob("*.json"))
    assert sources == ["user_000", "user_001", "user_002"]


def test_users_differ_by_seed_and_runs_repeat(tmp_path: Path) -> None:
    a = run_conbo_method("single_gp", "tiny", 0, tmp_path / "a", FAST)
    b = run_conbo_method("single_gp", "tiny", 0, tmp_path / "b", FAST)
    c = run_conbo_method("single_gp", "tiny", 1, tmp_path / "c", FAST)
    assert a["users"][2]["x"] == b["users"][2]["x"]
    assert a["users"][0]["variant"] != c["users"][0]["variant"]


def test_unknown_method_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        run_conbo_method("magic", "tiny", 0, tmp_path)
