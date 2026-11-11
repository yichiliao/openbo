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


def test_iteration_times_and_variance_maps_are_recorded(tmp_path: Path) -> None:
    conbo = run_conbo_method("conbo", "tiny", 0, tmp_path, FAST)
    assert all(len(u["iteration_seconds"]) == 6 for u in conbo["users"])
    grid = conbo["variance_map"]
    assert grid["points"] == 50 and len(grid["var"]) == 2500 and len(grid["mean"]) == 2500
    assert run_conbo_method("bo", "tiny", 0, tmp_path, FAST).get("variance_map") is None


def test_ablations_change_one_switch_each(tmp_path: Path) -> None:
    no_gp = run_conbo_method("conbo_no_gp", "tiny", 0, tmp_path, FAST)
    full = run_conbo_method("conbo", "tiny", 0, tmp_path, FAST)
    assert no_gp["users"][0]["x"] == full["users"][0]["x"]  # user 1 has no population yet
    assert no_gp["users"][2]["x"] != full["users"][2]["x"]
    from openbo.optimizers.conbo import ConBOConfig

    with pytest.raises(ValueError, match="memory"):
        ConBOConfig(input_dim=2, memory="bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize("memory", ["none", "direct"])
def test_bnn_memory_modes_train_without_replay(memory: str) -> None:
    from openbo.optimizers.conbo import ConBOConfig, ConBOPopulation

    population = ConBOPopulation(ConBOConfig(input_dim=2, r0=3, dr=1, memory=memory, **FAST))  # type: ignore[arg-type]
    rng = np.random.default_rng(0)
    for _ in range(2):
        opt = population.new_user()
        for _ in range(4):
            x = opt.suggest()
            opt.observe(x, np.array([float(-np.sum((x - 0.5) ** 2))]))
        report = population.finish_user(opt)
        assert report.max_variance is None and report.n_inputs_kept > 0
    assert population.bnn is not None
    mean, var = population.bnn.predict(rng.random((5, 2)))
    assert np.all(np.isfinite(mean)) and np.all(var > 0.0)


def test_revisit_test_reoptimizes_each_user_from_the_final_population(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(
        conbo_runner.CONBO_TESTS,
        "tiny_revisit",
        ConBOTest("branin", 3, 0.3, 0.2, 5, 4, 2, 50.0, revisit=True),
    )
    result = run_conbo_method("bnn_no_replay", "tiny_revisit", 0, tmp_path, FAST)
    reopt = result["reoptimization"]
    assert [r["name"] for r in reopt] == [f"user_{i:03d}" for i in range(3)]
    assert all(len(r["x"]) == 5 and r["n_random"] == 0 for r in reopt)
    with pytest.raises(ValueError, match="revisit"):
        run_conbo_method("bo", "tiny_revisit", 0, tmp_path, FAST)


def test_paper_tests_use_the_papers_settings() -> None:
    t1, t2, t3 = (conbo_runner.CONBO_TESTS[k] for k in ("branin_paper", "mccormick_paper", "branin_revisit"))
    assert (t1.alpha1, t1.alpha2, t1.n_evals, t1.taf_d2) == (16.0, 0.1, 30, 0.05)
    assert (t2.base, t2.n_users, t2.n_evals, t2.r0, t2.dr, t2.max_variance, t2.shift_range) == (
        "mccormick", 15, 7, 3, 3, 10.0, 0.5,
    )
    assert (t2.alpha1, t2.alpha2, t2.taf_d2) == (3.0, 0.15, 0.15)
    assert t3.revisit and (t3.n_users, t3.shift_range, t3.scale_range) == (10, 0.4, 0.4)
    assert conbo_runner.conbo_config(t2, 2, 0).alpha1 == 3.0


def test_mccormick_test_runs_and_has_a_finite_optimum(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(
        conbo_runner.CONBO_TESTS, "tiny_mc", ConBOTest("mccormick", 3, 0.5, 0.2, 4, 2, 1, 10.0, 3.0, 0.15)
    )
    result = run_conbo_method("taf_paper", "tiny_mc", 0, tmp_path, FAST)
    for user in result["users"]:
        assert np.isfinite(user["optimum"]) and np.max(user["y"]) <= user["optimum"] + 1e-9


def test_replay_between_user_variance_adds_the_spread_of_the_means() -> None:
    from openbo.models.conbo_replay import replay_targets
    from openbo.optimizers.conbo import gp_from_state, gp_state_dict, new_user_gp

    rng = np.random.default_rng(0)
    gps = []
    for offset in (0.0, 2.0):
        x = rng.random((8, 2))
        y = np.sin(3 * x[:, 0]) + offset
        gp = new_user_gp("matern52", 2, 0)
        gp.fit(x, y)
        gps.append(gp_from_state(gp_state_dict(gp), x, y))
    grid = rng.random((20, 2))
    base = replay_targets(gps, grid, None)
    total = replay_targets(gps, grid, None, between_users=True)
    np.testing.assert_allclose(base.mean, total.mean)
    assert np.all(total.var > base.var)
    # two users: the added term is the population variance of their two means
    means = np.stack([g.posterior(grid)[0] for g in gps])
    np.testing.assert_allclose(total.var - base.var, means.var(axis=0), rtol=1e-6, atol=1e-9)
