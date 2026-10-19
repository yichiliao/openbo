"""Tests for the ConBO population and per-user optimizer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from openbo.optimizers.bo_taf import _load_source_surrogates
from openbo.optimizers.conbo import (
    ConBOConfig,
    argmax_on_grid,
    ConBOPopulation,
    ConBOSequentialOptimizer,
    run_conbo_sequence,
)
from openbo.test_functions.families import generate_user_variants, make_user_sequence

# Small settings so the tests stay fast; the paper's values are the defaults.
# The replay filter is off so that tiny users always train the population model.
FAST = ConBOConfig(
    input_dim=2,
    r0=6,
    dr=2,
    candidate_grid_points=10,
    replay_grid_points=8,
    replay_random_points=5,
    population_epochs=40,
    online_epochs=2,
    hidden_units=16,
    n_predict_samples=8,
    replay_filter=False,
)


def _users(n: int, seed: int = 0) -> list:
    return make_user_sequence("branin", generate_user_variants("branin", n, 0.3, 0.2, seed=seed))


def _run_user(opt: ConBOSequentialOptimizer, objective, n: int) -> None:  # type: ignore[no-untyped-def]
    for _ in range(n):
        x = opt.suggest()
        opt.observe(x, objective(x))


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        ConBOConfig(input_dim=0)
    with pytest.raises(ValueError, match="r0"):
        ConBOConfig(input_dim=2, r0=-1)
    with pytest.raises(ValueError, match="retrain"):
        ConBOConfig(input_dim=2, retrain="sometimes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="weight_clock"):
        ConBOConfig(input_dim=2, weight_clock="wall")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="replay_max_variance"):
        ConBOConfig(input_dim=2, replay_max_variance=0.0)
    with pytest.raises(ValueError, match="alpha2"):
        ConBOConfig(input_dim=2, alpha2=0.0)


def test_random_starts_decay_per_user_and_first_user_gets_at_least_one() -> None:
    users = _users(5)
    population = ConBOPopulation(FAST)
    counts = []
    for user in users:
        opt = population.new_user()
        counts.append(opt.n_random)
        _run_user(opt, user.objective, 7)
        population.finish_user(opt)
    assert counts == [6, 4, 2, 0, 0]
    assert ConBOPopulation(replace(FAST, r0=0)).new_user().n_random == 1


def test_first_user_is_plain_bo_and_later_users_follow_the_schedule() -> None:
    users = _users(2)
    population = ConBOPopulation(FAST)
    first = population.new_user()
    _run_user(first, users[0].objective, 8)
    assert first.weight_history == [None] * 6 + [0.0, 0.0]
    population.finish_user(first)

    second = population.new_user()  # 4 random starts
    _run_user(second, users[1].objective, 10)
    assert second.weight_history[:4] == [None] * 4
    np.testing.assert_allclose(second.weight_history[4:], [1.0, 0.8, 0.6, 0.4, 0.2, 0.0])


def test_bo_steps_clock_starts_the_schedule_after_random_starts() -> None:
    config = replace(FAST, weight_clock="bo_steps")
    users = _users(2)
    population = ConBOPopulation(config)
    first = population.new_user()
    _run_user(first, users[0].objective, 6)
    population.finish_user(first)
    second = population.new_user()
    _run_user(second, users[1].objective, 11)
    np.testing.assert_allclose(second.weight_history[4:], [1, 1, 1, 1, 1, 0.8, 0.6])


def test_without_user_gp_the_population_keeps_full_weight() -> None:
    config = replace(FAST, use_user_gp=False)
    users = _users(2)
    population = ConBOPopulation(config)
    first = population.new_user()
    _run_user(first, users[0].objective, 6)
    population.finish_user(first)
    second = population.new_user()
    _run_user(second, users[1].objective, 9)
    assert second.weight_history[4:] == [1.0] * 5


def test_without_refinement_bo_suggestions_come_from_the_candidate_grid() -> None:
    population = ConBOPopulation(replace(FAST, refine_suggestion=False))
    opt = population.new_user()
    _run_user(opt, _users(1)[0].objective, 9)
    for x in opt.x_obs[6:]:
        assert any(np.array_equal(x, c) for c in opt.candidates)
    assert np.all((opt.x_obs >= 0.0) & (opt.x_obs <= 1.0))


def test_refinement_polishes_the_best_grid_point() -> None:
    target = np.array([0.537, 0.281])

    def acquisition(x: np.ndarray) -> np.ndarray:
        return np.exp(-np.sum((x - target) ** 2, axis=1) / 0.02)

    grid = np.stack(np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10)), -1).reshape(-1, 2)
    on_grid = argmax_on_grid(acquisition, grid, refine=False)
    refined = argmax_on_grid(acquisition, grid, refine=True)
    assert any(np.array_equal(on_grid[0], c) for c in grid)
    assert refined.shape == (1, 2)
    np.testing.assert_allclose(refined[0], target, atol=1e-4)
    assert acquisition(refined)[0] > acquisition(on_grid)[0]


def test_refinement_keeps_the_grid_point_when_acquisition_is_flat() -> None:
    grid = np.random.default_rng(0).random((20, 2))
    x = argmax_on_grid(lambda z: np.zeros(z.shape[0]), grid, refine=True)
    np.testing.assert_array_equal(x[0], grid[0])


def test_refined_suggestions_stay_in_bounds() -> None:
    population = ConBOPopulation(FAST)
    opt = population.new_user()
    _run_user(opt, _users(1)[0].objective, 9)
    assert np.all((opt.x_obs >= 0.0) & (opt.x_obs <= 1.0))


def test_abandoned_session_leaves_the_population_model_unchanged() -> None:
    users = _users(2)
    population = ConBOPopulation(FAST)
    first = population.new_user()
    _run_user(first, users[0].objective, 6)
    population.finish_user(first)
    probe = np.random.default_rng(0).random((5, 2))
    before = population.bnn.predict(probe)[0]  # type: ignore[union-attr]
    abandoned = population.new_user()
    _run_user(abandoned, users[1].objective, 8)
    np.testing.assert_array_equal(population.bnn.predict(probe)[0], before)  # type: ignore[union-attr]
    assert not np.array_equal(abandoned.bnn.predict(probe)[0], before)  # type: ignore[union-attr]
    assert population.n_users == 1


def test_memory_is_saved_in_taf_layout_and_reloads(tmp_path: Path) -> None:
    users = _users(2)
    result = run_conbo_sequence([u.objective for u in users], FAST, 7, memory_dir=tmp_path)
    assert sorted(p.name for p in (tmp_path / "gp_states").iterdir()) == [
        "user_000.json",
        "user_001.json",
    ]
    assert (tmp_path / "population_bnn.pt").exists()
    # TAF can use a ConBO memory directory as its source directory.
    assert [s.name for s in _load_source_surrogates(tmp_path)] == ["user_000", "user_001"]

    loaded = ConBOPopulation.load(tmp_path)
    assert loaded.n_users == 2 and loaded.config == FAST
    np.testing.assert_array_equal(loaded.users[1].y, result.users[1].y_obs)
    assert len(loaded.replay_reports) == 2
    probe = np.random.default_rng(1).random((4, 2))
    original = ConBOPopulation(FAST)
    for user in users:
        opt = original.new_user()
        _run_user(opt, user.objective, 7)
        original.finish_user(opt)
    np.testing.assert_array_equal(loaded.bnn.predict(probe)[0], original.bnn.predict(probe)[0])  # type: ignore[union-attr]
    assert loaded.new_user().user_index == 3


def test_runs_are_reproducible() -> None:
    objectives = [u.objective for u in _users(2)]
    a = run_conbo_sequence(objectives, FAST, 8)
    b = run_conbo_sequence(objectives, FAST, 8)
    for ua, ub in zip(a.users, b.users):
        np.testing.assert_array_equal(ua.x_obs, ub.x_obs)


def test_run_result_contents() -> None:
    users = _users(3)
    result = run_conbo_sequence(
        [u.objective for u in users], FAST, 7, optima=[u.optimum for u in users]
    )
    assert [u.name for u in result.users] == ["user_000", "user_001", "user_002"]
    assert [u.n_random for u in result.users] == [6, 4, 2]
    for user, spec in zip(result.users, users):
        assert user.x_obs.shape == (7, 2)
        assert np.all(np.diff(user.best_so_far) >= 0.0)
        assert user.optimum == spec.optimum
        assert np.all(user.best_so_far <= spec.optimum + 1e-9)
        assert user.replay.n_users == int(user.name[-3:]) + 1


def test_replay_threshold_options() -> None:
    users = _users(1)
    for config, expected in (
        (FAST, None),
        (replace(FAST, replay_filter=True, replay_max_variance=50.0), 50.0),
    ):
        population = ConBOPopulation(config)
        opt = population.new_user()
        _run_user(opt, users[0].objective, 6)
        assert population.finish_user(opt).max_variance == expected


def test_strict_filter_leaves_no_population_model() -> None:
    config = replace(FAST, replay_filter=True, replay_max_variance=1e-9)
    users = _users(2)
    population = ConBOPopulation(config)
    first = population.new_user()
    _run_user(first, users[0].objective, 6)
    report = population.finish_user(first)
    assert report.n_inputs_kept == 0 and report.excluded_users == [0]
    assert population.bnn is None
    second = population.new_user()
    assert second.n_random == 4 and second.current_weight() == 0.0


def test_retrain_modes_differ() -> None:
    objectives = [u.objective for u in _users(2)]
    probe = np.random.default_rng(2).random((4, 2))
    preds = []
    for mode in ("reset", "continue"):
        population = ConBOPopulation(replace(FAST, retrain=mode))
        for objective in objectives:
            opt = population.new_user()
            _run_user(opt, objective, 6)
            population.finish_user(opt)
        preds.append(population.bnn.predict(probe)[0])  # type: ignore[union-attr]
    assert not np.allclose(preds[0], preds[1])


def test_population_validation() -> None:
    population = ConBOPopulation(FAST)
    opt = population.new_user()
    with pytest.raises(ValueError, match="without observations"):
        population.finish_user(opt)
    with pytest.raises(ValueError, match="x_new must have shape"):
        opt.observe(np.zeros((1, 3)), np.zeros(1))
    with pytest.raises(ValueError, match="finite"):
        opt.observe(np.zeros((1, 2)), np.array([np.inf]))
    _run_user(opt, _users(1)[0].objective, 2)
    late = population.new_user()
    _run_user(late, _users(1)[0].objective, 1)
    population.finish_user(opt)
    with pytest.raises(ValueError, match="in the order"):
        population.finish_user(late)
    with pytest.raises(ValueError, match="already in memory"):
        population.new_user(name="user_000")
    with pytest.raises(ValueError, match="No memory_dir"):
        population.save()
    with pytest.raises(ValueError, match="n_iter_per_user"):
        run_conbo_sequence([], FAST, 0)
