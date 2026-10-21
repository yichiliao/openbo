"""Tests for TAF-EIPV (acquisition/taf_eipv.py, optimizers/mobo_taf_eipv.py)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from openbo.acquisition.taf_eipv import (
    LOG_FLOOR,
    log_taf_eipv,
    reference_cells,
    source_improvement,
    taf_r_source_weights,
)
from openbo.acquisition.taf_plus import ranking_distance
from openbo.benchmarks.hypervolume import hypervolume
from openbo.optimizers.bo_taf_plus import TAFPlusPopulation
from openbo.optimizers.mobo import MOBOConfig, MOBOSequentialOptimizer
from openbo.optimizers.mobo_taf import write_population_sources
from openbo.optimizers.mobo_taf_eipv import (
    TAFEIPVConfig,
    TAFEIPVSequentialOptimizer,
    load_sources,
    run_mobo_taf_eipv,
    sources_from_population,
)
from openbo.test_functions.taf_plus_users import make_sphere_users

V_REF = np.array([-1.0, -1.0, -1.0])
W_REF = np.array([1.2, 1.2, 1.2])
POPULATION = "benchmark_results/taf_plus_sims/populations/v1_sim2_r01_s0"


@pytest.fixture(scope="module")
def population() -> TAFPlusPopulation:
    pop = TAFPlusPopulation.load(POPULATION)
    pop.users = pop.users[:3]  # keep the tests quick
    return pop


@pytest.fixture(scope="module")
def user():
    return make_sphere_users(1, shift_range=0.1, scale_range=0.1, seed=7)[0]


def config(**kwargs) -> TAFEIPVConfig:
    base = dict(bounds=[(0.0, 1.0)] * 4, v_ref=V_REF, w_ref=W_REF, n_init=0, n_candidates=128, seed=0)
    base.update(kwargs)
    return TAFEIPVConfig(**base)


# ---------------------------------------------------------------- acquisition


@pytest.mark.parametrize("n_front", [0, 1, 6])
def test_source_improvement_matches_brute_force_hypervolume(n_front) -> None:
    rng = np.random.default_rng(n_front)
    front = rng.uniform(-0.8, 1.0, size=(n_front, 3))
    points = rng.uniform(-0.9, 1.1, size=(20, 3))
    gains = source_improvement(points, reference_cells(front, V_REF, W_REF))
    for p, g in zip(points, gains):
        before = hypervolume(front, V_REF) if n_front else 0.0
        assert g == pytest.approx(hypervolume(np.vstack([front, p]), V_REF) - before, abs=1e-12)


def test_source_improvement_is_zero_at_reference_points() -> None:
    """The fading reference: designs already evaluated add nothing for that source."""
    front = np.random.default_rng(1).uniform(-0.5, 1.0, size=(5, 3))
    np.testing.assert_allclose(source_improvement(front, reference_cells(front, V_REF, W_REF)), 0.0, atol=1e-12)


def test_taf_r_weights_follow_pooled_ranking_distance() -> None:
    rng = np.random.default_rng(2)
    y = rng.normal(size=(8, 3))
    same, reversed_, noisy = y, -y, y + rng.normal(scale=1.0, size=y.shape)
    weights = taf_r_source_weights(y, np.stack([same, reversed_, noisy]))
    assert weights[0] == pytest.approx(0.75)
    assert weights[1] == pytest.approx(0.0)
    d = np.mean([ranking_distance(y[:, l], noisy[:, l]) for l in range(3)])
    assert weights[2] == pytest.approx(0.75 * (1 - d**2))
    # Fewer than two observations: no ranking evidence, full weight.
    np.testing.assert_allclose(taf_r_source_weights(y[:1], np.stack([reversed_[:1]])), 0.75)


def test_log_blend_is_the_log_of_the_weighted_average() -> None:
    rng = np.random.default_rng(3)
    target = rng.uniform(0.1, 2.0, size=5)
    terms = rng.uniform(0.0, 2.0, size=(3, 5))
    terms[1, 2] = 0.0
    weights = np.array([0.5, 0.0, 0.25])  # the second source drops out
    expected = np.log((0.75 * target + weights @ terms) / (0.75 + 0.5 + 0.25))
    np.testing.assert_allclose(log_taf_eipv(np.log(target), terms, weights, 0.75), expected, rtol=1e-12)
    # Before target data: the sources' weighted average alone.
    expected = np.log((weights @ terms) / 0.75)
    np.testing.assert_allclose(log_taf_eipv(None, terms, weights, 0.75), expected, rtol=1e-12)
    assert np.all(log_taf_eipv(None, np.zeros((1, 4)), np.array([0.75]), 0.75) == LOG_FLOOR)
    with pytest.raises(ValueError):
        log_taf_eipv(None, terms, np.zeros(3), 0.75)


# ---------------------------------------------------------------- optimizer


def test_without_sources_it_is_plain_mobo(user) -> None:
    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    kwargs = dict(bounds=[(0.0, 1.0)] * 4, v_ref=V_REF, w_ref=W_REF, n_init=3, n_candidates=128,
                  model="independent", acquisition="ieipv", seed=0)
    plain, taf = MOBOSequentialOptimizer(MOBOConfig(**kwargs)), TAFEIPVSequentialOptimizer(TAFEIPVConfig(**kwargs))
    plain.bootstrap(lambda x: user.observe(x, rng_a))
    taf.bootstrap(lambda x: user.observe(x, rng_b))
    for _ in range(2):
        xa, xb = plain.suggest(), taf.suggest()
        np.testing.assert_array_equal(xa, xb)
        plain.observe(xa, user.observe(xa, rng_a))
        taf.observe(xb, user.observe(xb, rng_b))


def test_first_point_maximizes_the_sources_predicted_hypervolume(population) -> None:
    opt = TAFEIPVSequentialOptimizer(config(), population)
    x = opt.suggest()
    sources = opt.sources
    empty = reference_cells(np.empty((0, 3)), V_REF, W_REF)

    def score(z):
        return np.sum([source_improvement(s.mean(z), empty) for s in sources], axis=0)

    others = np.random.default_rng(0).random((400, 4))
    assert score(x)[0] >= score(others).max() - 1e-9


def test_runs_from_the_sources_alone_and_records_weights(population, user) -> None:
    rng = np.random.default_rng(0)
    result = run_mobo_taf_eipv(lambda x: user.observe(x, rng), config(), n_iter=4, population=population)
    assert result.x_obs.shape == (4, 4) and result.y_obs.shape == (4, 3)
    assert result.hv_history.shape == (4,) and np.all(np.diff(result.hv_history) >= -1e-12)
    state = result.final_model_state
    assert state["sources"] == [u.name for u in population.users]
    assert len(state["last_source_weights"]) == 3


def test_decay_shrinks_the_source_weights(population, user) -> None:
    rng = np.random.default_rng(0)
    opt = TAFEIPVSequentialOptimizer(config(decay=True, d1=1, d2=0.5), population)
    for _ in range(3):
        x = opt.suggest()
        opt.observe(x, user.observe(x, rng))
    assert np.all(opt._source_weights() == 0.0)  # t = 4 > d1 + 1 / d2


def test_sources_load_from_both_directory_layouts(population, tmp_path) -> None:
    x = np.random.default_rng(4).random((6, 4))
    expected = [s.mean(x) for s in sources_from_population(population)]
    population.save(tmp_path / "taf_plus")  # {"gp_states": [...]}
    write_population_sources(population, tmp_path / "taf_ehvi")  # {"gp_state": {"objectives": [...]}}
    for layout in ("taf_plus", "taf_ehvi"):
        loaded = load_sources(tmp_path / layout)
        assert [s.name for s in loaded] == [u.name for u in population.users]
        for s, mean in zip(loaded, expected):
            np.testing.assert_allclose(s.mean(x), mean, rtol=1e-8, atol=1e-10)


def test_config_and_inputs_are_validated(population) -> None:
    with pytest.raises(ValueError):
        config(weight_mode="taf_m")
    with pytest.raises(ValueError):
        config(rho=0.0)
    with pytest.raises(ValueError):
        TAFEIPVSequentialOptimizer(config(taf_run_dir="somewhere"), population)
    with pytest.raises(ValueError):
        TAFEIPVSequentialOptimizer(TAFEIPVConfig(bounds=[(0.0, 1.0)] * 3, v_ref=V_REF, w_ref=W_REF), population)
    with pytest.raises(ValueError):
        TAFEIPVSequentialOptimizer(config()).suggest()  # no data and no sources
