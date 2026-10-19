"""Tests for MetaBO's Goldstein-Price and Hartmann-3 families (NAF, more families)."""

from __future__ import annotations

import warnings

import math

import numpy as np
import pytest
from scipy.optimize import minimize

from openbo.benchmarks.naf_runner import NAF_FAMILIES, NAF_METHODS, naf_test_tasks, run_naf_task
from openbo.optimizers.naf import (
    NAF_FAMILY_SPECS,
    FamilyGPHyperparameters,
    NAFSequentialOptimizer,
    get_naf_family,
    run_naf,
    save_naf,
    train_naf,
)
from openbo.test_functions.metabo_families import (
    HARTMANN3_ARGMAX,
    GoldsteinPriceTask,
    Hartmann3Task,
    goldstein_price_grid_tasks,
    hartmann3_grid_tasks,
    normalized_goldstein_price,
    normalized_hartmann3,
    sample_goldstein_price_tasks,
    sample_hartmann3_tasks,
)

NEW_FAMILIES = ("goldstein_price", "hartmann3")


def test_goldstein_price_matches_the_textbook_minimum_and_the_reference_normalization() -> None:
    # Textbook: minimum 3 at (0, -1) on [-2, 2]^2, which is (0.5, 0.25) on the unit square.
    value = normalized_goldstein_price(np.array([[0.5, 0.25]]))[0]
    assert value == pytest.approx(-(math.log(3.0) - 8.693) / 2.427)
    # A second textbook point: f(0, 0) = 600 at (0.5, 0.5).
    assert normalized_goldstein_price(np.array([[0.5, 0.5]]))[0] == pytest.approx(-(math.log(600.0) - 8.693) / 2.427)


def test_hartmann3_matches_the_textbook_maximum_and_the_reference_normalization() -> None:
    # Textbook: minimum -3.86278 at (0.114614, 0.555649, 0.852547) (rounded).
    value = normalized_hartmann3(np.array([HARTMANN3_ARGMAX]))[0]
    assert value == pytest.approx((3.86278 + (-0.93)) / 0.95, abs=1e-5)
    assert normalized_hartmann3(np.array([[0.114614, 0.555649, 0.852547]]))[0] == pytest.approx(value, abs=1e-8)


@pytest.mark.parametrize("function, dim", [(normalized_goldstein_price, 2), (normalized_hartmann3, 3)])
def test_base_functions_are_normalized_to_about_unit_scale(function, dim: int) -> None:
    values = function(np.random.default_rng(0).random((20000, dim)))
    assert abs(values.mean()) < 0.1 and 0.9 < values.std() < 1.1


@pytest.mark.parametrize(
    "task",
    sample_goldstein_price_tasks(4, seed=3) + goldstein_price_grid_tasks()[:3]
    + sample_hartmann3_tasks(4, seed=3) + hartmann3_grid_tasks()[:3],
)
def test_tracked_optimum_is_the_maximum_in_the_domain(task: GoldsteinPriceTask | Hartmann3Task) -> None:
    dim = len(task.translation)
    points = np.random.default_rng(0).random((30000, dim))
    values = task(points)
    best = float(values.max())
    for start in points[np.argsort(-values)[:5]]:
        result = minimize(lambda z: -task(z)[0], start, bounds=[(0.0, 1.0)] * dim)
        best = max(best, -result.fun)
    assert best <= task.optimum + 1e-9
    assert task(task.argmax)[0] == pytest.approx(task.optimum)
    assert np.all((task.argmax >= 0.0) & (task.argmax <= 1.0))


@pytest.mark.parametrize(
    "sample, grid, dim",
    [(sample_goldstein_price_tasks, goldstein_price_grid_tasks, 2), (sample_hartmann3_tasks, hartmann3_grid_tasks, 3)],
)
def test_tasks_stay_in_range_and_are_reproducible(sample, grid, dim: int) -> None:
    tasks = sample(100, 0) + grid(50)
    t = np.array([task.translation for task in tasks])
    s = np.array([task.scaling for task in tasks])
    assert t.shape == (150, dim)
    assert np.all(np.abs(t) <= 0.1) and np.all((s >= 0.9) & (s <= 1.1))
    assert sample(3, 0) == tasks[:3]
    assert len(set(grid(50))) == 50 and grid(50) == grid(50)
    with pytest.raises(ValueError, match="n_tasks"):
        sample(0, 0)
    with pytest.raises(ValueError, match="n_tasks"):
        grid(0)


def test_each_family_has_its_own_candidate_set_and_gp() -> None:
    assert set(NAF_FAMILIES) == set(NAF_FAMILY_SPECS) == {"branin", "goldstein_price", "hartmann3"}
    # The reference's rule: floor(N_MS ** (1 / D)) points per dimension, N_LS = N_MS, k = 5.
    assert get_naf_family("goldstein_price").config().adaptive_set().size == 31**2 + 5
    assert get_naf_family("hartmann3").config().adaptive_set().size == 12**3 + 5
    assert get_naf_family("hartmann3").config().n_local == 2000
    for name, spec in NAF_FAMILY_SPECS.items():
        assert len(spec.gp_hyperparameters.lengthscale) == spec.dim and spec.config().dim == spec.dim
        assert len(spec.sample_tasks(1, 0)[0].translation) == spec.dim
        assert FamilyGPHyperparameters.from_dict(spec.gp_hyperparameters.to_dict()) == spec.gp_hyperparameters
    with pytest.raises(ValueError, match="Unknown family"):
        get_naf_family("nope")


@pytest.mark.parametrize("family", NEW_FAMILIES)
def test_family_gp_beats_its_prior_on_a_task(family: str) -> None:
    """The fixed hyperparameters must describe the family: 40 points predict held-out values far better than 0."""
    spec = get_naf_family(family)
    task = spec.sample_tasks(1, 7)[0]
    rng = np.random.default_rng(1)
    x, x_test = rng.random((40, spec.dim)), rng.random((300, spec.dim))
    gp = spec.gp_hyperparameters.new_gp()
    gp.fit(x, task(x))
    mean, var = gp.posterior(x_test)
    error = np.sqrt(np.mean((mean - task(x_test)) ** 2))
    assert error < 0.8 * np.sqrt(np.mean(task(x_test) ** 2))
    assert np.all(var >= -1e-9) and np.all(var <= spec.gp_hyperparameters.variance + 1e-9)


@pytest.fixture(scope="module", params=NEW_FAMILIES)
def family_and_af_dir(request, tmp_path_factory):
    """An untrained AF of the family with a small candidate set, saved like a training run."""
    spec = get_naf_family(request.param)
    config = spec.config(budget=3, hidden=(16, 16), points_per_dim=5, n_local=20)
    directory = tmp_path_factory.mktemp(request.param)
    save_naf(directory, config.new_af(), config, spec.gp_hyperparameters, name="best_af")
    return request.param, str(directory)


@pytest.mark.parametrize("method", NAF_METHODS)
def test_every_method_runs_on_the_new_families(method: str, family_and_af_dir) -> None:
    family, af_dir = family_and_af_dir
    task = naf_test_tasks(family, n_tasks=1, seed=100)[0]
    dim = get_naf_family(family).dim
    record = run_naf_task(method, task, 0, family=family, af_dir=af_dir, budget=3)
    x, y, regret = np.asarray(record["x"]), np.asarray(record["y"]), np.asarray(record["regret"])
    assert x.shape == (3, dim) and y.shape == (3,)
    np.testing.assert_allclose(y, task(x))
    np.testing.assert_allclose(regret, task.optimum - np.maximum.accumulate(y))
    assert np.all((x >= 0.0) & (x <= 1.0))


def test_optimizer_runs_a_three_dimensional_family(family_and_af_dir) -> None:
    family, af_dir = family_and_af_dir
    task = naf_test_tasks(family, n_tasks=1, seed=100)[0]
    optimizer = NAFSequentialOptimizer.from_directory(af_dir)
    x, y = run_naf(task, optimizer)
    assert x.shape == (3, get_naf_family(family).dim) and optimizer.t == 3
    np.testing.assert_allclose(y, task(x))


def test_training_runs_on_hartmann3() -> None:
    """One tiny PPO iteration on the 3D family: the AF input, grid and GP all follow the family's dimension."""
    spec = get_naf_family("hartmann3")
    config = spec.config(
        budget=2, hidden=(8, 8), points_per_dim=4, n_local=10, n_iterations=1, steps_per_iteration=4, minibatch_size=2
    )
    result = train_naf(config, spec.gp_hyperparameters, task_pool=spec.grid_tasks(4))
    assert len(result.mean_return) == 1 and np.isfinite(result.mean_return[0])
    assert result.af.dim == 3


def _small_af_dir(tmp_path_factory, family: str, gp_hyperparameters=None) -> str:
    spec = get_naf_family(family)
    config = spec.config(budget=3, hidden=(8, 8), points_per_dim=4, n_local=10)
    directory = tmp_path_factory.mktemp(f"af_{family}")
    save_naf(directory, config.new_af(), config, gp_hyperparameters or spec.gp_hyperparameters, name="best_af")
    return str(directory)


@pytest.fixture(scope="module")
def small_afs(tmp_path_factory) -> dict[str, str]:
    return {family: _small_af_dir(tmp_path_factory, family) for family in ("branin", "goldstein_price", "hartmann3")}


def test_benchmark_accepts_an_af_of_the_right_family_without_warnings(small_afs) -> None:

    from openbo.benchmarks.naf_runner import run_naf_method

    for family, af_dir in small_afs.items():
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warnings.filterwarnings("ignore", message="The balance properties")
            result = run_naf_method("naf", family, n_tasks=1, af_dir=af_dir, budget=3)
        assert result["family"] == family and len(result["tasks"]) == 1


def test_benchmark_refuses_an_af_of_another_dimension_with_a_clear_message(small_afs) -> None:
    from openbo.benchmarks.naf_runner import run_naf_method

    with pytest.raises(ValueError, match=r"takes 3D inputs, but family 'branin' is 2D.*'hartmann3'"):
        run_naf_method("naf", "branin", n_tasks=1, af_dir=small_afs["hartmann3"], budget=3)
    with pytest.raises(ValueError, match=r"takes 2D inputs, but family 'hartmann3' is 3D.*'branin'"):
        run_naf_method("naf", "hartmann3", n_tasks=1, af_dir=small_afs["branin"], budget=3)


def test_benchmark_warns_when_a_same_dimension_af_belongs_to_another_family(small_afs) -> None:
    """A Goldstein-Price AF is 2D like Branin's, so only its GP hyperparameters give it away."""
    from openbo.benchmarks.naf_runner import run_naf_method

    with pytest.warns(UserWarning, match=r"not trained with the GP hyperparameters of family 'branin'.*Its GP hyperparameters are those of 'goldstein_price'"):
        run_naf_method("naf", "branin", n_tasks=1, af_dir=small_afs["goldstein_price"], budget=3)


def test_benchmark_only_warns_for_an_af_with_hyperparameters_of_its_own(tmp_path_factory) -> None:

    from openbo.benchmarks.naf_runner import run_naf_method

    own = FamilyGPHyperparameters(lengthscale=(0.5, 0.5), variance=1.0)
    af_dir = _small_af_dir(tmp_path_factory, "branin", own)
    with pytest.warns(UserWarning, match="not trained with the GP hyperparameters of family 'branin'") as record:
        run_naf_method("naf", "branin", n_tasks=1, af_dir=af_dir, budget=3)
    assert "Its GP hyperparameters are those of" not in str(record[0].message)


def test_the_check_applies_only_to_the_naf_method(small_afs) -> None:
    from openbo.benchmarks.naf_runner import run_naf_method

    # ei and random do not use the AF, so a mismatching af_dir is ignored.
    assert len(run_naf_method("ei", "branin", n_tasks=1, af_dir=small_afs["hartmann3"], budget=2)["tasks"]) == 1
