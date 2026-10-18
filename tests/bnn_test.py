"""Tests for the MC-dropout population BNN used by ConBO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openbo.models.bnn import BNNConfig, PopulationBNN


def _sine_data(n: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Noisy sine observations on `[0, 0.4]`."""
    rng = np.random.default_rng(seed)
    x = rng.random((n, 1)) * 0.4
    y = np.sin(6.0 * x[:, 0]) + 0.05 * rng.standard_normal(n)
    return x, y


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        BNNConfig(input_dim=0)
    with pytest.raises(ValueError, match="dropout"):
        BNNConfig(input_dim=1, dropout=1.0)
    with pytest.raises(ValueError, match="learning_rate"):
        BNNConfig(input_dim=1, learning_rate=0.0)
    with pytest.raises(ValueError, match="batch_size"):
        BNNConfig(input_dim=1, batch_size=0)
    with pytest.raises(ValueError, match="n_predict_samples"):
        BNNConfig(input_dim=1, n_predict_samples=0)


def test_untrained_prediction_shapes_and_positive_variance() -> None:
    model = PopulationBNN(BNNConfig(input_dim=2))
    mean, var = model.predict(np.random.default_rng(0).random((7, 2)))
    assert mean.shape == (7,) and var.shape == (7,)
    assert np.all(np.isfinite(mean))
    assert np.all(var > 0.0)


def test_prediction_is_deterministic_per_call_and_per_seed() -> None:
    x, y = _sine_data()
    a = PopulationBNN(BNNConfig(input_dim=1, seed=3))
    b = PopulationBNN(BNNConfig(input_dim=1, seed=3))
    c = PopulationBNN(BNNConfig(input_dim=1, seed=4))
    for model in (a, b, c):
        model.fit_nll(x, y, epochs=20)
    np.testing.assert_array_equal(a.predict(x)[0], a.predict(x)[0])
    np.testing.assert_array_equal(a.predict(x)[0], b.predict(x)[0])
    assert not np.allclose(a.predict(x)[0], c.predict(x)[0])


def test_nll_fit_tracks_data_and_is_less_certain_away_from_it() -> None:
    x, y = _sine_data()
    model = PopulationBNN(BNNConfig(input_dim=1))
    history = model.fit_nll(x, y, epochs=800)
    assert len(history) == 800 and history[-1] < history[0]
    x_in = np.array([[0.1], [0.2], [0.3]])
    mean_in, var_in = model.predict(x_in)
    np.testing.assert_allclose(mean_in, np.sin(6.0 * x_in[:, 0]), atol=0.1)
    _, var_out = model.predict(np.array([[0.9]]))
    assert var_out[0] > 2.0 * var_in.max()


def test_nll_fit_learns_input_dependent_noise() -> None:
    rng = np.random.default_rng(1)
    x = rng.random((400, 1))
    noise = np.where(x[:, 0] > 0.5, 0.6, 0.1)
    y = noise * rng.standard_normal(400)
    model = PopulationBNN(BNNConfig(input_dim=1))
    model.fit_nll(x, y, epochs=800)
    _, var = model.predict(np.array([[0.25], [0.75]]))
    assert var[1] > 10.0 * var[0]
    np.testing.assert_allclose(var, [0.01, 0.36], rtol=0.5)


def test_replay_fit_matches_targets_in_raw_units() -> None:
    x = np.linspace(0.0, 1.0, 200)[:, None]
    mean_target = 100.0 * np.sin(6.0 * x[:, 0])
    var_target = (5.0 + 20.0 * x[:, 0]) ** 2
    model = PopulationBNN(BNNConfig(input_dim=1))
    model.fit_replay(x, mean_target, var_target, epochs=1200)
    mean, var = model.predict(x[::20])
    assert np.max(np.abs(mean - mean_target[::20])) < 5.0
    # Eq. 7 matches large variances closely and small ones loosely.
    assert var[-1] == pytest.approx(var_target[::20][-1], rel=0.3)
    assert var[-1] > 2.0 * var[0]


def test_normalization_is_fixed_after_first_fit_until_replay_renormalizes() -> None:
    x, y = _sine_data()
    model = PopulationBNN(BNNConfig(input_dim=1))
    assert not model.is_normalized
    model.fit_nll(x, y, epochs=1)
    first = (model.y_offset, model.y_scale)
    model.fit_nll(x, 1000.0 * y, epochs=1)
    assert (model.y_offset, model.y_scale) == first
    model.fit_replay(x, 1000.0 * y, np.ones_like(y), epochs=1, renormalize=False)
    assert (model.y_offset, model.y_scale) == first
    model.fit_replay(x, 1000.0 * y, np.ones_like(y), epochs=1)
    assert model.y_scale == pytest.approx(1000.0 * first[1])


def test_constant_targets_get_unit_scale() -> None:
    model = PopulationBNN(BNNConfig(input_dim=1))
    model.set_normalization(np.full(5, 2.0))
    assert (model.y_offset, model.y_scale) == (2.0, 1.0)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    x, y = _sine_data()
    model = PopulationBNN(BNNConfig(input_dim=1, seed=5))
    model.fit_nll(x, y, epochs=30)
    model.save(tmp_path / "bnn.pt")
    loaded = PopulationBNN.load(tmp_path / "bnn.pt")
    assert loaded.config == model.config
    np.testing.assert_array_equal(loaded.predict(x)[1], model.predict(x)[1])
    # Optimizer and dropout RNG state are restored too, so training continues identically.
    model.fit_nll(x, y, epochs=5)
    loaded.fit_nll(x, y, epochs=5)
    np.testing.assert_array_equal(loaded.predict(x)[0], model.predict(x)[0])


def test_reset_parameters_restores_initial_weights() -> None:
    x, y = _sine_data()
    fresh = PopulationBNN(BNNConfig(input_dim=1))
    model = PopulationBNN(BNNConfig(input_dim=1))
    model.fit_nll(x, y, epochs=30)
    model.reset_parameters()
    fresh.y_offset, fresh.y_scale = model.y_offset, model.y_scale
    np.testing.assert_array_equal(model.predict(x)[0], fresh.predict(x)[0])


def test_minibatch_training_runs() -> None:
    x, y = _sine_data()
    model = PopulationBNN(BNNConfig(input_dim=1, batch_size=16))
    history = model.fit_nll(x, y, epochs=10)
    assert len(history) == 10 and all(np.isfinite(history))


def test_input_validation() -> None:
    model = PopulationBNN(BNNConfig(input_dim=2))
    x = np.zeros((3, 2))
    with pytest.raises(ValueError, match="Expected x shape"):
        model.predict(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="y must have shape"):
        model.fit_nll(x, np.zeros(2), epochs=1)
    with pytest.raises(ValueError, match="finite"):
        model.fit_nll(x, np.array([0.0, np.nan, 1.0]), epochs=1)
    with pytest.raises(ValueError, match="non-negative"):
        model.fit_replay(x, np.zeros(3), np.array([1.0, -1.0, 1.0]), epochs=1)
    with pytest.raises(ValueError, match="epochs"):
        model.fit_nll(x, np.zeros(3), epochs=-1)
    with pytest.raises(ValueError, match="at least one"):
        model.fit_nll(np.zeros((0, 2)), np.zeros(0), epochs=1)
    with pytest.raises(ValueError, match="n_samples"):
        model.predict(x, n_samples=0)
