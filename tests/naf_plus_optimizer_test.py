"""Tests for the NAF+ optimizer, its training entry point and the benchmark runner (NAF+, Phase 4)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from openbo.acquisition.naf import naf_score_fn
from openbo.acquisition.naf_plus import CALIBRATED_TAU, ei_score_fn, mixed_score_fn
from openbo.benchmarks.naf_plus_runner import (
    NAF_PLUS_METHODS,
    naf_plus_test_tasks,
    run_naf_plus_method,
    run_naf_plus_task,
)
from openbo.optimizers.naf import NAFConfig, NAFSequentialOptimizer, load_naf
from openbo.optimizers.naf_plus import (
    DETECTOR_FILE,
    NAFPlusConfig,
    NAFPlusSequentialOptimizer,
    NoveltyCalibration,
    calibrate_novelty,
    run_naf_plus,
    train_naf_plus,
)
from openbo.optimizers.naf_plus import DOUBLE_SPHERE_GP_HYPERPARAMETERS, DOUBLE_SPHERE_WIDE_GP_HYPERPARAMETERS
from openbo.test_functions.naf_plus_families import (
    NOVEL_CENTRES,
    TRAINING_CENTRES,
    WIDE_NOVEL_CENTRES,
    WIDE_TRAINING_CENTRES,
)

BUDGET = 3
PLUS = NAFPlusConfig(detector_epochs=5, detector_users=2, detector_points_per_dim=4, detector_hidden_units=16)


def _naf_config(n_weights: int = 2, **overrides: object) -> NAFConfig:
    settings: dict[str, object] = dict(
        dim=2,
        budget=BUDGET,
        hidden=(16,),
        points_per_dim=5,
        n_local=50,
        n_iterations=1,
        steps_per_iteration=60,
        minibatch_size=20,
        seed=0,
        n_weights=n_weights,
    )
    settings.update(overrides)
    return NAFConfig(**settings)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Two tiny training runs: NAF+ (with weights) and the ablation without weights."""
    root = tmp_path_factory.mktemp("naf_plus_runs")
    train_naf_plus(_naf_config(2), PLUS, root / "with_w")
    result = train_naf_plus(_naf_config(0), PLUS, root / "no_w", detector_from=root / "with_w")
    assert result.detector_seconds == 0.0  # copied, not fitted
    calibrate_novelty(root / "with_w", n_users=6).save(root / "with_w")
    return root / "with_w", root / "no_w"


@dataclass
class StubDetector:
    """A detector that gives every observation the same p-value."""

    p: float
    dim: int = 2
    n_weights: int = 2

    z: float = 0.0

    def p_values(self, x: np.ndarray, weights: object, y: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(y).reshape(-1).shape, self.p)

    def z_scores(self, x: np.ndarray, weights: object, y: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(y).reshape(-1).shape, self.z)


def _optimizer(run: Path, detector: object | None, weights: tuple[float, float] = (0.3, 0.7)) -> NAFPlusSequentialOptimizer:
    af, config, hp = load_naf(run, "best_af")
    return NAFPlusSequentialOptimizer(af, hp, config.budget, config.adaptive_set(), weights, detector)  # type: ignore[arg-type]


def test_without_a_detector_naf_plus_is_the_neural_af(runs: tuple[Path, Path]) -> None:
    task = naf_plus_test_tasks("double_sphere", PLUS, 1)[0]
    plus = NAFPlusSequentialOptimizer.from_run(runs[0], task.weights, use_fallback=False)
    assert plus.detector is None
    naf = NAFSequentialOptimizer.from_directory(runs[0], weights=task.weights)
    x_plus, _ = run_naf_plus(task, plus)
    x_naf, _ = run_naf_plus(task, naf)  # type: ignore[arg-type]
    np.testing.assert_array_equal(x_plus, x_naf)
    assert [h.lambda_ei for h in plus.history] == [0.0] * BUDGET


def test_a_typical_user_keeps_the_neural_af(runs: tuple[Path, Path]) -> None:
    task = naf_plus_test_tasks("double_sphere", PLUS, 1)[0]
    plus = _optimizer(runs[0], StubDetector(p=0.5), task.weights)
    x_plus, _ = run_naf_plus(task, plus)
    x_naf, _ = run_naf_plus(task, NAFSequentialOptimizer.from_directory(runs[0], weights=task.weights))  # type: ignore[arg-type]
    np.testing.assert_array_equal(x_plus, x_naf)
    assert plus.p_bar == 0.5


def test_a_novel_user_hands_over_to_ei_after_the_first_iterations(runs: tuple[Path, Path]) -> None:
    task = naf_plus_test_tasks("double_sphere", PLUS, 1)[0]
    plus = _optimizer(runs[0], StubDetector(p=0.0), task.weights)
    plus.naf_only_iterations = 2
    for _ in range(2):
        x = plus.suggest()
        plus.observe(x, task(x))
    assert [h.lambda_ei for h in plus.history[:2]] == [0.0, 0.0]
    # From the third iteration on, lambda = 1: the suggestion is EI's argmax.
    expected = plus.adaptive_set.maximize(ei_score_fn(plus.gp, float(np.max(plus.y_obs))))
    np.testing.assert_array_equal(plus.suggest(), expected)
    assert plus.history[-1].lambda_ei == 1.0 and plus.history[-1].p_bar == 0.0


def test_an_intermediate_p_bar_mixes_the_scores(runs: tuple[Path, Path]) -> None:
    task = naf_plus_test_tasks("double_sphere", PLUS, 1)[0]
    plus = _optimizer(runs[0], StubDetector(p=0.05), task.weights)
    plus.naf_only_iterations = 1
    x = plus.suggest()
    plus.observe(x, task(x))
    assert plus.current_ei_weight() == pytest.approx(0.5)
    naf = naf_score_fn(plus.af, plus.gp, plus.t, plus.budget, plus.weights)
    ei = ei_score_fn(plus.gp, float(np.max(plus.y_obs)))
    expected = plus.adaptive_set.maximize(mixed_score_fn(naf, ei, 0.5, plus.adaptive_set.global_grid))
    np.testing.assert_array_equal(plus.suggest(), expected)


def test_a_sampled_af_draws_from_its_softmax_and_the_fallback_stays_greedy(runs: tuple[Path, Path]) -> None:
    task = naf_plus_test_tasks("double_sphere", PLUS, 1)[0]

    def sampled(seed: int, p: float) -> NAFPlusSequentialOptimizer:
        plus = _optimizer(runs[0], StubDetector(p=p), task.weights)
        plus.rng = np.random.default_rng(seed)
        return plus

    first = [sampled(seed, 0.5).suggest() for seed in (0, 0, 1, 2, 3)]
    np.testing.assert_array_equal(first[0], first[1])  # seeded
    assert len({tuple(x.ravel()) for x in first}) > 1  # not the argmax every time
    plus = sampled(0, 0.0)
    plus.naf_only_iterations = 1
    x = plus.suggest()
    plus.observe(x, task(x))
    expected = plus.adaptive_set.maximize(ei_score_fn(plus.gp, float(np.max(plus.y_obs))))
    np.testing.assert_array_equal(plus.suggest(), expected)  # lambda = 1: EI's argmax, no sampling


def test_the_real_detector_scores_every_observation(runs: tuple[Path, Path]) -> None:
    task = naf_plus_test_tasks("double_sphere", PLUS, 1)[0]
    plus = NAFPlusSequentialOptimizer.from_run(runs[0], task.weights)
    run_naf_plus(task, plus)
    assert len(plus.p_values) == BUDGET and all(0.0 <= p <= 1.0 for p in plus.p_values)
    assert [h.t for h in plus.history] == list(range(BUDGET))
    with pytest.raises(RuntimeError, match="budget"):
        plus.suggest()
    with pytest.raises(ValueError, match="sum to 1"):
        NAFPlusSequentialOptimizer.from_run(runs[0], (0.5, 0.6))


def test_training_writes_the_run_and_checks_its_settings(runs: tuple[Path, Path]) -> None:
    with_w, no_w = runs
    for name in ("best_af.pt", "naf_config.json", "naf_plus_config.json", DETECTOR_FILE, "detector.json"):
        assert (with_w / name).exists() and (no_w / name).exists()
    assert NAFPlusConfig.load(with_w) == PLUS
    assert load_naf(with_w)[0].n_weights == 2 and load_naf(no_w)[0].n_weights == 0
    # Resuming keeps the detector and continues the AF.
    resumed = train_naf_plus(_naf_config(2, n_iterations=2), PLUS, with_w, resume=True)
    assert len(resumed.naf.mean_return) == 2 and resumed.detector_seconds == 0.0
    with pytest.raises(ValueError, match="different NAF\\+ settings"):
        train_naf_plus(_naf_config(2), replace(PLUS, tau=0.2), with_w)
    with pytest.raises(ValueError, match="n_weights"):
        train_naf_plus(_naf_config(3), PLUS, with_w)


def test_config_validates_and_draws_the_right_users() -> None:
    with pytest.raises(ValueError, match="family"):
        NAFPlusConfig(family="branin")
    with pytest.raises(ValueError, match="tau"):
        NAFPlusConfig(tau=0.0)
    assert all(u.centres == NOVEL_CENTRES for u in PLUS.users(3, 0, novel=True))
    tasks = naf_plus_test_tasks("double_sphere", PLUS, 4, seed=100)
    assert all(t.user.centres == TRAINING_CENTRES for t in tasks)
    assert naf_plus_test_tasks("double_sphere_novel", PLUS, 4, seed=100)[0].user.centres == NOVEL_CENTRES
    with pytest.raises(ValueError, match="test"):
        naf_plus_test_tasks("branin")
    assert PLUS.gp_hyperparameters == DOUBLE_SPHERE_GP_HYPERPARAMETERS
    wide = replace(PLUS, family="double_sphere_wide")
    assert wide.gp_hyperparameters == DOUBLE_SPHERE_WIDE_GP_HYPERPARAMETERS
    assert all(t.user.centres == WIDE_TRAINING_CENTRES for t in naf_plus_test_tasks("double_sphere", wide, 4))
    assert naf_plus_test_tasks("double_sphere_novel", wide, 4)[0].user.centres == WIDE_NOVEL_CENTRES


def test_every_benchmark_method_runs(runs: tuple[Path, Path]) -> None:
    with_w, no_w = runs
    for method in NAF_PLUS_METHODS:
        result = run_naf_plus_method(method, "double_sphere_novel", 2, with_w, no_w, budget=BUDGET)
        assert result["naf_plus_config"]["detector_epochs"] == PLUS.detector_epochs
        for record in result["tasks"]:  # type: ignore[union-attr]
            regret = np.asarray(record["regret"])
            assert regret.shape == (BUDGET,) and np.all(np.diff(regret) <= 0) and regret[-1] >= -1e-12
            assert ("history" in record) == method.startswith("naf_plus")


def test_avoid_repeats_never_suggests_an_evaluated_point(runs: tuple[Path, Path]) -> None:
    task = naf_plus_test_tasks("double_sphere", PLUS, 1)[0]
    plain = NAFPlusSequentialOptimizer.from_run(runs[0], task.weights)
    norepeat = NAFPlusSequentialOptimizer.from_run(runs[0], task.weights, avoid_repeats=True)
    assert np.array_equal(plain.suggest(), norepeat.suggest())  # nothing evaluated yet
    # Feed both the same point several times: the plain AF may repeat it, the other must not.
    x = plain.suggest()
    for opt in (plain, norepeat):
        opt.observe(x, task(x))
    assert not np.allclose(norepeat.suggest(), x)
    result = run_naf_plus_method("naf_plus_norepeat", "double_sphere", 3, runs[0], budget=BUDGET)
    for record in result["tasks"]:  # type: ignore[union-attr]
        points = np.asarray(record["x"])
        assert np.unique(points, axis=0).shape[0] == points.shape[0]


def test_parallel_runs_match_serial_ones(runs: tuple[Path, Path]) -> None:
    for method in ("naf_plus", "taf_plus_15"):
        serial = run_naf_plus_method(method, "double_sphere", 2, runs[0], budget=BUDGET)
        parallel = run_naf_plus_method(method, "double_sphere", 2, runs[0], budget=BUDGET, n_workers=2)
        assert serial["tasks"] == parallel["tasks"]


def test_runner_checks_the_run_directories(runs: tuple[Path, Path]) -> None:
    task = naf_plus_test_tasks("double_sphere", PLUS, 1)[0]
    with pytest.raises(ValueError, match="needs run_dir_no_w"):
        run_naf_plus_task("naf_plus_no_w", task, 0, runs[0], None, BUDGET)
    with pytest.raises(ValueError, match="does not hold"):
        run_naf_plus_task("naf_plus_no_w", task, 0, runs[0], runs[0], BUDGET)
    with pytest.raises(ValueError, match="budget"):
        run_naf_plus_task("naf_plus", task, 0, runs[0], None, BUDGET + 1)


# The calibrated test and the policy mix (plan, "Design stance", items 2-3).


def _null(n_users: int = 99) -> NoveltyCalibration:
    """A calibration whose null S_n is 0, 1, ..., n_users - 1 at every n and policy."""
    rows = [list(map(float, range(n_users)))] * BUDGET
    return NoveltyCalibration({"greedy": rows, "sample": rows}, n_users, 0)


def test_calibration_records_s_n_per_policy_and_round_trips(runs: tuple[Path, Path], tmp_path: Path) -> None:
    calibration = NoveltyCalibration.load(runs[0])
    assert calibration.n_users == 6 and set(calibration.statistics) == {"greedy", "sample"}
    for per_n in calibration.statistics.values():
        values = np.asarray(per_n)
        assert values.shape == (BUDGET, 6)
        assert np.all(values >= 0.0) and np.all(np.diff(values, axis=0) >= 0.0)  # S_n only grows
    calibration.save(tmp_path)
    assert NoveltyCalibration.load(tmp_path) == calibration
    with pytest.raises(FileNotFoundError, match="calibrate_naf_plus_novelty"):
        NoveltyCalibration.load(tmp_path / "missing")
    with pytest.raises(ValueError, match="n must be"):
        calibration.p_value("greedy", 1.0, BUDGET + 1)
    with pytest.raises(ValueError, match="policy"):
        calibration.p_value("other", 1.0, 1)


def test_a_typical_user_passes_the_calibrated_test(runs: tuple[Path, Path]) -> None:
    task = naf_plus_test_tasks("double_sphere", PLUS, 1)[0]
    af, config, hp = load_naf(runs[0], "best_af")
    plus = NAFPlusSequentialOptimizer(
        af, hp, config.budget, config.adaptive_set(), task.weights, StubDetector(p=0.0, z=0.0),  # type: ignore[arg-type]
        tau=CALIBRATED_TAU, naf_only_iterations=1, novelty_test="calibrated", calibration=_null(),
        mix="policy", mix_rng=np.random.default_rng(0),
    )
    x_plus, _ = run_naf_plus(task, plus)
    x_naf, _ = run_naf_plus(task, NAFSequentialOptimizer.from_directory(runs[0], weights=task.weights))  # type: ignore[arg-type]
    np.testing.assert_array_equal(x_plus, x_naf)  # p_bar = 0 would flag it; the calibrated test does not
    assert plus.novelty_p == 1.0 and [h.source for h in plus.history] == ["af"] * BUDGET


def test_an_extreme_user_gets_the_calibrated_ei_weight(runs: tuple[Path, Path]) -> None:
    task = naf_plus_test_tasks("double_sphere", PLUS, 1)[0]
    af, config, hp = load_naf(runs[0], "best_af")
    plus = NAFPlusSequentialOptimizer(
        af, hp, config.budget, config.adaptive_set(), task.weights, StubDetector(p=0.5, z=100.0),  # type: ignore[arg-type]
        tau=CALIBRATED_TAU, naf_only_iterations=1, novelty_test="calibrated", calibration=_null(),
        mix="policy", mix_rng=np.random.default_rng(0),
    )
    x = plus.suggest()
    plus.observe(x, task(x))
    assert plus.novelty_p == pytest.approx(0.01)  # beyond all 99 simulated users: 1 / 100
    assert plus.current_ei_weight() == pytest.approx((CALIBRATED_TAU - 0.01) / CALIBRATED_TAU)


def test_the_policy_mix_picks_ei_or_the_af_by_a_coin(runs: tuple[Path, Path]) -> None:
    task = naf_plus_test_tasks("double_sphere", PLUS, 1)[0]
    af, config, hp = load_naf(runs[0], "best_af")
    sources = []
    for seed in range(12):
        plus = NAFPlusSequentialOptimizer(
            af, hp, config.budget, config.adaptive_set(), task.weights, StubDetector(p=0.05),  # type: ignore[arg-type]
            naf_only_iterations=1, mix="policy", mix_rng=np.random.default_rng(seed),
        )
        x = plus.suggest()
        plus.observe(x, task(x))
        assert plus.current_ei_weight() == pytest.approx(0.5)
        naf = plus.adaptive_set.maximize(naf_score_fn(plus.af, plus.gp, plus.t, plus.budget, plus.weights))
        ei = plus.adaptive_set.maximize(ei_score_fn(plus.gp, float(np.max(plus.y_obs))))
        suggestion = plus.suggest()
        source = plus.history[-1].source
        np.testing.assert_array_equal(suggestion, ei if source == "ei" else naf)
        sources.append(source)
    assert set(sources) == {"ei", "af"}


def test_the_new_options_are_checked(runs: tuple[Path, Path]) -> None:
    af, config, hp = load_naf(runs[0], "best_af")
    args = (af, hp, config.budget, config.adaptive_set(), (0.5, 0.5), StubDetector(p=0.5))
    with pytest.raises(ValueError, match="needs a calibration"):
        NAFPlusSequentialOptimizer(*args, novelty_test="calibrated")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mix_rng"):
        NAFPlusSequentialOptimizer(*args, mix="policy")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="novelty_test"):
        NAFPlusSequentialOptimizer(*args, novelty_test="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mix must"):
        NAFPlusSequentialOptimizer(*args, mix="other")  # type: ignore[arg-type]
