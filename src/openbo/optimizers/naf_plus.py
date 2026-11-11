"""NAF+ (Liao et al., CHI 2026): MetaBO's neural AF with objective weights and an EI fallback.

Plan: ``docs/plans/naf_plus.md``. NAF+ reuses ``optimizers/naf.py`` (the
weight-conditioned AF is ``NAFConfig.n_weights``); this module holds what is
specific to it: the fixed GP hyperparameters of its families (Phase 1), the
novelty detector (Phase 3), and the NAF+ optimizer with its training entry
point (Phase 4).

A training run directory holds the neural AF exactly as ``train_naf`` writes
it (``af.pt``, ``best_af.pt``, ``naf_config.json``, ``training_history.json``,
``training_state.pt``) plus ``naf_plus_config.json`` (the family and the
fallback settings) and the detector (``detector.pt``, ``detector.json``).
"""

from __future__ import annotations

import itertools
import json
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm, qmc
from threadpoolctl import threadpool_limits

from openbo.acquisition.naf import AdaptiveSet, NeuralAF, naf_score_fn, regular_grid
from openbo.acquisition.naf_plus import (
    CALIBRATED_TAU,
    NAF_ONLY_ITERATIONS,
    NOVELTY_THRESHOLD,
    calibrated_p_value,
    ei_score_fn,
    ei_weight,
    mixed_score_fn,
    novelty_p_values,
    novelty_statistic,
    novelty_z_scores,
)
from openbo.models.bnn import BNNConfig, PopulationBNN
from openbo.optimizers.naf import (
    FamilyGPHyperparameters,
    NAFConfig,
    NAFSequentialOptimizer,
    NAFTrainingResult,
    load_naf,
    run_naf,
    train_naf,
)
from openbo.test_functions.naf_plus_families import (
    DEFAULT_SCALE_RANGE,
    DEFAULT_SHIFT_RANGE,
    DOUBLE_SPHERE_FAMILIES,
    DoubleSphereUser,
    WeightedTask,
    sample_double_sphere_users,
    sample_weighted_tasks,
    validate_weights,
)

Task = Callable[[NDArray[np.float64]], NDArray[np.float64]]

# Double-sphere users at the default ranges, from
# select_family_hyperparameters(sample_weighted_tasks(60, seed=11), dim=2, seed=0).
# Type-2 ML has no finite optimum on this family (see the helper's docstring).
DOUBLE_SPHERE_GP_HYPERPARAMETERS = FamilyGPHyperparameters(lengthscale=(0.35, 0.35), variance=1.41)
# The wide family (centres (0.2, 0.2) and (0.8, 0.8)), selected the same way:
# select_family_hyperparameters(sample_weighted_tasks(60, seed=11, centres=WIDE_TRAINING_CENTRES), 2, seed=0).
DOUBLE_SPHERE_WIDE_GP_HYPERPARAMETERS = FamilyGPHyperparameters(lengthscale=(0.55, 0.55), variance=6.07)
FAMILY_GP_HYPERPARAMETERS = {
    "double_sphere": DOUBLE_SPHERE_GP_HYPERPARAMETERS,
    "double_sphere_wide": DOUBLE_SPHERE_WIDE_GP_HYPERPARAMETERS,
}


def select_family_hyperparameters(
    tasks: Sequence[Task],
    dim: int,
    lengthscales: Sequence[float] = tuple(np.round(np.arange(0.1, 1.0001, 0.05), 2)),
    n_observations: Sequence[int] = (3, 10, 20),
    n_test: int = 200,
    seed: int = 0,
) -> FamilyGPHyperparameters:
    """Fixed hyperparameters for a smooth family: matched variance, held-out lengthscale.

    Used where type-2 ML is ill-posed. On a noise-free, very smooth family
    (the double sphere is a quadratic) the RBF likelihood keeps improving as
    the lengthscale and variance grow together, both on the fitting data and
    on held-out points, so no finite optimum exists. Instead:

    - The variance is the family's mean ``E[y^2]`` over uniform points, the
      second moment a zero-mean prior should match. It also keeps the raw
      ``sigma`` feature the AF reads on the scale of the values.
    - The isotropic lengthscale minimizes the held-out predictive negative
      log likelihood, the median over tasks averaged over ``n_observations``
      random training points, which covers the GP's sizes in an episode.

    The noise stays at NAF's near-zero ``NAF_GP_NOISE``.
    """
    if not tasks:
        raise ValueError("tasks must be non-empty.")
    if not lengthscales or min(lengthscales) <= 0.0:
        raise ValueError("lengthscales must be non-empty and positive.")
    if not n_observations or min(n_observations) < 1:
        raise ValueError("n_observations must be non-empty and positive.")
    rng = np.random.default_rng(seed)
    probe = qmc.Sobol(d=dim, scramble=True, seed=seed).random(1024)
    variance = float(np.mean([np.mean(np.asarray(task(probe)) ** 2) for task in tasks]))
    n_max = max(n_observations)
    splits = [(rng.random((n_max, dim)), rng.random((n_test, dim))) for _ in tasks]
    targets = [(np.asarray(task(x_obs)), np.asarray(task(x_test))) for task, (x_obs, x_test) in zip(tasks, splits)]

    best: tuple[float, float] | None = None
    for lengthscale in lengthscales:
        candidate = FamilyGPHyperparameters(lengthscale=(float(lengthscale),) * dim, variance=variance)
        scores = []
        for n in n_observations:
            nll = []
            for (x_obs, x_test), (y_obs, y_test) in zip(splits, targets):
                gp = candidate.new_gp()
                gp.fit(x_obs[:n], y_obs[:n])
                mean, var = gp.posterior(x_test)
                std = np.sqrt(np.maximum(var, 1e-12))
                nll.append(-float(np.mean(norm.logpdf(y_test, mean, std))))
            scores.append(float(np.median(nll)))
        score = float(np.mean(scores))
        if best is None or score < best[0]:
            best = (score, float(lengthscale))
    assert best is not None
    return FamilyGPHyperparameters(lengthscale=(best[1],) * dim, variance=round(variance, 2))


# The paper's detector (Sec. 5.6): a 2 x 256 MC-dropout BNN, dropout 0.2, 2000 epochs.
DETECTOR_HIDDEN_UNITS = 256
DETECTOR_HIDDEN_LAYERS = 2
DETECTOR_DROPOUT = 0.2
DETECTOR_EPOCHS = 2000
DETECTOR_FILE = "detector.pt"
DETECTOR_META_FILE = "detector.json"


class WeightedUser(Protocol):
    """A synthetic user whose scalarized objective takes the weights (``DoubleSphereUser``)."""

    def __call__(self, x: ArrayLike, weights: ArrayLike) -> NDArray[np.float64]: ...


def weight_grid(n_objectives: int, step: float = 0.1) -> NDArray[np.float64]:
    """Every weight vector on the simplex with entries in multiples of ``step``, zeros included.

    For two objectives and ``step = 0.1`` these are the paper's 11 detector
    settings, ``w_1 in {0, 0.1, ..., 1}``. Shape ``(K, n_objectives)``.
    """
    if n_objectives < 1:
        raise ValueError("n_objectives must be at least 1.")
    n_steps = round(1.0 / step)
    if n_steps < 1 or abs(n_steps * step - 1.0) > 1e-9:
        raise ValueError("step must divide 1.")
    rows = [c for c in itertools.product(range(n_steps + 1), repeat=n_objectives) if sum(c) == n_steps]
    return np.asarray(rows, dtype=np.float64) / n_steps


def build_detector_dataset(
    user_sampler: Callable[[int, int], Sequence[WeightedUser]],
    weights: NDArray[np.float64],
    dim: int,
    n_users: int = 20,
    points_per_dim: int = 10,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The detector's training data, after the paper's user study (Sec. 5.6).

    For each weight vector, ``n_users`` fresh users (``user_sampler(n, seed)``)
    are evaluated on a regular grid of ``points_per_dim^dim`` designs (the
    paper's 20 users x 100 designs per weight). Returns inputs ``[x, w]`` of
    shape ``(N, dim + L)`` and the scalarized values ``(N,)``.
    """
    w_all = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    if n_users < 1:
        raise ValueError("n_users must be positive.")
    grid = regular_grid(points_per_dim, dim)
    inputs, targets = [], []
    for k, w in enumerate(w_all):
        for user in user_sampler(n_users, seed + 1009 * k):
            inputs.append(np.hstack([grid, np.repeat(w[None, :], grid.shape[0], axis=0)]))
            targets.append(np.asarray(user(grid, w), dtype=np.float64))
    return np.vstack(inputs), np.concatenate(targets)


@dataclass
class NoveltyDetector:
    """NAF+'s novelty detector (Sec. 4.6): an MC-dropout BNN of the population, ``(x, w) -> y``.

    Its predictive variance is the BNN's total variance (MC dropout plus the
    variance head). The population's spread between users at the same
    ``(x, w)`` lands in the variance head, so the p-value asks whether a
    value is plausible for a typical user (author question 3).
    """

    bnn: PopulationBNN
    dim: int
    n_weights: int

    def _inputs(self, x: ArrayLike, weights: ArrayLike) -> NDArray[np.float64]:
        x_arr = np.atleast_2d(np.asarray(x, dtype=np.float64))
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if x_arr.shape[1] != self.dim or w.shape[0] != self.n_weights:
            raise ValueError(f"x must have {self.dim} columns and weights {self.n_weights} entries.")
        return np.hstack([x_arr, np.repeat(w[None, :], x_arr.shape[0], axis=0)])

    def predict(self, x: ArrayLike, weights: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predictive mean and std of the population at ``x`` under ``weights``."""
        mean, var = self.bnn.predict(self._inputs(x, weights))
        return mean, np.sqrt(np.maximum(var, 1e-300))

    def p_values(self, x: ArrayLike, weights: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
        """Two-tailed p-values of the observations ``y`` at ``x`` (eq. 3)."""
        mean, std = self.predict(x, weights)
        return novelty_p_values(mean, std, np.asarray(y, dtype=np.float64).reshape(-1))

    def z_scores(self, x: ArrayLike, weights: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
        """Standardized residuals of the observations ``y`` at ``x`` (the calibrated test's input)."""
        mean, std = self.predict(x, weights)
        return novelty_z_scores(mean, std, np.asarray(y, dtype=np.float64).reshape(-1))

    def save(self, directory: str | Path) -> None:
        """Write ``detector.pt`` (the BNN) and ``detector.json`` (its input layout)."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.bnn.save(root / DETECTOR_FILE)
        meta = {"dim": self.dim, "n_weights": self.n_weights}
        (root / DETECTOR_META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> NoveltyDetector:
        """Load a detector written by :meth:`save`."""
        root = Path(directory)
        meta = json.loads((root / DETECTOR_META_FILE).read_text(encoding="utf-8"))
        return cls(PopulationBNN.load(root / DETECTOR_FILE), int(meta["dim"]), int(meta["n_weights"]))


def detector_config(input_dim: int, **overrides: object) -> BNNConfig:
    """The paper's detector BNN (2 x 256, dropout 0.2) for ``input_dim = dim + n_weights``."""
    config = BNNConfig(
        input_dim=input_dim,
        hidden_units=DETECTOR_HIDDEN_UNITS,
        n_hidden_layers=DETECTOR_HIDDEN_LAYERS,
        dropout=DETECTOR_DROPOUT,
    )
    return replace(config, **overrides)  # type: ignore[arg-type]


def fit_detector(
    inputs: NDArray[np.float64],
    y: NDArray[np.float64],
    dim: int,
    n_weights: int,
    epochs: int = DETECTOR_EPOCHS,
    config: BNNConfig | None = None,
) -> NoveltyDetector:
    """Train a detector on ``(inputs, y)`` from :func:`build_detector_dataset`."""
    if inputs.ndim != 2 or inputs.shape[1] != dim + n_weights:
        raise ValueError(f"inputs must have shape (N, {dim + n_weights}).")
    bnn = PopulationBNN(config if config is not None else detector_config(dim + n_weights))
    if bnn.config.input_dim != dim + n_weights:
        raise ValueError("config.input_dim must be dim + n_weights.")
    bnn.fit_nll(inputs, y, epochs)
    return NoveltyDetector(bnn, dim, n_weights)


NAF_PLUS_CONFIG_FILE = "naf_plus_config.json"
NAF_PLUS_FAMILIES = tuple(DOUBLE_SPHERE_FAMILIES)
# The paper's budget for the synthetic tests (App. A.3).
DOUBLE_SPHERE_BUDGET = 20


@dataclass(frozen=True)
class NAFPlusConfig:
    """What NAF+ adds to a ``NAFConfig``: the user family, the fallback and its detector.

    The family's fixed GP hyperparameters were selected at the default shift
    and scale ranges; other ranges keep them (select again if the ranges
    change a lot).
    """

    family: str = "double_sphere"
    shift_range: float = DEFAULT_SHIFT_RANGE
    scale_range: float = DEFAULT_SCALE_RANGE
    tau: float = NOVELTY_THRESHOLD
    naf_only_iterations: int = NAF_ONLY_ITERATIONS
    detector_epochs: int = DETECTOR_EPOCHS
    detector_users: int = 20
    detector_points_per_dim: int = 10
    detector_weight_step: float = 0.1
    detector_hidden_units: int = DETECTOR_HIDDEN_UNITS
    detector_seed: int = 0

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.family not in NAF_PLUS_FAMILIES:
            raise ValueError(f"family must be one of {list(NAF_PLUS_FAMILIES)}.")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1].")
        if self.naf_only_iterations < 0:
            raise ValueError("naf_only_iterations must be non-negative.")
        if self.detector_epochs < 1 or self.detector_users < 1 or self.detector_points_per_dim < 2:
            raise ValueError("detector_epochs and detector_users must be positive, detector_points_per_dim >= 2.")
        sample_double_sphere_users(1, 0, self.shift_range, self.scale_range)  # validates the ranges

    @property
    def dim(self) -> int:
        """Input dimension of the family."""
        return 2

    @property
    def n_objectives(self) -> int:
        """Number of objectives (and of weights)."""
        return 2

    @property
    def gp_hyperparameters(self) -> FamilyGPHyperparameters:
        """The family's fixed GP hyperparameters."""
        return FAMILY_GP_HYPERPARAMETERS[self.family]

    def centres(self, novel: bool = False) -> tuple[tuple[float, float], ...]:
        """The family's objective centres: the training ones, or the novel users' (Test 1.2)."""
        training, novel_centres = DOUBLE_SPHERE_FAMILIES[self.family]
        return novel_centres if novel else training

    def users(self, n: int, seed: int, novel: bool = False) -> list[DoubleSphereUser]:
        """``n`` users; ``novel`` uses the family's novel centres (Test 1.2)."""
        return sample_double_sphere_users(n, seed, self.shift_range, self.scale_range, self.centres(novel))

    def tasks(self, n: int, seed: int, novel: bool = False) -> list[WeightedTask]:
        """``n`` users, each with random weights (training episodes and test tasks)."""
        return sample_weighted_tasks(n, seed, self.shift_range, self.scale_range, self.centres(novel))

    def save(self, directory: str | Path) -> None:
        """Write ``naf_plus_config.json``."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / NAF_PLUS_CONFIG_FILE).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> NAFPlusConfig:
        """Read ``naf_plus_config.json``."""
        return cls(**json.loads((Path(directory) / NAF_PLUS_CONFIG_FILE).read_text(encoding="utf-8")))


CALIBRATION_FILE = "novelty_calibration.json"
# Draws the calibration users; no test or training draw uses this seed.
CALIBRATION_SEED = 5003
CALIBRATION_USERS = 500
CALIBRATION_POLICIES = ("greedy", "sample")


@dataclass
class NoveltyCalibration:
    """The calibrated novelty test's null distribution (plan, "Design stance", item 3).

    ``statistics[policy][n - 1]`` holds ``S_n = sum z_i^2`` after ``n``
    observations for each simulated training-like user, run with the neural
    AF alone (``policy`` "greedy" takes its argmax, "sample" samples it), so
    the observation points are where the policy actually puts them.
    """

    statistics: dict[str, list[list[float]]]
    n_users: int
    seed: int

    def p_value(self, policy: str, statistic: float, n: int) -> float:
        """Calibrated p-value of ``statistic`` after ``n`` observations under ``policy``."""
        if policy not in self.statistics:
            raise ValueError(f"No calibration for policy {policy!r}; have {sorted(self.statistics)}.")
        per_n = self.statistics[policy]
        if not 1 <= n <= len(per_n):
            raise ValueError(f"n must be in [1, {len(per_n)}].")
        return calibrated_p_value(per_n[n - 1], statistic)

    def save(self, directory: str | Path) -> None:
        """Write ``novelty_calibration.json``."""
        payload = {"n_users": self.n_users, "seed": self.seed, "statistics": self.statistics}
        (Path(directory) / CALIBRATION_FILE).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> NoveltyCalibration:
        """Read ``novelty_calibration.json``."""
        path = Path(directory) / CALIBRATION_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run scripts/calibrate_naf_plus_novelty.py first.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["statistics"], int(payload["n_users"]), int(payload["seed"]))


def calibration_statistics(
    directory: str | Path, task: WeightedTask, policy: str, rng: np.random.Generator | None
) -> list[float]:
    """``S_n`` for ``n = 1..budget`` of one training-like user, run with the AF alone under ``policy``."""
    if policy not in CALIBRATION_POLICIES:
        raise ValueError(f"policy must be one of {list(CALIBRATION_POLICIES)}.")
    detector = NoveltyDetector.load(directory)
    optimizer = NAFPlusSequentialOptimizer.from_run(
        directory, task.weights, use_fallback=False, rng=rng if policy == "sample" else None
    )
    z: list[float] = []
    for _ in range(optimizer.budget):
        x = optimizer.suggest()
        y = task(x)
        optimizer.observe(x, y)
        # One point per call, as the optimizer scores observations.
        z.extend(detector.z_scores(x, task.weights, y).tolist())
    return np.cumsum(np.square(z)).tolist()


def _calibration_worker(payload: tuple) -> list[float]:  # type: ignore[type-arg]
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        directory, task, policy, seed = payload
        return calibration_statistics(directory, task, policy, np.random.default_rng(seed))


def calibrate_novelty(
    directory: str | Path,
    n_users: int = CALIBRATION_USERS,
    seed: int = CALIBRATION_SEED,
    workers: int = 1,
    policies: Sequence[str] = CALIBRATION_POLICIES,
) -> NoveltyCalibration:
    """Simulate ``n_users`` fresh training-like users per policy and return the null distribution of ``S_n``.

    Users and weights come from the run's own family (``NAFPlusConfig``), drawn
    with ``seed``; the sampled policy seeds user ``i`` with ``seed + 1009 i``.
    """
    if n_users < 1:
        raise ValueError("n_users must be positive.")
    plus = NAFPlusConfig.load(directory)
    tasks = plus.tasks(n_users, seed)
    statistics: dict[str, list[list[float]]] = {}
    for policy in policies:
        payloads = [(str(directory), task, policy, seed + 1009 * i) for i, task in enumerate(tasks)]
        if workers > 1:
            import multiprocessing

            with multiprocessing.get_context("spawn").Pool(workers) as pool:
                rows = pool.map(_calibration_worker, payloads)
        else:
            rows = [_calibration_worker(p) for p in payloads]
        statistics[policy] = np.asarray(rows).T.tolist()
    return NoveltyCalibration(statistics, n_users, seed)


@dataclass
class NAFPlusStep:
    """What the fallback did before one suggestion: the mean p-value so far and ``lambda_EI``."""

    t: int
    p_bar: float | None
    lambda_ei: float
    novelty_p: float | None = None
    source: str = "af"


class NAFPlusSequentialOptimizer(NAFSequentialOptimizer):
    """Ask/tell NAF+: the neural AF mixed with EI by the novelty detector (CHI 2026, Sec. 4).

    ``observe`` takes the scalarized objective ``sum_k w_k f_k``. Every
    observation gets a p-value under the detector; from iteration
    ``naf_only_iterations + 1`` on, ``lambda_EI`` (eq. 5) comes from the mean
    p-value of all observations so far (author question 3), and the next
    point maximizes eq. 4 over the adaptive set. Before that, or without a
    detector (the "without EI" ablation), it is the neural AF alone, exactly
    as ``NAFSequentialOptimizer``. ``history`` holds one :class:`NAFPlusStep`
    per suggestion. With an ``rng`` the neural AF's points are sampled from its
    softmax (as in training); once ``lambda_EI > 0``, the point is still the
    argmax of the mixed score (eq. 4), whose normalized scale is no policy.

    OpenBO's alternatives (plan, "Design stance", items 2-3), off by default:
    ``novelty_test="calibrated"`` replaces the mean p-value with the
    calibrated p-value of ``S_n = sum z_i^2`` (needs ``calibration``; pass
    ``tau=CALIBRATED_TAU``), and ``mix="policy"`` replaces eq. 4: with
    probability ``lambda_EI`` (a coin from ``mix_rng``) the point is EI's
    argmax, otherwise the neural AF's own choice (argmax, or a sample with
    ``rng``).
    """

    def __init__(
        self,
        af: NeuralAF,
        gp_hyperparameters: FamilyGPHyperparameters,
        budget: int,
        adaptive_set: AdaptiveSet,
        weights: Sequence[float],
        detector: NoveltyDetector | None = None,
        tau: float = NOVELTY_THRESHOLD,
        naf_only_iterations: int = NAF_ONLY_ITERATIONS,
        rng: np.random.Generator | None = None,
        avoid_repeats: bool = False,
        novelty_test: str = "mean_p",
        calibration: NoveltyCalibration | None = None,
        mix: str = "score",
        mix_rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(af, gp_hyperparameters, budget, adaptive_set, weights, rng, avoid_repeats)
        if novelty_test not in ("mean_p", "calibrated"):
            raise ValueError("novelty_test must be 'mean_p' or 'calibrated'.")
        if mix not in ("score", "policy"):
            raise ValueError("mix must be 'score' or 'policy'.")
        if detector is not None and novelty_test == "calibrated" and calibration is None:
            raise ValueError("novelty_test='calibrated' needs a calibration.")
        if detector is not None and mix == "policy" and mix_rng is None:
            raise ValueError("mix='policy' needs mix_rng.")
        # A plain AF ignores the weights, but the detector and the scalarization use them.
        n_weights = detector.n_weights if detector is not None else len(weights)
        self.objective_weights = validate_weights(weights, n_weights)
        if detector is not None and detector.dim != af.dim:
            raise ValueError("The detector's input dimension does not match the AF's.")
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in (0, 1].")
        if naf_only_iterations < 0:
            raise ValueError("naf_only_iterations must be non-negative.")
        self.detector = detector
        self.tau = tau
        self.naf_only_iterations = naf_only_iterations
        self.novelty_test = novelty_test
        self.calibration = calibration
        self.mix = mix
        self.mix_rng = mix_rng
        self.p_values: list[float] = []
        self.z_values: list[float] = []
        self.history: list[NAFPlusStep] = []

    @classmethod
    def from_run(
        cls,
        directory: str | Path,
        weights: Sequence[float],
        name: str = "best_af",
        use_fallback: bool = True,
        rng: np.random.Generator | None = None,
        avoid_repeats: bool = False,
        novelty_test: str = "mean_p",
        mix: str = "score",
        mix_rng: np.random.Generator | None = None,
    ) -> NAFPlusSequentialOptimizer:
        """Load a ``train_naf_plus`` run; ``use_fallback=False`` skips the detector, ``rng`` samples the AF.

        ``avoid_repeats`` never suggests a point already evaluated.
        ``novelty_test="calibrated"`` loads ``novelty_calibration.json`` and uses
        ``tau = CALIBRATED_TAU``; ``mix="policy"`` needs ``mix_rng``.
        """
        af, config, gp_hyperparameters = load_naf(directory, name)
        af.eval()
        plus = NAFPlusConfig.load(directory)
        detector = NoveltyDetector.load(directory) if use_fallback else None
        calibrated = use_fallback and novelty_test == "calibrated"
        calibration = NoveltyCalibration.load(directory) if calibrated else None
        return cls(
            af,
            gp_hyperparameters,
            config.budget,
            config.adaptive_set(),
            weights,
            detector,
            CALIBRATED_TAU if calibrated else plus.tau,
            plus.naf_only_iterations,
            rng,
            avoid_repeats,
            novelty_test,
            calibration,
            mix,
            mix_rng,
        )

    @property
    def p_bar(self) -> float | None:
        """Mean p-value of the observations so far (``None`` before any, or without a detector)."""
        return float(np.mean(self.p_values)) if self.p_values else None

    @property
    def novelty_p(self) -> float | None:
        """The p-value the fallback uses: ``p_bar``, or the calibrated p-value of ``S_n``."""
        if self.novelty_test == "mean_p":
            return self.p_bar
        if self.detector is None or not self.z_values:
            return None
        assert self.calibration is not None
        policy = "greedy" if self.rng is None else "sample"
        return self.calibration.p_value(policy, novelty_statistic(self.z_values), len(self.z_values))

    def current_ei_weight(self) -> float:
        """``lambda_EI`` for the next suggestion."""
        p = self.novelty_p
        if self.detector is None or self.t < self.naf_only_iterations or p is None:
            return 0.0
        return ei_weight(p, self.tau)

    def suggest(self) -> NDArray[np.float64]:
        """The next point, shape ``(1, d)``."""
        if self.t >= self.budget:
            raise RuntimeError(f"The budget of {self.budget} evaluations is used up.")
        lam = self.current_ei_weight()
        step = NAFPlusStep(self.t, self.p_bar, lam, self.novelty_p)
        self.history.append(step)
        score = naf_score_fn(self.af, self.gp, self.t, self.budget, self.weights)
        if lam > 0.0 and self.mix == "policy":
            assert self.mix_rng is not None
            if self.mix_rng.random() < lam:
                step.source = "ei"
                return self.adaptive_set.maximize(ei_score_fn(self.gp, float(np.max(self.y_obs))))
        elif lam > 0.0:
            step.source = "mixed"
            ei = ei_score_fn(self.gp, float(np.max(self.y_obs)))
            mixed = mixed_score_fn(score, ei, lam, self.adaptive_set.global_grid)
            return self.adaptive_set.maximize(mixed, self.excluded_points)
        if self.rng is not None:
            return self.adaptive_set.sample(score, self.rng, self.excluded_points)
        return self.adaptive_set.maximize(score, self.excluded_points)

    def observe(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Record evaluations of the scalarized objective, refit the GP, and score their novelty."""
        super().observe(x, y)
        if self.detector is not None:
            x_new = np.atleast_2d(np.asarray(x, dtype=np.float64))
            y_new = np.asarray(y, dtype=np.float64).reshape(-1)
            self.p_values.extend(self.detector.p_values(x_new, self.objective_weights, y_new).tolist())
            if self.novelty_test == "calibrated":
                self.z_values.extend(self.detector.z_scores(x_new, self.objective_weights, y_new).tolist())


def run_naf_plus(
    objective: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    optimizer: NAFPlusSequentialOptimizer,
    n_iter: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run ``n_iter`` evaluations of the scalarized ``objective`` (default: the AF's budget)."""
    return run_naf(objective, optimizer, n_iter)


@dataclass
class NAFPlusTrainingResult:
    """The neural AF's training result plus the detector and the seconds it took to fit."""

    naf: NAFTrainingResult
    detector: NoveltyDetector
    detector_seconds: float = field(default=0.0)


def train_naf_plus(
    naf_config: NAFConfig,
    plus_config: NAFPlusConfig,
    run_dir: str | Path,
    log: Callable[[str], None] | None = None,
    resume: bool = False,
    detector_from: str | Path | None = None,
) -> NAFPlusTrainingResult:
    """Train NAF+: the detector first (once), then the neural AF with PPO on fresh users.

    Every episode draws a new user with new weights from ``plus_config``
    (MetaBO's "full" mode; the paper's users generated on the fly).
    ``naf_config.n_weights`` is the family's number of objectives for NAF+, or
    0 for the "without weights" ablation (same users, the AF blind to ``w``).
    The detector is fitted unless the run directory already holds one, or
    copied from ``detector_from`` (another run with the same family settings).
    ``resume`` continues the AF training as ``train_naf`` does.
    """
    if naf_config.dim != plus_config.dim:
        raise ValueError(f"naf_config.dim must be {plus_config.dim} for {plus_config.family}.")
    if naf_config.n_weights not in (0, plus_config.n_objectives):
        raise ValueError(f"naf_config.n_weights must be 0 or {plus_config.n_objectives}.")
    root = Path(run_dir)
    if (root / NAF_PLUS_CONFIG_FILE).exists():
        if NAFPlusConfig.load(root) != plus_config:
            raise ValueError(f"{root} was trained with different NAF+ settings.")
    else:
        plus_config.save(root)

    detector_seconds = 0.0
    if (root / DETECTOR_FILE).exists():
        detector = NoveltyDetector.load(root)
    elif detector_from is not None:
        source = Path(detector_from)
        if NAFPlusConfig.load(source) != plus_config:
            raise ValueError(f"{source} has different NAF+ settings.")
        for name in (DETECTOR_FILE, DETECTOR_META_FILE):
            shutil.copy2(source / name, root / name)
        detector = NoveltyDetector.load(root)
    else:
        start = time.perf_counter()
        users = lambda n, seed: plus_config.users(n, plus_config.detector_seed + seed)  # noqa: E731
        inputs, y = build_detector_dataset(
            users,
            weight_grid(plus_config.n_objectives, plus_config.detector_weight_step),
            plus_config.dim,
            plus_config.detector_users,
            plus_config.detector_points_per_dim,
        )
        bnn_config = detector_config(
            plus_config.dim + plus_config.n_objectives,
            hidden_units=plus_config.detector_hidden_units,
            seed=plus_config.detector_seed,
        )
        detector = fit_detector(
            inputs, y, plus_config.dim, plus_config.n_objectives, plus_config.detector_epochs, bnn_config
        )
        detector.save(root)
        detector_seconds = time.perf_counter() - start
        if log is not None:
            log(f"detector: {inputs.shape[0]} points, {plus_config.detector_epochs} epochs, {detector_seconds:.0f} s")

    result = train_naf(
        naf_config,
        plus_config.gp_hyperparameters,
        task_sampler=lambda n, seed: plus_config.tasks(n, seed),
        checkpoint_dir=root,
        log=log,
        resume=resume,
    )
    return NAFPlusTrainingResult(result, detector, detector_seconds)
