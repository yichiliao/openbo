"""MetaBO-style neural acquisition functions (Volpp et al., ICLR 2020).

Plan: ``docs/plans/naf.md``. Where the paper and the authors' reference code
disagree, OpenBO follows the code's behaviour (reimplemented; the code is
AGPL-3.0 and none of it is copied).

- Phase 1: fixed GP hyperparameters per function family.
- Phase 3: ``BOEpisode`` (one BO run as an RL episode), ``NAFConfig`` (Table 4
  of the paper plus the reference code's choices) and ``train_naf`` (PPO with
  GAE, a value function of ``(t, T)`` only, parallel rollouts).
- Optional extension: ``NAFConfig.n_weights > 0`` trains a weight-conditioned
  AF on tasks that carry objective ``weights``; the AF and the value function
  then also read the weights. With ``n_weights = 0`` everything is MetaBO's
  NAF, also on weighted tasks.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

from openbo.acquisition.naf import (
    AdaptiveSet,
    NeuralAF,
    af_weights,
    gp_mean_std,
    mlp,
    naf_features,
    naf_score_fn,
    validate_weights,
)
from openbo.models.gp_scratch import GPScratch
from openbo.test_functions.metabo_families import (
    branin_grid_tasks,
    goldstein_price_grid_tasks,
    hartmann3_grid_tasks,
    sample_branin_tasks,
    sample_goldstein_price_tasks,
    sample_hartmann3_tasks,
)

# The reference's GP noise is effectively zero (8.9e-16 for Branin); GPScratch
# adds its own 1e-10 jitter on top of this.
NAF_GP_NOISE = 1e-10
TRAINING_STATE_FILE = "training_state.pt"


@dataclass(frozen=True)
class FamilyGPHyperparameters:
    """Fixed hyperparameters of the zero-mean ARD RBF GP used for one family."""

    lengthscale: tuple[float, ...]
    variance: float
    noise: float = NAF_GP_NOISE

    def new_gp(self) -> GPScratch:
        """A GP with these hyperparameters: no fitting, no output standardization."""
        return GPScratch(
            lengthscale=np.asarray(self.lengthscale, dtype=np.float64),
            variance=self.variance,
            noise=self.noise,
            kernel_type="rbf",
            optimize_hyperparameters=False,
            standardize_targets=False,
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly form."""
        return {"lengthscale": list(self.lengthscale), "variance": self.variance, "noise": self.noise}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FamilyGPHyperparameters:
        """Inverse of :meth:`to_dict`."""
        return cls(
            lengthscale=tuple(float(v) for v in payload["lengthscale"]),  # type: ignore[union-attr]
            variance=float(payload["variance"]),  # type: ignore[arg-type]
            noise=float(payload.get("noise", NAF_GP_NOISE)),  # type: ignore[arg-type]
        )


# The reference code's fixed Branin hyperparameters (GPy type-2 ML on 100 points).
# OpenBO's own fit on normalized Branin finds a different optimum (lengthscales
# about 0.32 and 2.3, variance about 290); the reference values are kept because
# the neural AF reads the raw posterior std, whose scale they set.
BRANIN_GP_HYPERPARAMETERS = FamilyGPHyperparameters(lengthscale=(0.235, 0.578), variance=2.0)

# These two are OpenBO's own fits (the paper's recipe: type-2 ML offline on
# functions of the family, here `fit_family_hyperparameters` with 100 Sobol points
# per task), made before it was noticed that the reference's training scripts give
# the authors' values: Goldstein-Price lengthscales (0.130, 0.07), variance 0.616,
# noise 1e-6; Hartmann-3 (0.716, 0.298, 0.186), variance 0.83, noise 1.688e-11.
# The fits are close to those (the authors' Hartmann-3 AF works as well with either
# set). Each is the median over 10 random tasks of the family
# (`sample_*_tasks(10, 0)`, fit seeds 0-9).
# Unlike Branin's fit, which jumps between variances of 1 and 400 from task to
# task, these are stable across the 10 tasks: variances between 0.51 and 0.93
# (Goldstein-Price) and 0.52 and 0.89 (Hartmann-3), lengthscales within about 25%.
GOLDSTEIN_PRICE_GP_HYPERPARAMETERS = FamilyGPHyperparameters(lengthscale=(0.109, 0.087), variance=0.652)
HARTMANN3_GP_HYPERPARAMETERS = FamilyGPHyperparameters(lengthscale=(0.709, 0.305, 0.173), variance=0.694)


def fit_family_hyperparameters(
    task: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    dim: int,
    n_points: int = 100,
    seed: int = 0,
) -> FamilyGPHyperparameters:
    """Type-2 ML fit of a zero-mean RBF GP on ``n_points`` Sobol samples of one task.

    For families the reference does not cover; pass a representative task
    (for example the untransformed base function). One function, as in the
    reference: pooling several translated tasks under near-zero noise forces
    very short lengthscales, since nearby points from different tasks
    disagree. The noise stays fixed and near zero.
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")
    x = qmc.Sobol(d=dim, scramble=True, seed=seed).random(n_points)
    gp = GPScratch(
        lengthscale=0.3,
        variance=1.0,
        noise=NAF_GP_NOISE,
        kernel_type="rbf",
        optimize_hyperparameters=True,
        optimizer_seed=seed,
        # A single L-BFGS-B start often stops in a poor local optimum on near-noise-free data;
        # 32 starts found the best optimum for all of 10 Sobol designs, 8 starts for only 7.
        optimizer_restarts=32,
        standardize_targets=False,
    )
    gp.fit(x, np.asarray(task(x), dtype=np.float64))
    return FamilyGPHyperparameters(
        lengthscale=tuple(float(v) for v in np.atleast_1d(gp.lengthscale)), variance=float(gp.variance)
    )


class Task(Protocol):
    """A maximized test function on ``[0, 1]^d`` with a known optimum value.

    For a weight-conditioned AF the task also has ``weights``, the objective
    weights its values and optimum already include.
    """

    optimum: float

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...


def log_regret_reward(regret: float) -> float:
    """The reference's training reward ``-log10(max(1e-20, R_t))``."""
    return -float(np.log10(max(1e-20, regret)))


class BOEpisode:
    """One BO run on one task as an RL episode of ``budget`` steps.

    The state is the candidate set ``xi_t`` with its features; the action is
    an index into it. The GP keeps fixed hyperparameters and is refitted on
    every new point. There is no initial design. Each step's reward is
    ``-log10`` of the simple regret after that step.
    """

    def __init__(
        self,
        task: Task,
        gp_hyperparameters: FamilyGPHyperparameters,
        adaptive_set: AdaptiveSet,
        budget: int,
    ) -> None:
        if budget < 1:
            raise ValueError("budget must be positive.")
        self.task = task
        self.gp = gp_hyperparameters.new_gp()
        self.adaptive_set = adaptive_set
        self.budget = budget
        dim = adaptive_set.global_grid.shape[1]
        self.x_obs = np.empty((0, dim), dtype=np.float64)
        self.y_obs = np.empty(0, dtype=np.float64)
        self.candidates: NDArray[np.float64] | None = None

    @property
    def t(self) -> int:
        """Evaluations so far."""
        return int(self.y_obs.shape[0])

    @property
    def done(self) -> bool:
        """Whether the budget is used up."""
        return self.t >= self.budget

    @property
    def regret(self) -> float:
        """Simple regret of the best evaluation so far (``inf`` before any)."""
        return float(self.task.optimum - self.y_obs.max()) if self.t else float("inf")

    def observation(self, af: NeuralAF) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Build ``xi_t`` with ``af`` and return its features ``(n, F)`` and logits ``(n,)``."""
        if self.done:
            raise RuntimeError("The episode is over.")
        weights = af_weights(af, getattr(self.task, "weights", None))
        points, logits = self.adaptive_set.candidates(naf_score_fn(af, self.gp, self.t, self.budget, weights))
        self.candidates = points
        mean, std = gp_mean_std(self.gp, points)
        features = naf_features(mean, std, points, self.t, self.budget, af.use_x, weights)
        return features, logits

    def step(self, index: int) -> float:
        """Evaluate candidate ``index`` of the last observation; return the reward."""
        if self.candidates is None:
            raise RuntimeError("Call observation() before step().")
        x = self.candidates[index][None, :]
        y = float(np.asarray(self.task(x), dtype=np.float64).reshape(-1)[0])
        self.x_obs = np.vstack([self.x_obs, x])
        self.y_obs = np.append(self.y_obs, y)
        self.gp.fit(self.x_obs, self.y_obs)
        self.candidates = None
        return log_regret_reward(self.regret)


@dataclass(frozen=True)
class NAFConfig:
    """MetaBO training settings: the paper's Table 4 and the reference code's choices."""

    dim: int
    budget: int = 30
    use_x: bool = True
    hidden: tuple[int, ...] = (200, 200, 200, 200)
    # Candidate set (the reference's, for 2D problems).
    points_per_dim: int = 31
    n_local_maxima: int = 5
    n_local: int = 1000
    # PPO.
    n_iterations: int = 2000
    steps_per_iteration: int = 1200
    epochs: int = 4
    minibatch_size: int = 60
    learning_rate: float = 1e-4
    clip: float = 0.15
    value_coefficient: float = 1.0
    entropy_coefficient: float = 0.01
    gamma: float = 0.98
    gae_lambda: float = 0.98
    n_workers: int = 1
    # Threads for the PPO update (rollout workers use one thread each).
    update_threads: int = 1
    seed: int = 0
    # Number of objective weights the AF and value function read (0 = MetaBO's NAF).
    n_weights: int = 0

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.dim < 1 or self.budget < 1:
            raise ValueError("dim and budget must be positive.")
        if self.steps_per_iteration % self.budget != 0:
            raise ValueError("steps_per_iteration must be a multiple of budget (whole episodes).")
        if self.n_iterations < 1 or self.epochs < 1 or self.minibatch_size < 1:
            raise ValueError("n_iterations, epochs and minibatch_size must be positive.")
        if self.n_workers < 1 or self.update_threads < 1:
            raise ValueError("n_workers and update_threads must be positive.")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma must be in (0, 1] and gae_lambda in [0, 1].")
        if self.n_weights < 0:
            raise ValueError("n_weights must be non-negative.")

    @property
    def episodes_per_iteration(self) -> int:
        """Whole episodes per PPO iteration."""
        return self.steps_per_iteration // self.budget

    def adaptive_set(self) -> AdaptiveSet:
        """The candidate set these settings describe (fixed for a whole run)."""
        return AdaptiveSet.build(
            self.dim, self.points_per_dim, self.n_local_maxima, self.n_local, seed=self.seed
        )

    def new_af(self) -> NeuralAF:
        """A freshly initialized neural AF."""
        return NeuralAF(self.dim, self.use_x, self.hidden, seed=self.seed, n_weights=self.n_weights)


@dataclass(frozen=True)
class NAFFamily:
    """One MetaBO global-optimization family: its tasks, GP hyperparameters and candidate set.

    ``points_per_dim`` and ``n_local`` follow the reference's rule: the global grid has
    ``floor(N_MS ** (1 / dim))`` points per dimension and each local cloud has ``N_LS = N_MS``
    points, with ``N_MS`` = 1000 for Branin and Goldstein-Price and 2000 for Hartmann-3
    (the paper's Table 4).
    """

    name: str
    dim: int
    points_per_dim: int
    n_local: int
    gp_hyperparameters: FamilyGPHyperparameters
    sample_tasks: Callable[[int, int], list]
    grid_tasks: Callable[[int], list]

    def config(self, **overrides: object) -> NAFConfig:
        """A ``NAFConfig`` with this family's dimension and candidate set; other settings by keyword."""
        settings: dict[str, object] = dict(
            dim=self.dim, points_per_dim=self.points_per_dim, n_local=self.n_local
        )
        settings.update(overrides)
        return NAFConfig(**settings)  # type: ignore[arg-type]


NAF_FAMILY_SPECS: dict[str, NAFFamily] = {
    "branin": NAFFamily("branin", 2, 31, 1000, BRANIN_GP_HYPERPARAMETERS, sample_branin_tasks, branin_grid_tasks),
    "goldstein_price": NAFFamily(
        "goldstein_price", 2, 31, 1000, GOLDSTEIN_PRICE_GP_HYPERPARAMETERS,
        sample_goldstein_price_tasks, goldstein_price_grid_tasks,
    ),
    "hartmann3": NAFFamily(
        "hartmann3", 3, 12, 2000, HARTMANN3_GP_HYPERPARAMETERS, sample_hartmann3_tasks, hartmann3_grid_tasks
    ),
}


def get_naf_family(name: str) -> NAFFamily:
    """Look up a family by name."""
    if name not in NAF_FAMILY_SPECS:
        raise ValueError(f"Unknown family {name!r}; defined: {list(NAF_FAMILY_SPECS)}.")
    return NAF_FAMILY_SPECS[name]


def value_features(
    t: NDArray[np.float64] | float, budget: int, weights: NDArray[np.float64] | None = None
) -> torch.Tensor:
    """Inputs ``[t, T]`` of the value function ``V(t, T)`` (raw, as in the reference).

    A weight-conditioned AF appends each step's objective weights (``weights`` of shape ``(m, N)``):
    returns depend on them, and unlike the task they are known.
    """
    t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
    columns = [t_arr[:, None], np.full((t_arr.shape[0], 1), float(budget))]
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(t_arr.shape[0], -1)
        columns.append(w)
    return torch.as_tensor(np.hstack(columns), dtype=torch.float32)


@dataclass
class Rollout:
    """Steps from whole episodes, in order; ``features`` has one ``(n, F)`` array per step."""

    features: list[NDArray[np.float32]] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    final_regrets: list[float] = field(default_factory=list)
    # The objective weights the AF read at each step (empty arrays for a plain NAF).
    weights: list[NDArray[np.float64]] = field(default_factory=list)

    def extend(self, other: Rollout) -> None:
        """Append another rollout's steps."""
        self.features += other.features
        self.actions += other.actions
        self.log_probs += other.log_probs
        self.rewards += other.rewards
        self.steps += other.steps
        self.final_regrets += other.final_regrets
        self.weights += other.weights


def _log_softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - logits.max()
    return shifted - np.log(np.exp(shifted).sum())


def run_episodes(
    af: NeuralAF,
    tasks: Sequence[Task],
    gp_hyperparameters: FamilyGPHyperparameters,
    adaptive_set: AdaptiveSet,
    budget: int,
    rng: np.random.Generator,
    greedy: bool = False,
) -> Rollout:
    """Run one episode per task with ``af`` as the policy (sampled, or argmax when ``greedy``)."""
    rollout = Rollout()
    for task in tasks:
        episode = BOEpisode(task, gp_hyperparameters, adaptive_set, budget)
        weights = af_weights(af, getattr(task, "weights", None))
        step_weights = np.empty(0) if weights is None else weights
        while not episode.done:
            features, logits = episode.observation(af)
            log_p = _log_softmax(logits)
            index = int(np.argmax(logits)) if greedy else int(rng.choice(log_p.size, p=np.exp(log_p)))
            rollout.features.append(features.astype(np.float32))
            rollout.actions.append(index)
            rollout.log_probs.append(float(log_p[index]))
            rollout.steps.append(episode.t)
            rollout.weights.append(step_weights)
            rollout.rewards.append(episode.step(index))
        rollout.final_regrets.append(episode.regret)
    return rollout


def gae_advantages(
    rewards: NDArray[np.float64],
    values: NDArray[np.float64],
    budget: int,
    gamma: float,
    gae_lambda: float,
) -> NDArray[np.float64]:
    """Generalized advantage estimates for whole episodes of length ``budget``, back to back.

    Each episode ends at its last step, so nothing is bootstrapped past it.
    """
    n = rewards.shape[0]
    if n % budget != 0 or values.shape != rewards.shape:
        raise ValueError("rewards and values must hold whole episodes of the same length.")
    advantages = np.zeros(n, dtype=np.float64)
    for start in range(0, n, budget):
        running = 0.0
        for i in reversed(range(start, start + budget)):
            next_value = values[i + 1] if i + 1 < start + budget else 0.0
            delta = rewards[i] + gamma * next_value - values[i]
            running = delta + gamma * gae_lambda * running
            advantages[i] = running
    return advantages


def ppo_loss(
    logits: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    clip: float,
    value_coefficient: float,
    entropy_coefficient: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped PPO loss for one minibatch of categorical actions.

    ``logits`` is ``(B, n)`` (one row per step, over that step's candidates).
    Advantages are normalized within the minibatch, as in the reference; the
    value loss is a plain squared error.
    """
    log_p = torch.log_softmax(logits, dim=-1)
    new_log_probs = log_p.gather(1, actions[:, None]).squeeze(1)
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    ratio = torch.exp(new_log_probs - old_log_probs)
    policy_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages).mean()
    value_loss = ((values - returns) ** 2).mean()
    entropy = -(log_p.exp() * log_p).sum(dim=-1).mean()
    loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
    stats = {
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
    }
    return loss, stats


def _rollout_worker(
    payload: tuple[NAFConfig, dict[str, torch.Tensor], list[Task], FamilyGPHyperparameters, int],
) -> Rollout:
    """Run episodes in a worker process with one torch and one BLAS thread."""
    config, state, tasks, gp_hyperparameters, seed = payload
    torch.set_num_threads(1)
    af = config.new_af()
    af.load_state_dict(state)
    with threadpool_limits(limits=1):
        return run_episodes(af, tasks, gp_hyperparameters, config.adaptive_set(), config.budget, np.random.default_rng(seed))


@dataclass
class NAFTrainingResult:
    """A trained AF plus the training history (one entry per iteration)."""

    af: NeuralAF
    best_af: NeuralAF
    best_iteration: int
    mean_return: list[float]
    mean_final_log_regret: list[float]
    seconds: list[float]
    stats: list[dict[str, float]]


def train_naf(
    config: NAFConfig,
    gp_hyperparameters: FamilyGPHyperparameters,
    task_pool: Sequence[Task] | None = None,
    task_sampler: Callable[[int, int], Sequence[Task]] | None = None,
    checkpoint_dir: str | Path | None = None,
    log: Callable[[str], None] | None = None,
    resume: bool = False,
) -> NAFTrainingResult:
    """Train a neural AF with PPO.

    Each episode's task is drawn uniformly from ``task_pool`` (MetaBO-50), or
    ``task_sampler(n, seed)`` draws fresh tasks (full MetaBO). The returned
    ``best_af`` has the highest mean training return, the reference's
    checkpoint choice. With ``checkpoint_dir``, the final and best AFs, the
    history and the full training state are saved there after every
    iteration. ``resume=True`` continues from that state up to
    ``config.n_iterations``; only ``n_iterations`` may differ from the saved
    config, and the continuation matches an uninterrupted run.
    """
    if (task_pool is None) == (task_sampler is None):
        raise ValueError("Pass exactly one of task_pool and task_sampler.")
    if task_pool is not None and not task_pool:
        raise ValueError("task_pool must be non-empty.")
    rng = np.random.default_rng(config.seed)
    af = config.new_af()
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(config.seed + 1)
        value_net = mlp(2 + config.n_weights, config.hidden)
    optimizer = torch.optim.Adam(list(af.parameters()) + list(value_net.parameters()), lr=config.learning_rate)
    best_af = config.new_af()
    best_return, best_iteration = -np.inf, -1
    history = NAFTrainingResult(af, best_af, -1, [], [], [], [])
    first_iteration = 0
    if resume:
        if checkpoint_dir is None:
            raise ValueError("resume needs checkpoint_dir.")
        state = torch.load(Path(checkpoint_dir) / TRAINING_STATE_FILE, weights_only=False)
        saved = NAFConfig(**{**state["config"], "hidden": tuple(state["config"]["hidden"])})
        if replace(saved, n_iterations=config.n_iterations) != config:
            raise ValueError("Only n_iterations may change when resuming.")
        af.load_state_dict(state["af"])
        best_af.load_state_dict(state["best_af"])
        value_net.load_state_dict(state["value_net"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["rng"]
        for key in ("mean_return", "mean_final_log_regret", "seconds", "stats"):
            setattr(history, key, list(state["history"][key]))
        best_iteration = history.best_iteration = int(state["history"]["best_iteration"])
        best_return = history.mean_return[best_iteration] if best_iteration >= 0 else -np.inf
        first_iteration = len(history.mean_return)
    executor = ProcessPoolExecutor(config.n_workers) if config.n_workers > 1 else None
    try:
        for iteration in range(first_iteration, config.n_iterations):
            start = time.perf_counter()
            n_episodes = config.episodes_per_iteration
            if task_pool is not None:
                tasks = [task_pool[i] for i in rng.integers(len(task_pool), size=n_episodes)]
            else:
                assert task_sampler is not None
                tasks = list(task_sampler(n_episodes, int(rng.integers(2**31))))
            rollout = _collect(config, af, tasks, gp_hyperparameters, iteration, executor)
            stats = _ppo_update(config, af, value_net, optimizer, rollout, rng)

            returns = np.asarray(rollout.rewards).reshape(-1, config.budget).sum(axis=1)
            mean_return = float(returns.mean())
            history.mean_return.append(mean_return)
            history.mean_final_log_regret.append(float(np.mean(np.log10(np.maximum(rollout.final_regrets, 1e-20)))))
            history.seconds.append(time.perf_counter() - start)
            history.stats.append(stats)
            if mean_return > best_return:
                best_return, best_iteration = mean_return, iteration
                best_af.load_state_dict(af.state_dict())
            history.best_iteration = best_iteration
            if checkpoint_dir is not None:
                save_training(checkpoint_dir, config, gp_hyperparameters, history)
                torch.save(
                    {
                        "config": asdict(config),
                        "af": af.state_dict(),
                        "best_af": best_af.state_dict(),
                        "value_net": value_net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rng": rng.bit_generator.state,
                        "history": _history_record(history),
                    },
                    Path(checkpoint_dir) / TRAINING_STATE_FILE,
                )
            if log is not None:
                log(
                    f"iter {iteration}: return={mean_return:.3f} "
                    f"final_log10_regret={history.mean_final_log_regret[-1]:.3f} "
                    f"entropy={stats['entropy']:.3f} seconds={history.seconds[-1]:.1f}"
                )
    finally:
        if executor is not None:
            executor.shutdown()
    return history


def _collect(
    config: NAFConfig,
    af: NeuralAF,
    tasks: list[Task],
    gp_hyperparameters: FamilyGPHyperparameters,
    iteration: int,
    executor: ProcessPoolExecutor | None,
) -> Rollout:
    """Run this iteration's episodes, split over the workers; results are in task order."""
    base_seed = config.seed + 1009 * (iteration + 1)
    if executor is None:
        return run_episodes(
            af, tasks, gp_hyperparameters, config.adaptive_set(), config.budget, np.random.default_rng(base_seed)
        )
    state = {k: v.detach().clone() for k, v in af.state_dict().items()}
    chunks = [tasks[w :: config.n_workers] for w in range(config.n_workers)]
    payloads = [(config, state, chunk, gp_hyperparameters, base_seed + w) for w, chunk in enumerate(chunks) if chunk]
    rollout = Rollout()
    for part in executor.map(_rollout_worker, payloads):
        rollout.extend(part)
    return rollout


def _flush_denormal_gradient(grad: torch.Tensor) -> torch.Tensor:
    """Zero the denormal entries of a gradient, as a CPU's flush-to-zero mode would."""
    return torch.where(grad.abs() < torch.finfo(grad.dtype).tiny, torch.zeros_like(grad), grad)


def _ppo_update(
    config: NAFConfig,
    af: NeuralAF,
    value_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Run ``epochs`` passes of minibatch PPO updates over one rollout.

    Denormal floats are flushed to zero during the update. A trained policy
    puts logits far apart (spreads up to about 300 on Branin), so some
    softmax probabilities, and the gradients the softmax passes back into the
    network, are float32 denormals, which the CPU multiplies very slowly.
    On Branin this took iterations from about 55 s to 140 s
    (``docs/plans/naf.md``, question 10). ``torch.set_flush_denormal`` only
    covers the calling thread, so the gradient entering the network is
    also flushed explicitly, for the intra-op worker threads. Rollouts are
    unaffected and keep the default.
    """
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.update_threads)
    torch.set_flush_denormal(True)
    try:
        steps = np.asarray(rollout.steps, dtype=np.float64)
        step_weights = np.stack(rollout.weights) if config.n_weights else None
        value_in = value_features(steps, config.budget, step_weights)
        with torch.no_grad():
            values = value_net(value_in).squeeze(-1).double().numpy()
        rewards = np.asarray(rollout.rewards, dtype=np.float64)
        advantages = gae_advantages(rewards, values, config.budget, config.gamma, config.gae_lambda)
        returns = advantages + values
        features = torch.as_tensor(np.stack(rollout.features))
        actions = torch.as_tensor(rollout.actions, dtype=torch.long)
        old_log_probs = torch.as_tensor(rollout.log_probs, dtype=torch.float32)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32)
        ret_t = torch.as_tensor(returns, dtype=torch.float32)
        n = len(rollout.actions)
        totals: dict[str, float] = {}
        count = 0
        for _ in range(config.epochs):
            order = rng.permutation(n)
            for start in range(0, n, config.minibatch_size):
                idx = torch.as_tensor(order[start : start + config.minibatch_size])
                logits = af(features[idx])
                logits.register_hook(_flush_denormal_gradient)
                loss, stats = ppo_loss(
                    logits,
                    actions[idx],
                    old_log_probs[idx],
                    adv_t[idx],
                    value_net(value_in[idx]).squeeze(-1),
                    ret_t[idx],
                    config.clip,
                    config.value_coefficient,
                    config.entropy_coefficient,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                for key, value in stats.items():
                    totals[key] = totals.get(key, 0.0) + value
                count += 1
        return {key: value / count for key, value in totals.items()}
    finally:
        torch.set_flush_denormal(False)
        torch.set_num_threads(previous_threads)


def save_naf(directory: str | Path, af: NeuralAF, config: NAFConfig, gp_hyperparameters: FamilyGPHyperparameters, name: str = "af") -> None:
    """Save an AF as ``<name>.pt`` plus ``naf_config.json`` (settings and GP hyperparameters)."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.save(af.state_dict(), root / f"{name}.pt")
    payload = {"config": asdict(config), "gp_hyperparameters": gp_hyperparameters.to_dict()}
    (root / "naf_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_naf(directory: str | Path, name: str = "af") -> tuple[NeuralAF, NAFConfig, FamilyGPHyperparameters]:
    """Load an AF saved by :func:`save_naf`."""
    root = Path(directory)
    payload = json.loads((root / "naf_config.json").read_text(encoding="utf-8"))
    raw = payload["config"]
    raw["hidden"] = tuple(raw["hidden"])
    config = NAFConfig(**raw)
    af = config.new_af()
    af.load_state_dict(torch.load(root / f"{name}.pt", weights_only=True))
    return af, config, FamilyGPHyperparameters.from_dict(payload["gp_hyperparameters"])


def save_training(
    directory: str | Path,
    config: NAFConfig,
    gp_hyperparameters: FamilyGPHyperparameters,
    history: NAFTrainingResult,
) -> None:
    """Save the final AF (``af.pt``), the best one (``best_af.pt``) and ``training_history.json``."""
    save_naf(directory, history.af, config, gp_hyperparameters, name="af")
    save_naf(directory, history.best_af, config, gp_hyperparameters, name="best_af")
    (Path(directory) / "training_history.json").write_text(json.dumps(_history_record(history)), encoding="utf-8")


def _history_record(history: NAFTrainingResult) -> dict[str, object]:
    return {
        "best_iteration": history.best_iteration,
        "mean_return": history.mean_return,
        "mean_final_log_regret": history.mean_final_log_regret,
        "seconds": history.seconds,
        "stats": history.stats,
    }


class NAFSequentialOptimizer:
    """Ask/tell BO with a trained neural AF (test time: greedy, the AF's argmax over ``xi_t``).

    The GP keeps the family's fixed hyperparameters. ``budget`` is the episode
    length ``T`` the AF was trained with; it is one of the AF's inputs, so a
    different budget feeds the AF values it never saw. A weight-conditioned
    AF needs the objective ``weights`` (on the simplex); a plain AF
    ignores them. With an ``rng``, each point is instead drawn from the AF's
    softmax over ``xi_t``, as during training (not MetaBO's test protocol).
    """

    def __init__(
        self,
        af: NeuralAF,
        gp_hyperparameters: FamilyGPHyperparameters,
        budget: int,
        adaptive_set: AdaptiveSet,
        weights: Sequence[float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if budget < 1:
            raise ValueError("budget must be positive.")
        dim = adaptive_set.global_grid.shape[1]
        if dim != af.dim:
            raise ValueError("The adaptive set's dimension does not match the AF's.")
        self.weights = af_weights(af, weights)
        if self.weights is not None:
            validate_weights(self.weights, af.n_weights)
        self.af = af
        self.gp = gp_hyperparameters.new_gp()
        self.budget = budget
        self.adaptive_set = adaptive_set
        self.rng = rng
        self.x_obs = np.empty((0, dim), dtype=np.float64)
        self.y_obs = np.empty(0, dtype=np.float64)

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        name: str = "best_af",
        weights: Sequence[float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> NAFSequentialOptimizer:
        """Load a trained AF (``best_af`` by default) with the settings it was trained with."""
        af, config, gp_hyperparameters = load_naf(directory, name)
        af.eval()
        return cls(af, gp_hyperparameters, config.budget, config.adaptive_set(), weights, rng)

    @property
    def t(self) -> int:
        """Evaluations so far."""
        return int(self.y_obs.shape[0])

    def suggest(self) -> NDArray[np.float64]:
        """The next point, shape ``(1, d)``."""
        if self.t >= self.budget:
            raise RuntimeError(f"The budget of {self.budget} evaluations is used up.")
        score = naf_score_fn(self.af, self.gp, self.t, self.budget, self.weights)
        if self.rng is not None:
            return self.adaptive_set.sample(score, self.rng)
        return self.adaptive_set.maximize(score)

    def observe(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """Record evaluations ``x`` ``(n, d)`` and ``y`` ``(n,)``, then refit the GP."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.shape != (y.shape[0], self.x_obs.shape[1]):
            raise ValueError("x must have shape (n, d) and y shape (n,).")
        self.x_obs = np.vstack([self.x_obs, x])
        self.y_obs = np.append(self.y_obs, y)
        self.gp.fit(self.x_obs, self.y_obs)


def run_naf(
    objective: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    optimizer: NAFSequentialOptimizer,
    n_iter: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run ``n_iter`` evaluations (default: the AF's budget); return ``x`` and ``y``."""
    for _ in range(optimizer.budget if n_iter is None else n_iter):
        x = optimizer.suggest()
        optimizer.observe(x, np.asarray(objective(x), dtype=np.float64).reshape(-1))
    return optimizer.x_obs.copy(), optimizer.y_obs.copy()
