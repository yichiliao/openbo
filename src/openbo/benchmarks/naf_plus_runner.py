"""NAF+ benchmark runner (Liao et al., CHI 2026, App. B; plan ``docs/plans/naf_plus.md``).

Tests (synthetic Test 1, the double sphere):

- ``double_sphere``: unseen users from the training distribution (Test 1.1).
- ``double_sphere_novel``: users with the centres moved outward (Test 1.2).

Every test user has random objective weights. Methods, all on the same users:

- ``naf_plus``: a trained NAF+ run (weight-conditioned AF, detector, EI fallback).
- ``naf_plus_no_w``: the "without weights" ablation, a run trained with
  ``n_weights = 0`` (the AF never sees the weights), with the fallback.
- ``naf_plus_no_ei``: the NAF+ run's AF without the fallback (no detector, no EI).
- ``naf_plus_sample``, ``naf_plus_no_w_sample`` and ``naf_plus_no_ei_sample``:
  ``naf_plus``, ``naf_plus_no_w`` and ``naf_plus_no_ei`` with the AF's points drawn from its softmax, as in
  training, instead of its argmax (MetaBO's and the paper's test protocol).
  Added because the greedy AF repeats points on this family (plan, Phase 5).
- ``naf_plus_norepeat``, ``naf_plus_no_w_norepeat`` and
  ``naf_plus_no_ei_norepeat``: the greedy methods, but never suggesting a
  point already evaluated (``avoid_repeats``; plan, "Design stance" item 1).
- ``naf_plus_mix``: ``naf_plus``'s mean-p-value test, but EI and the AF mixed
  as policies (a coin with probability ``lambda_EI`` picks EI's point) instead
  of eq. 4's score mix (plan, "Design stance" item 2).
- ``naf_plus_cal`` and ``naf_plus_cal_sample``: the calibrated novelty test
  (``S_n = sum z^2`` against simulated training-like users, ``tau = 0.05``;
  item 3) with the policy mix; greedy, and with the AF's points sampled. They
  need ``novelty_calibration.json`` in the run (``scripts/calibrate_naf_plus_novelty.py``).
- ``ei``: EI with one scrambled Sobol point first, as in the NAF benchmark.
- ``taf_plus_15``: TAF+ (``taf_plus_acquisition``, product-of-experts weights)
  with 15 earlier users as sources, one GP per objective each, the test
  user's weights and the paper's decay ``alpha_1 = 4``, ``alpha_2 = 0.2``. The
  target observes the per-objective values. Its first point maximizes the
  sources' product-of-experts mean, as in TAF+.
- ``random``: uniform random points.

As in the NAF benchmark, every GP uses the family's fixed hyperparameters and
every AF is maximized over the same candidate set. Two deviations from the
paper's baselines, both to keep the comparison inside MetaBO's protocol: EI
starts from one Sobol point (the paper's BO: 6 random points), and the TAF+
sources are fitted on 25 Sobol points per user (the paper's: 10 random plus 15
BO steps, also 25).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

from openbo.acquisition.naf import AdaptiveSet, gp_mean_std
from openbo.acquisition.taf_plus import (
    decay_factor,
    population_mean,
    source_references,
    taf_plus_acquisition,
)
from openbo.benchmarks.naf_runner import ei_score_factory, grid_bo
from openbo.models.gp_scratch import GPScratch
from openbo.optimizers.naf import FamilyGPHyperparameters
from openbo.optimizers.naf_plus import (
    DOUBLE_SPHERE_BUDGET,
    NAFPlusConfig,
    NAFPlusSequentialOptimizer,
)
from openbo.test_functions.naf_plus_families import WeightedTask

NAF_PLUS_METHODS = (
    "naf_plus",
    "naf_plus_no_w",
    "naf_plus_no_ei",
    "naf_plus_sample",
    "naf_plus_no_w_sample",
    "naf_plus_no_ei_sample",
    "naf_plus_norepeat",
    "naf_plus_no_w_norepeat",
    "naf_plus_no_ei_norepeat",
    "naf_plus_mix",
    "naf_plus_cal",
    "naf_plus_cal_sample",
    "ei",
    "taf_plus_15",
    "random",
)
NAF_PLUS_TESTS = ("double_sphere", "double_sphere_novel")
N_TAF_SOURCES = 15
N_TAF_SOURCE_POINTS = 25
TAF_SOURCE_SEED = 7
TAF_DECAY = (4, 0.2)


def naf_plus_test_tasks(
    test: str, plus_config: NAFPlusConfig | None = None, n_tasks: int = 100, seed: int = 100
) -> list[WeightedTask]:
    """Unseen test users of a test, each with random weights."""
    if test not in NAF_PLUS_TESTS:
        raise ValueError(f"test must be one of {list(NAF_PLUS_TESTS)}.")
    config = NAFPlusConfig() if plus_config is None else plus_config
    return config.tasks(n_tasks, seed, novel=test.endswith("_novel"))


class TAFPlusSources:
    """The TAF+ baseline's sources: one fixed-hyperparameter GP per objective of each earlier user."""

    def __init__(self, plus_config: NAFPlusConfig, adaptive_set: AdaptiveSet, seed: int = 0) -> None:
        hp = plus_config.gp_hyperparameters
        x = qmc.Sobol(d=plus_config.dim, scramble=True, seed=seed).random(N_TAF_SOURCE_POINTS)
        self.gps: list[list[GPScratch]] = []
        for user in plus_config.users(N_TAF_SOURCES, TAF_SOURCE_SEED + seed):
            values = user.objectives(x)
            per_objective = []
            for k in range(values.shape[1]):
                gp = hp.new_gp()
                gp.fit(x, values[:, k])
                per_objective.append(gp)
            self.gps.append(per_objective)
        self._grid = adaptive_set.global_grid
        self._grid_prediction = self._predict(self._grid)

    def _predict(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        means = np.empty((len(self.gps), x.shape[0], len(self.gps[0])))
        variances = np.empty_like(means)
        for j, per_objective in enumerate(self.gps):
            for k, gp in enumerate(per_objective):
                means[j, :, k], variances[j, :, k] = gp.posterior(x)
        return means, variances

    def predict(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Source means and variances at ``x``, shape ``(M, n, L)`` (the global grid is cached)."""
        if x is self._grid:
            return self._grid_prediction
        return self._predict(x)


def _taf_plus_bo(
    task: WeightedTask,
    sources: TAFPlusSources,
    hp: FamilyGPHyperparameters,
    adaptive_set: AdaptiveSet,
    budget: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """TAF+ with per-objective target GPs; returns the points and the scalarized values."""
    weights = np.asarray(task.weights)
    n_obj = weights.shape[0]
    dim = adaptive_set.global_grid.shape[1]
    gps = [hp.new_gp() for _ in range(n_obj)]
    x_obs = np.empty((0, dim))
    f_obs = np.empty((0, n_obj))
    for _ in range(budget):
        if x_obs.shape[0] == 0:

            def score(x: NDArray[np.float64]) -> NDArray[np.float64]:
                mean, var = sources.predict(x)
                return population_mean(mean, var, weights)

        else:
            mean_at_target, _ = sources.predict(x_obs)
            references = source_references(mean_at_target)
            best = f_obs.max(axis=0)
            decay = decay_factor(x_obs.shape[0] + 1, *TAF_DECAY)

            def score(x: NDArray[np.float64]) -> NDArray[np.float64]:
                stats = [gp_mean_std(gp, x) for gp in gps]
                target_mean = np.stack([m for m, _ in stats], axis=1)
                target_var = np.stack([s**2 for _, s in stats], axis=1)
                source_mean, source_var = sources.predict(x)
                return taf_plus_acquisition(
                    target_mean, target_var, best, source_mean, source_var, references, weights, decay
                )

        x = adaptive_set.maximize(score)
        x_obs = np.vstack([x_obs, x])
        f_obs = np.vstack([f_obs, task.user.objectives(x)])
        for k, gp in enumerate(gps):
            gp.fit(x_obs, f_obs[:, k])
    return x_obs, f_obs @ weights


def run_naf_plus_task(
    method: str,
    task: WeightedTask,
    task_index: int,
    run_dir: str | Path | None = None,
    run_dir_no_w: str | Path | None = None,
    budget: int = DOUBLE_SPHERE_BUDGET,
    seed: int = 0,
    plus_config: NAFPlusConfig | None = None,
    sources: TAFPlusSources | None = None,
) -> dict[str, object]:
    """Run one method on one test user; return the points, values, simple regret and fallback history."""
    if method not in NAF_PLUS_METHODS:
        raise ValueError(f"method must be one of {list(NAF_PLUS_METHODS)}.")
    config = NAFPlusConfig() if plus_config is None else plus_config
    rng = np.random.default_rng(seed + 1009 * task_index)
    history: list[dict[str, object]] | None = None
    if method.startswith("naf_plus"):
        no_w = method.startswith("naf_plus_no_w")
        directory = run_dir_no_w if no_w else run_dir
        if directory is None:
            flag = "run_dir_no_w" if no_w else "run_dir"
            raise ValueError(f"{method} needs {flag}, a train_naf_plus output folder.")
        optimizer = NAFPlusSequentialOptimizer.from_run(
            directory,
            task.weights,
            use_fallback=not method.startswith("naf_plus_no_ei"),
            rng=rng if method.endswith("_sample") else None,
            avoid_repeats=method.endswith("_norepeat"),
            novelty_test="calibrated" if method.startswith("naf_plus_cal") else "mean_p",
            mix="policy" if method.startswith(("naf_plus_mix", "naf_plus_cal")) else "score",
            # A separate stream, so the coin does not shift the sampled AF's draws.
            mix_rng=np.random.default_rng([seed, task_index, 1]),
        )
        if (optimizer.af.n_weights == 0) != no_w:
            raise ValueError(f"{directory} does not hold the AF {method} needs (n_weights={optimizer.af.n_weights}).")
        if optimizer.budget != budget:
            raise ValueError(f"The AF was trained with budget {optimizer.budget}, not {budget}.")
        for _ in range(budget):
            x = optimizer.suggest()
            optimizer.observe(x, task(x))
        x_obs, y_obs = optimizer.x_obs, optimizer.y_obs
        history = [
            {"t": h.t, "p_bar": h.p_bar, "lambda_ei": h.lambda_ei, "novelty_p": h.novelty_p, "source": h.source}
            for h in optimizer.history
        ]
    else:
        hp = config.gp_hyperparameters
        adaptive_set = AdaptiveSet.build(config.dim, seed=seed)
        if method == "random":
            x_obs = rng.random((budget, config.dim))
            y_obs = task(x_obs)
        elif method == "ei":
            x_init = qmc.Sobol(d=config.dim, scramble=True, seed=seed + 1009 * task_index).random(1)
            x_obs, y_obs = grid_bo(task, ei_score_factory, hp, adaptive_set, budget, x_init)
        else:
            if sources is None:
                sources = TAFPlusSources(config, adaptive_set, seed=seed)
            x_obs, y_obs = _taf_plus_bo(task, sources, hp, adaptive_set, budget)
    regret = task.optimum - np.maximum.accumulate(np.asarray(y_obs))
    record: dict[str, object] = {
        "task_index": task_index,
        "weights": list(task.weights),
        "optimum": float(task.optimum),
        "x": np.asarray(x_obs).tolist(),
        "y": np.asarray(y_obs).tolist(),
        "regret": regret.tolist(),
    }
    if history is not None:
        record["history"] = history
    return record


def _task_worker(payload: tuple) -> dict[str, object]:  # type: ignore[type-arg]
    # One thread per worker, as in the NAF runner (multithreaded BLAS in every
    # worker oversubscribes the CPU).
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        return run_naf_plus_task(*payload)


def run_naf_plus_method(
    method: str,
    test: str = "double_sphere",
    n_tasks: int = 100,
    run_dir: str | Path | None = None,
    run_dir_no_w: str | Path | None = None,
    budget: int = DOUBLE_SPHERE_BUDGET,
    seed: int = 0,
    test_seed: int = 100,
    n_workers: int = 1,
    plus_config: NAFPlusConfig | None = None,
) -> dict[str, object]:
    """Run one method on ``n_tasks`` unseen test users; results are in task order.

    The family settings come from ``plus_config``, else from ``run_dir``'s
    ``naf_plus_config.json``, else the defaults.
    """
    if plus_config is None and run_dir is not None:
        plus_config = NAFPlusConfig.load(run_dir)
    config = NAFPlusConfig() if plus_config is None else plus_config
    tasks = naf_plus_test_tasks(test, config, n_tasks, test_seed)
    dirs = tuple(None if d is None else str(d) for d in (run_dir, run_dir_no_w))
    if n_workers > 1:
        payloads = [(method, task, i, *dirs, budget, seed, config) for i, task in enumerate(tasks)]
        with ProcessPoolExecutor(n_workers) as executor:
            records = list(executor.map(_task_worker, payloads))
    else:
        sources = (
            TAFPlusSources(config, AdaptiveSet.build(config.dim, seed=seed), seed=seed)
            if method == "taf_plus_15"
            else None
        )
        records = [run_naf_plus_task(method, task, i, *dirs, budget, seed, config, sources) for i, task in enumerate(tasks)]
    return {
        "method": method,
        "test": test,
        "budget": budget,
        "seed": seed,
        "test_seed": test_seed,
        "run_dir": dirs[0],
        "run_dir_no_w": dirs[1],
        "naf_plus_config": {k: v for k, v in vars(config).items()},
        "tasks": records,
    }

