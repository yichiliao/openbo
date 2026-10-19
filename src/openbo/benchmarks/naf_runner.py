"""MetaBO benchmark runner (Volpp et al., ICLR 2020, Fig. 2; plan ``docs/plans/naf.md``).

Methods, all on the same test tasks:

- ``naf``: a trained neural AF (``best_af`` of a training run), greedy.
- ``ei``: EI, with one scrambled Sobol point as its initial design. The paper
  (App. B.1) and the reference code both start EI from the domain midpoint
  ``[0.5, 0.5]``; this runner deviates (see ``docs/review/naf.md``, H.1: the
  midpoint changes EI's curve up to about evaluation 20, not its value at 30).
- ``taf_me_50`` / ``taf_r_50``: the original TAF with the 50 MetaBO-50 tasks as
  sources (100 Sobol points each), i.e. ``taf_plus_acquisition`` with one
  objective and no decay, with TAF-ME (= TAF-PoE) or TAF-R weights.
- ``random``: uniform random points.

As in the reference, every GP (target and sources) uses the family's fixed
hyperparameters, and every AF is maximized over the same candidate set
(``AdaptiveSet``: 31 x 31 grid plus local clouds). One deviation: before any
data, the reference's TAF uses the task's own corner value as its incumbent
(privileged information); here TAF's first point maximizes the sources'
product-of-experts mean instead, as in TAF+.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

from openbo.acquisition.ei import expected_improvement_maximization
from openbo.acquisition.naf import AdaptiveSet, gp_mean_std
from openbo.acquisition.taf_plus import (
    population_mean,
    source_references,
    taf_plus_acquisition,
    taf_r_weights,
)
from openbo.models.gp_scratch import GPScratch
from openbo.optimizers.naf import (
    NAF_FAMILY_SPECS,
    FamilyGPHyperparameters,
    NAFFamily,
    NAFSequentialOptimizer,
    Task,
    get_naf_family,
)

NAF_METHODS = ("naf", "ei", "taf_me_50", "taf_r_50", "random")
NAF_FAMILIES = tuple(NAF_FAMILY_SPECS)
N_TAF_SOURCES = 50
N_TAF_SOURCE_POINTS = 100

ScoreFactory = Callable[[GPScratch, NDArray[np.float64], NDArray[np.float64]], Callable[[NDArray[np.float64]], NDArray[np.float64]]]


def naf_test_tasks(family: str, n_tasks: int = 100, seed: int = 100) -> list[Task]:
    """Unseen test tasks of a family (random translations and scalings)."""
    return list(get_naf_family(family).sample_tasks(n_tasks, seed))


def family_gp_hyperparameters(family: str) -> FamilyGPHyperparameters:
    """The fixed GP hyperparameters of a family."""
    return get_naf_family(family).gp_hyperparameters


def family_adaptive_set(family: NAFFamily, seed: int = 0) -> AdaptiveSet:
    """The family's candidate set (global grid plus local clouds), as in training."""
    return AdaptiveSet.build(family.dim, family.points_per_dim, n_local=family.n_local, seed=seed)


class TAFSources:
    """The TAF baselines' source GPs: one fixed-hyperparameter GP per MetaBO-50 task."""

    def __init__(self, family: str, adaptive_set: AdaptiveSet, seed: int = 0) -> None:
        hp = family_gp_hyperparameters(family)
        dim = adaptive_set.global_grid.shape[1]
        x = qmc.Sobol(d=dim, scramble=True, seed=seed).random(N_TAF_SOURCE_POINTS)
        self.gps = []
        for task in get_naf_family(family).grid_tasks(N_TAF_SOURCES):
            gp = hp.new_gp()
            gp.fit(x, np.asarray(task(x), dtype=np.float64))
            self.gps.append(gp)
        self._grid = adaptive_set.global_grid
        self._grid_prediction = self._predict(self._grid)

    def _predict(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        means, variances = zip(*(gp.posterior(x) for gp in self.gps))
        return np.stack(means)[:, :, None], np.stack(variances)[:, :, None]

    def predict(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Source means and variances at ``x``, shape ``(M, n, 1)`` (the global grid is cached)."""
        if x is self._grid:
            return self._grid_prediction
        return self._predict(x)


def ei_score_factory(gp: GPScratch, x_obs: NDArray[np.float64], y_obs: NDArray[np.float64]) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    best = float(np.max(y_obs))

    def score(x: NDArray[np.float64]) -> NDArray[np.float64]:
        mean, std = gp_mean_std(gp, x)
        return expected_improvement_maximization(mean, std**2, best)

    return score


def _taf_scores(sources: TAFSources, weight_mode: str) -> ScoreFactory:
    def factory(gp: GPScratch, x_obs: NDArray[np.float64], y_obs: NDArray[np.float64]) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        if y_obs.size == 0:
            # No data: maximize the sources' product-of-experts mean (TAF+'s choice).
            def first(x: NDArray[np.float64]) -> NDArray[np.float64]:
                mean, var = sources.predict(x)
                return population_mean(mean, var, [1.0], weight_mode)

            return first
        mean_at_target, _ = sources._predict(x_obs)
        references = source_references(mean_at_target)
        ranking = taf_r_weights(y_obs[:, None], mean_at_target) if weight_mode == "taf_r" else None
        best = np.array([np.max(y_obs)])

        def score(x: NDArray[np.float64]) -> NDArray[np.float64]:
            target_mean, target_std = gp_mean_std(gp, x)
            source_mean, source_var = sources.predict(x)
            return taf_plus_acquisition(
                target_mean[:, None],
                (target_std**2)[:, None],
                best,
                source_mean,
                source_var,
                references,
                [1.0],
                decay=1.0,
                weight_mode=weight_mode,
                ranking_weights=ranking,
            )

        return score

    return factory


def grid_bo(
    task: Task,
    score_factory: ScoreFactory,
    gp_hyperparameters: FamilyGPHyperparameters,
    adaptive_set: AdaptiveSet,
    budget: int,
    x_init: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """BO with a fixed-hyperparameter GP and an AF maximized over ``adaptive_set``."""
    gp = gp_hyperparameters.new_gp()
    dim = adaptive_set.global_grid.shape[1]
    x_obs = np.empty((0, dim))
    y_obs = np.empty(0)
    for _ in range(budget):
        if x_obs.shape[0] < x_init.shape[0]:
            x = x_init[x_obs.shape[0]][None, :]
        else:
            x = adaptive_set.maximize(score_factory(gp, x_obs, y_obs))
        x_obs = np.vstack([x_obs, x])
        y_obs = np.append(y_obs, float(np.asarray(task(x)).reshape(-1)[0]))
        gp.fit(x_obs, y_obs)
    return x_obs, y_obs


def run_naf_task(
    method: str,
    task: Task,
    task_index: int,
    family: str = "branin",
    af_dir: str | Path | None = None,
    budget: int = 30,
    seed: int = 0,
    sources: TAFSources | None = None,
) -> dict[str, object]:
    """Run one method on one task; return the points, values and simple regret per step."""
    if method not in NAF_METHODS:
        raise ValueError(f"method must be one of {list(NAF_METHODS)}.")
    rng = np.random.default_rng(seed + 1009 * task_index)
    if method == "naf":
        if af_dir is None:
            raise ValueError("naf needs af_dir, a training run's output folder.")
        optimizer = NAFSequentialOptimizer.from_directory(af_dir)
        if optimizer.budget != budget:
            raise ValueError(f"The AF was trained with budget {optimizer.budget}, not {budget}.")
        for _ in range(budget):
            x = optimizer.suggest()
            optimizer.observe(x, np.asarray(task(x)).reshape(-1))
        x_obs, y_obs = optimizer.x_obs, optimizer.y_obs
    else:
        spec = get_naf_family(family)
        hp = spec.gp_hyperparameters
        adaptive_set = family_adaptive_set(spec, seed)
        if method == "random":
            x_obs = rng.random((budget, spec.dim))
            y_obs = np.asarray(task(x_obs), dtype=np.float64)
        elif method == "ei":
            x_init = qmc.Sobol(d=spec.dim, scramble=True, seed=seed + 1009 * task_index).random(1)
            x_obs, y_obs = grid_bo(task, ei_score_factory, hp, adaptive_set, budget, x_init)
        else:
            if sources is None:
                sources = TAFSources(family, adaptive_set, seed=seed)
            mode = "poe" if method == "taf_me_50" else "taf_r"
            x_obs, y_obs = grid_bo(task, _taf_scores(sources, mode), hp, adaptive_set, budget, np.empty((0, spec.dim)))
    regret = task.optimum - np.maximum.accumulate(y_obs)
    return {
        "task_index": task_index,
        "optimum": float(task.optimum),
        "x": np.asarray(x_obs).tolist(),
        "y": np.asarray(y_obs).tolist(),
        "regret": regret.tolist(),
    }


def _task_worker(payload: tuple[str, Task, int, str, str | None, int, int]) -> dict[str, object]:
    # One thread per worker: with n workers each running a multithreaded BLAS,
    # the CPU is oversubscribed (the TAF baselines ran several times slower).
    torch.set_num_threads(1)
    method, task, index, family, af_dir, budget, seed = payload
    with threadpool_limits(limits=1):
        return run_naf_task(method, task, index, family, af_dir, budget, seed)


def check_af_matches_family(af_dir: str | Path, family: str) -> None:
    """Refuse a trained AF whose input dimension is not the family's, warn if its GP is another family's.

    ``family`` sets the test tasks, the GP and the baselines' candidate set, but the AF carries
    its own settings, so a Goldstein-Price AF run with ``family="branin"`` (both 2D) would
    silently be tested on Branin tasks. The dimension must match (an error); a different GP
    (hyperparameters) is only a warning, since an AF trained with hyperparameters of your own
    is legitimate.
    """
    spec = get_naf_family(family)
    saved = json.loads((Path(af_dir) / "naf_config.json").read_text(encoding="utf-8"))
    dim = int(saved["config"]["dim"])
    stored = saved["gp_hyperparameters"]
    belongs_to = [
        name
        for name, other in NAF_FAMILY_SPECS.items()
        if other.dim == dim
        and np.allclose(stored["lengthscale"], other.gp_hyperparameters.lengthscale, rtol=1e-6)
        and np.isclose(stored["variance"], other.gp_hyperparameters.variance, rtol=1e-6)
    ]
    hint = f" Its GP hyperparameters are those of {belongs_to[0]!r}." if belongs_to else ""
    if dim != spec.dim:
        raise ValueError(
            f"The AF in {af_dir} takes {dim}D inputs, but family {family!r} is {spec.dim}D. "
            f"Pass --family for the family it was trained on.{hint}"
        )
    if family not in belongs_to:
        warnings.warn(
            f"The AF in {af_dir} was not trained with the GP hyperparameters of family {family!r}, so it "
            f"probably belongs to another family; the test tasks and the baselines use {family!r}.{hint} "
            "Pass --family for the family it was trained on.",
            UserWarning,
            stacklevel=2,
        )


def run_naf_method(
    method: str,
    family: str = "branin",
    n_tasks: int = 100,
    af_dir: str | Path | None = None,
    budget: int = 30,
    seed: int = 0,
    test_seed: int = 100,
    n_workers: int = 1,
) -> dict[str, object]:
    """Run one method on ``n_tasks`` unseen test tasks; results are in task order."""
    if method == "naf" and af_dir is not None:
        check_af_matches_family(af_dir, family)
    tasks = naf_test_tasks(family, n_tasks, test_seed)
    af = None if af_dir is None else str(af_dir)
    if n_workers > 1:
        payloads = [(method, task, i, family, af, budget, seed) for i, task in enumerate(tasks)]
        with ProcessPoolExecutor(n_workers) as executor:
            records = list(executor.map(_task_worker, payloads))
    else:
        sources = TAFSources(family, family_adaptive_set(get_naf_family(family), seed), seed=seed) if method.startswith("taf") else None
        records = [run_naf_task(method, task, i, family, af, budget, seed, sources) for i, task in enumerate(tasks)]
    return {
        "method": method,
        "family": family,
        "budget": budget,
        "seed": seed,
        "test_seed": test_seed,
        "af_dir": af,
        "tasks": records,
    }
