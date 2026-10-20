"""Simulations for TAF+ (Liao et al., CHI 2024, App. B), on the sphere users.

Each simulation samples 20 users from one similarity range: 10 population
users and 10 new users. Population modeling runs multi-objective BO on each
population user (20 random points, then 20 qLogNEHVI iterations) and fits one
GP per objective to their data. Every method then runs on the same new users.

Kept out of ``runner.py`` like MOBO and ConBO: the score is a weighted sum of
several objectives, and a run is a population plus a set of new users.

Methods (all scored on the noise-free weighted sum ``f(x) @ alpha``: the
regret of the design at each iteration, ``instant_regret``, which is what a
user experiences, and of the best design so far, ``regret``):
- ``taf_plus``: TAF+ with decay ``(d1, d2)``. Its first point maximizes the
  population's product-of-experts mean, not the domain centre, which sits
  close to every user's optimum here.
- ``taf``: TAF+ without decay, the paper's "TAF (multi-objective)" baseline.
- ``bo``: scratch BO with EI on the noisy weighted sum, ``bo_init`` random starts.
- ``random``: uniform random points.
- ``taf_ehvi``: TAF-EHVI (``optimizers/mobo_taf.py``) with the population's
  users as sources, converted by ``write_population_sources``. It optimizes
  hypervolume and never sees ``alpha``; ``mobo_init`` random starts (it needs
  at least one), then qLogNEHVI blended with the sources' improvements, with
  the TAF-EHVI defaults (objective-wise TAF-R weights, no warmup, no decay).
- ``mobo``: plain qLogNEHVI with the same starts and settings, TAF-EHVI
  without sources.

``hypervolume_curve`` and ``pareto_hypervolume`` score any run on the
noise-free hypervolume of the designs evaluated so far, against the user's
true Pareto front.

Deviations from the paper: qLogNEHVI instead of qEHVI, with 256 raw samples
and 4 restarts instead of 1024 and 10 (the hypervolume reached was about the
same, at less than half the time); TAF+ searches a fresh 256-point Sobol pool
each step (the paper: 256 random points) and polishes the best candidate with
L-BFGS-B, which the paper does not (without it, the pool's resolution in 4D
capped TAF+ at a regret of about 0.1, while its first point reached 0.003 with
it); BO uses 5 random starts (the paper's user
study) instead of the simulations' 20, so that BO is not pure random search
over a 20-evaluation budget.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from openbo.optimizers.bo_scratch import run_bo_scratch
from openbo.optimizers.bo_taf_plus import (
    PopulationUser,
    TAFPlusConfig,
    TAFPlusPopulation,
    run_bo_taf_plus,
)
from openbo.benchmarks.hypervolume import hypervolume
from openbo.optimizers.mobo_botorch import MOBoTorchConfig, compute_hypervolume, run_mobo_botorch
from openbo.optimizers.mobo_taf import MOTAFConfig, MOTAFSequentialOptimizer, write_population_sources
from openbo.test_functions.taf_plus_users import (
    SPHERE_INPUT_DIM,
    SphereUser,
    make_sphere_users,
)

TAF_PLUS_METHODS = ("taf_plus", "taf", "bo", "random", "taf_ehvi", "mobo")

# Shift and scale range per simulation (App. B.3-B.4).
TAF_PLUS_SIMULATIONS = {
    "sim1": 0.01,
    "sim2_r005": 0.05,
    "sim2_r01": 0.1,
    "sim2_r02": 0.2,
    "sim2_r03": 0.3,
}

# Simulation 1's objective weights (App. B.3); Simulation 2 uses the last one.
SIM1_OBJECTIVE_WEIGHTS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.33, 0.33, 0.34),
    (0.5, 0.3, 0.2),
    (0.3, 0.5, 0.2),
)
SIM2_OBJECTIVE_WEIGHTS = (0.3, 0.5, 0.2)

# The paper's best decay per Simulation 2 range (App. B.4), for reference.
PAPER_SIM2_DECAY = {"sim2_r005": (7, 0.1), "sim2_r01": (4, 0.1), "sim2_r02": (4, 0.2), "sim2_r03": (3, 0.2)}


@dataclass(frozen=True)
class SimulationSettings:
    """Sizes shared by every simulation."""

    n_population: int = 10
    n_new: int = 10
    population_init: int = 20
    population_iter: int = 20
    v_ref: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    mobo_raw_samples: int = 256
    mobo_restarts: int = 4
    n_evals: int = 20
    bo_init: int = 5
    n_candidates: int = 256
    mobo_init: int = 1


def simulation_users(sim: str, seed: int, settings: SimulationSettings) -> tuple[list[SphereUser], list[SphereUser]]:
    """Return ``(population users, new users)`` for a simulation and seed."""
    if sim not in TAF_PLUS_SIMULATIONS:
        raise ValueError(f"Unknown simulation '{sim}'.")
    r = TAF_PLUS_SIMULATIONS[sim]
    users = make_sphere_users(settings.n_population + settings.n_new, r, r, seed)
    return users[: settings.n_population], users[settings.n_population :]


def _model_population_user(args: tuple[SphereUser, int, SimulationSettings]) -> dict[str, object]:
    """Run MOBO on one population user and fit their GPs (one process per user)."""
    user, seed, settings = args
    rng = np.random.default_rng(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_mobo_botorch(
            lambda x: user.observe(x, rng),
            MOBoTorchConfig(
                bounds=[(0.0, 1.0)] * SPHERE_INPUT_DIM,
                v_ref=list(settings.v_ref),
                n_init=settings.population_init,
                num_restarts=settings.mobo_restarts,
                raw_samples=settings.mobo_raw_samples,
                seed=seed,
            ),
            n_iter=settings.population_iter,
        )
    single = TAFPlusPopulation(SPHERE_INPUT_DIM, 3)
    fitted = single.add_user("user", result.x_obs, result.y_obs, seed=seed)
    return {"x": result.x_obs, "y": result.y_obs, "gp_states": fitted.gp_states, "hv": result.hv_history.tolist()}


def build_population(
    sim: str,
    seed: int,
    settings: SimulationSettings,
    cache_dir: str | Path,
    n_workers: int = 1,
) -> TAFPlusPopulation:
    """Return the simulation's population, from `cache_dir` if it was built before."""
    root = Path(cache_dir)
    if (root / "taf_plus_population.json").exists():
        return TAFPlusPopulation.load(root)
    population_users, _ = simulation_users(sim, seed, settings)
    jobs = [(u, seed * 1000 + i, settings) for i, u in enumerate(population_users)]
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            modeled = list(pool.map(_model_population_user, jobs))
    else:
        modeled = [_model_population_user(job) for job in jobs]
    population = TAFPlusPopulation(SPHERE_INPUT_DIM, 3)
    for i, m in enumerate(modeled):
        population.users.append(PopulationUser(f"pop_{i:03d}", m["x"], m["y"], m["gp_states"]))  # type: ignore[arg-type]
    population.save(root)
    (root / "population_modeling.json").write_text(
        json.dumps(
            {
                "sim": sim,
                "seed": seed,
                "settings": asdict(settings),
                "users": [asdict(u) for u in population_users],
                "hypervolume": [m["hv"] for m in modeled],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return population


def _run_mobo_user(
    user: SphereUser,
    rng: np.random.Generator,
    seed: int,
    settings: SimulationSettings,
    source_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run qLogNEHVI on one new user, with TAF-EHVI sources when ``source_dir`` is given."""
    common = dict(
        bounds=[(0.0, 1.0)] * SPHERE_INPUT_DIM,
        v_ref=list(settings.v_ref),
        n_init=settings.mobo_init,
        num_restarts=settings.mobo_restarts,
        raw_samples=settings.mobo_raw_samples,
        seed=seed,
    )
    n_iter = settings.n_evals - settings.mobo_init
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        # BoTorch warns about fitting a GP to very few points.
        warnings.simplefilter("ignore")
        if source_dir is None:
            result = run_mobo_botorch(lambda x: user.observe(x, rng), MOBoTorchConfig(**common), n_iter=n_iter)
            return result.x_obs, result.y_obs
        optimizer = MOTAFSequentialOptimizer(MOTAFConfig(**common, taf_run_dir=source_dir, n_iter=n_iter))
        optimizer.bootstrap(lambda x: user.observe(x, rng))
        for _ in range(n_iter):
            x = optimizer.suggest()
            optimizer.observe(x, user.observe(x, rng))
        return optimizer.x_obs, optimizer.y_obs


def pareto_hypervolume(user: SphereUser, v_ref: tuple[float, ...], step: float = 0.025) -> float:
    """Hypervolume of the user's noise-free Pareto front above ``v_ref``.

    Each objective is a concave quadratic, so every Pareto-optimal design
    maximizes some weighted sum of the objectives: the front is traced by the
    weighted-sum maximizers over a grid on the weight simplex (a slight
    underestimate, closing as ``step`` shrinks).
    """
    k = round(1.0 / step)
    weights = [(i / k, j / k, (k - i - j) / k) for i in range(k + 1) for j in range(k + 1 - i)]
    designs = np.vstack([user.weighted_optimum(np.asarray(w))[0] for w in weights])
    return compute_hypervolume(user.noise_free(designs), np.asarray(v_ref, dtype=np.float64))


def hypervolume_curve(user: SphereUser, x_obs: np.ndarray, v_ref: tuple[float, ...]) -> np.ndarray:
    """Noise-free hypervolume of the first ``t`` designs, for ``t = 1..n``."""
    y = user.noise_free(np.asarray(x_obs, dtype=np.float64))
    return np.asarray([hypervolume(y[: t + 1], v_ref) for t in range(y.shape[0])])


def run_taf_plus_method(
    method: str,
    sim: str,
    objective_weights: tuple[float, ...],
    seed: int,
    population: TAFPlusPopulation,
    settings: SimulationSettings,
    d1: int = 2,
    d2: float = 0.3,
) -> dict[str, object]:
    """Run one method on every new user of a simulation and return a JSON-ready dict."""
    if method not in TAF_PLUS_METHODS:
        raise ValueError(f"Unknown method '{method}'.")
    alpha = np.asarray(objective_weights, dtype=np.float64)
    _, new_users = simulation_users(sim, seed, settings)
    source_dir = None
    if method == "taf_ehvi":
        source_dir = write_population_sources(population, tempfile.mkdtemp(prefix="taf_ehvi_sources_"))
    records = []
    for i, user in enumerate(new_users):
        rng = np.random.default_rng(seed * 7919 + i)
        user_seed = seed * 1000 + 500 + i
        if method in ("taf_plus", "taf"):
            config = TAFPlusConfig(
                input_dim=SPHERE_INPUT_DIM,
                objective_weights=tuple(float(a) for a in alpha),
                decay=method == "taf_plus",
                d1=d1,
                d2=d2,
                n_candidates=settings.n_candidates,
                refine_suggestion=True,
                seed=user_seed,
            )
            def user_objective(
                x: np.ndarray, user: SphereUser = user, rng: np.random.Generator = rng
            ) -> np.ndarray:
                return user.observe(x, rng)

            result = run_bo_taf_plus(user_objective, config, population, settings.n_evals)
            x_obs, y_obs = result.x_obs, result.y_obs
        elif method == "bo":
            observed: list[np.ndarray] = []

            def weighted(x: np.ndarray, user=user, rng=rng, observed=observed) -> np.ndarray:
                y = user.observe(x, rng)
                observed.append(y)
                return y @ alpha

            with contextlib.redirect_stdout(io.StringIO()):
                bo = run_bo_scratch(
                    weighted,
                    bounds=[(0.0, 1.0)] * SPHERE_INPUT_DIM,
                    n_init=settings.bo_init,
                    n_iter=settings.n_evals - settings.bo_init,
                    kernel_type="rbf",
                    seed=user_seed,
                )
            x_obs, y_obs = bo.x_obs, np.vstack(observed)
            if y_obs.shape[0] != x_obs.shape[0]:
                raise RuntimeError("BO evaluated the objective outside its recorded observations.")
        elif method in ("taf_ehvi", "mobo"):
            x_obs, y_obs = _run_mobo_user(user, rng, user_seed, settings, source_dir)
        else:
            x_obs = np.random.default_rng(user_seed).random((settings.n_evals, SPHERE_INPUT_DIM))
            y_obs = user.observe(x_obs, rng)
        _, optimum = user.weighted_optimum(alpha)
        true_values = user.noise_free(x_obs) @ alpha
        instant_regret = optimum - true_values
        regret = optimum - np.maximum.accumulate(true_values)
        records.append(
            {
                "shift": list(user.shift),
                "scale": user.scale,
                "optimum": optimum,
                "x": x_obs.tolist(),
                "y": y_obs.tolist(),
                "true_weighted": true_values.tolist(),
                "instant_regret": instant_regret.tolist(),
                "regret": regret.tolist(),
            }
        )
    return {
        "method": method,
        "sim": sim,
        "objective_weights": [float(a) for a in alpha],
        "seed": seed,
        "decay": [d1, d2] if method == "taf_plus" else None,
        "settings": asdict(settings),
        "users": records,
    }
