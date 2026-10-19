"""Dispatch multi-objective benchmark methods by name.

Methods:

* ``mobo_ceipv_slf``, ``mobo_ceipv_mt``: CEIPV with a correlated GP and MAP hyperparameters.
* ``mobo_ceipv_slf_slice``: CEIPV-SLF averaged over slice-sampled hyperparameters (the paper's setup).
* ``mobo_ieipv``: IEIPV with independent GPs (correlation ablation).
* ``mobo_botorch``: BoTorch qLogNEHVI with independent GPs (external baseline).
* ``random``: uniform random search (floor).
"""

from __future__ import annotations

from typing import Any

from openbo.optimizers.mobo import MOBOConfig, MOBORunResult, run_mo_random, run_mobo
from openbo.test_functions.multi_objective import (
    MOFunctionSpec,
    get_mo_function_spec,
    make_slf_gp_problem,
)

MOBO_METHODS = ("mobo_ceipv_slf", "mobo_ceipv_slf_slice", "mobo_ceipv_mt", "mobo_ieipv", "mobo_botorch", "random")
MO_PROBLEMS = ("oka2", "vlmop3", "dtlz1a", "slf_gp")


def get_mo_problem(name: str, *, rho: float = 0.0, problem_seed: int = 0) -> MOFunctionSpec:
    """Return a registered problem, or an SLF-GP draw for ``name="slf_gp"``."""
    if name == "slf_gp":
        return make_slf_gp_problem(rho=rho, seed=problem_seed)
    return get_mo_function_spec(name)


def run_mobo_method(
    method: str,
    spec: MOFunctionSpec,
    *,
    n_init: int = 5,
    n_iter: int = 100,
    seed: int = 0,
    **options: Any,
) -> MOBORunResult:
    """Run ``method`` on ``spec``; ``options`` go to the method's config."""
    bounds = [(0.0, 1.0)] * spec.dim
    if method == "random":
        return run_mo_random(spec.objective, bounds, spec.v_ref, n_init=n_init, n_iter=n_iter, seed=seed)
    if method == "mobo_botorch":
        from openbo.optimizers.mobo_botorch import MOBoTorchConfig, run_mobo_botorch

        config = MOBoTorchConfig(bounds=bounds, v_ref=spec.v_ref, n_init=n_init, seed=seed, **options)
        return run_mobo_botorch(spec.objective, config, n_iter=n_iter)

    if method == "mobo_ceipv_slf_slice":
        options = {"hyperparameters": "slice", **options}
    model_and_acq = {
        "mobo_ceipv_slf": ("slf", "ceipv"),
        "mobo_ceipv_slf_slice": ("slf", "ceipv"),
        "mobo_ceipv_mt": ("mt", "ceipv"),
        "mobo_ieipv": ("independent", "ieipv"),
    }
    if method not in model_and_acq:
        raise ValueError(f"Unknown method {method!r}; expected one of {MOBO_METHODS}.")
    model, acquisition = model_and_acq[method]
    mobo_config = MOBOConfig(
        bounds=bounds,
        v_ref=spec.v_ref,
        w_ref=spec.w_ref,
        n_init=n_init,
        model=model,  # type: ignore[arg-type]
        acquisition=acquisition,  # type: ignore[arg-type]
        seed=seed,
        **options,
    )
    return run_mobo(spec.objective, mobo_config, n_iter=n_iter)
