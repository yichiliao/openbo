"""TAF's source-only warmup, per-call BoTorch seeds and high-dimensional variant optima."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pytest

from openbo.optimizers.bo_botorch import per_call_torch_seed
from openbo.optimizers.bo_taf import TAFConfig, TAFSequentialOptimizer
from openbo.test_functions import registry
from openbo.test_functions.registry import get_function_spec, make_variant_function_spec
from openbo.test_functions.tasks import TaskVariantSpec

SOURCES = "meta-bo-training/taf-gps/branin_train_v1"


def _optimizer(n_init: int, n_iter: int = 8) -> TAFSequentialOptimizer:
    config = TAFConfig(
        bounds=[(0.0, 1.0)] * 2,
        taf_run_dir=SOURCES,
        n_init=n_init,
        n_iter=n_iter,
        n_candidates=64,
        search_strategy="grid",
        optimize_hyperparameters=False,
        seed=0,
    )
    return TAFSequentialOptimizer(config)


def _phases(opt: TAFSequentialOptimizer, objective, n_steps: int, caplog) -> list[str]:
    """Run ``n_steps`` suggest/observe rounds and return each step's phase."""
    phases = []
    for _ in range(n_steps):
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="openbo.optimizers.bo_taf"):
            x = opt.suggest()
        phases.append(next(r.getMessage() for r in caplog.records if r.getMessage().endswith("_phase")))
        opt.observe(x, objective(x))
    return phases


def test_warmup_counts_target_points_without_random_starts(caplog) -> None:
    objective = get_function_spec("branin").objective
    phases = _phases(_optimizer(n_init=0), objective, 4, caplog)
    assert phases == ["source_only_phase"] * 3 + ["target_phase"]


def test_random_batch_counts_point_by_point(caplog) -> None:
    """Five bootstrapped points already exceed the 3-point warmup."""
    objective = get_function_spec("branin").objective
    opt = _optimizer(n_init=5)
    opt.bootstrap(objective)
    assert _phases(opt, objective, 1, caplog) == ["target_phase"]


def test_points_observed_one_by_one_count_toward_warmup(caplog) -> None:
    """As in the ConBO benchmark: random starts are passed to observe() one at a time."""
    objective = get_function_spec("branin").objective
    opt = _optimizer(n_init=0)
    rng = np.random.default_rng(0)
    for _ in range(2):
        x = rng.random((1, 2))
        opt.observe(x, objective(x))
    assert _phases(opt, objective, 2, caplog) == ["source_only_phase", "target_phase"]


def test_per_call_torch_seeds_differ_across_runs_and_calls() -> None:
    assert per_call_torch_seed(0, 3) == per_call_torch_seed(0, 3)
    # seed + call would make these two equal.
    assert per_call_torch_seed(0, 1) != per_call_torch_seed(1, 0)
    seeds = {per_call_torch_seed(s, c) for s in range(5) for c in range(5)}
    assert len(seeds) == 25
    assert all(0 <= s < 2**63 for s in seeds)


def test_high_dimensional_variant_optimum_warns_once() -> None:
    registry._warned_high_dim_optimum = False
    variant = TaskVariantSpec(input_shift=(0.05,) * 6, input_scale=(1.0,) * 6, output_scale=1.2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        specs = [make_variant_function_spec("hartmann6", variant) for _ in range(3)]
    assert len([w for w in caught if "analytic upper bound" in str(w.message)]) == 1
    base = get_function_spec("hartmann6").optimum
    assert all(spec.optimum == pytest.approx(1.2 * base) for spec in specs)
