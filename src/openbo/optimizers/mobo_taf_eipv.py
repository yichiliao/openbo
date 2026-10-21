"""TAF-EIPV: transfer for OpenBO's multi-objective BO (``docs/plans/taf_eipv.md``).

``TAFEIPVSequentialOptimizer`` is :class:`~openbo.optimizers.mobo.MOBOSequentialOptimizer`
with the original TAF's acquisition (``acquisition/taf_eipv.py``): the target's
IEIPV (default) or CEIPV, blended with the hypervolume each source's posterior
mean adds to the front of what that source predicts at the designs already
evaluated. With no target data it suggests from the sources alone, so
``n_init`` may be 0. With no sources it is exactly ``MOBOSequentialOptimizer``.

Sources are earlier tasks or users with one ``GPScratch`` per objective: a
``TAFPlusPopulation``, or a directory in the TAF run layout (``gp_states/`` and
``trajectories/``, as written by ``TAFPlusPopulation.save`` or
``mobo_taf.write_population_sources``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from openbo.acquisition.taf_eipv import (
    log_taf_eipv,
    reference_cells,
    source_improvement,
    taf_r_source_weights,
)
from openbo.acquisition.taf_plus import decay_factor
from openbo.models.gp_scratch import GPScratch
from openbo.optimizers.bo_taf_plus import TAFPlusPopulation
from openbo.optimizers.conbo import gp_from_state
from openbo.optimizers.mobo import (
    AcquisitionFn,
    MOBOConfig,
    MOBORunResult,
    MOBOSequentialOptimizer,
    MultiObjective,
)
from openbo.models.multi_output_gp import MultiOutputGP


@dataclass
class TAFEIPVConfig(MOBOConfig):
    """``MOBOConfig`` plus the transfer settings; IEIPV on independent GPs by default.

    ``taf_run_dir`` holds the sources (or pass a population to the optimizer).
    ``weight_mode`` is ``"taf_r"`` (ranking agreement, the original's best) or
    ``"uniform"`` (every source ``3/4``). ``target_weight`` defaults to the
    original's ``gamma(0) = 3/4``. ``decay`` multiplies the source weights by
    TAF+'s ``delta(t)`` with ``(d1, d2)``.
    """

    model: Literal["independent", "mt", "slf"] = "independent"
    acquisition: Literal["ceipv", "ieipv"] = "ieipv"
    taf_run_dir: str | Path | None = None
    weight_mode: Literal["taf_r", "uniform"] = "taf_r"
    rho: float = 1.0
    target_weight: float = 0.75
    decay: bool = False
    d1: int = 2
    d2: float = 0.3

    def __post_init__(self) -> None:
        if self.weight_mode not in ("taf_r", "uniform"):
            raise ValueError("weight_mode must be 'taf_r' or 'uniform'.")
        if self.rho <= 0.0:
            raise ValueError("rho must be positive.")
        if self.target_weight <= 0.0:
            raise ValueError("target_weight must be positive.")
        if self.decay:
            decay_factor(1, self.d1, self.d2)  # validates d1, d2


@dataclass
class EIPVSource:
    """One earlier task or user: its data and one fitted GP per objective."""

    name: str
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    gps: list[GPScratch] = field(repr=False)

    def mean(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Posterior mean per objective, shape ``(n, L)``."""
        return np.column_stack([gp.posterior(x)[0] for gp in self.gps])


def sources_from_population(population: TAFPlusPopulation) -> list[EIPVSource]:
    """Use a TAF+ population's users (and their fitted GPs) as sources."""
    return [EIPVSource(u.name, u.x, u.y, list(u.gps)) for u in population.users]


def load_sources(taf_run_dir: str | Path) -> list[EIPVSource]:
    """Load sources from ``gp_states/<task>.json`` and ``trajectories/<task>.json``.

    A GP-state file holds ``{"gp_state": {"objectives": [...]}}`` (TAF-EHVI's
    layout), ``{"gp_states": [...]}`` (a saved TAF+ population), or one shared
    ``{"gp_state": {...}}``.
    """
    root = Path(taf_run_dir)
    sources = []
    for gp_path in sorted((root / "gp_states").glob("*.json")):
        traj_path = root / "trajectories" / gp_path.name
        if not traj_path.exists():
            raise ValueError(f"Source '{gp_path.stem}' has no trajectory at {traj_path}.")
        payload = json.loads(gp_path.read_text(encoding="utf-8"))
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
        x = np.asarray(traj["x_values"], dtype=np.float64)
        y = np.asarray(traj["y_values"], dtype=np.float64)
        if y.ndim != 2 or x.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError(f"Source '{gp_path.stem}' needs x_values (n, d) and y_values (n, L).")
        if "gp_states" in payload:
            states = payload["gp_states"]
        else:
            state = payload["gp_state"]
            states = state["objectives"] if "objectives" in state else [state] * y.shape[1]
        if len(states) != y.shape[1]:
            raise ValueError(f"Source '{gp_path.stem}' needs one GP state per objective.")
        gps = [gp_from_state(states[l], x, y[:, l]) for l in range(y.shape[1])]
        sources.append(EIPVSource(gp_path.stem, x, y, gps))
    return sources


class TAFEIPVSequentialOptimizer(MOBOSequentialOptimizer):
    """Ask/tell multi-objective TAF on OpenBO's NumPy MOBO."""

    def __init__(self, config: TAFEIPVConfig, population: TAFPlusPopulation | None = None) -> None:
        super().__init__(config)
        self.config: TAFEIPVConfig = config
        if population is not None and config.taf_run_dir is not None:
            raise ValueError("Pass either a population or config.taf_run_dir, not both.")
        if population is not None:
            self.sources = sources_from_population(population)
        elif config.taf_run_dir is not None:
            self.sources = load_sources(config.taf_run_dir)
        else:
            self.sources = []
        for s in self.sources:
            if s.x.shape[1] != self.d or s.y.shape[1] != self.n_objectives:
                raise ValueError(
                    f"Source '{s.name}' has d={s.x.shape[1]}, L={s.y.shape[1]}; "
                    f"the target has d={self.d}, L={self.n_objectives}."
                )
        self.last_source_weights = np.zeros(len(self.sources), dtype=np.float64)

    def _source_weights(self) -> NDArray[np.float64]:
        """Weights for the next suggestion, from the target's observations so far."""
        n_sources = len(self.sources)
        t = self.y_obs.shape[0]
        if self.config.weight_mode == "uniform" or t < 2:
            weights = np.full(n_sources, 0.75, dtype=np.float64)
        else:
            means = np.stack([s.mean(self.x_obs) for s in self.sources])
            weights = taf_r_source_weights(self.y_obs, means, self.config.rho)
        if self.config.decay:
            weights = weights * decay_factor(t + 1, self.config.d1, self.config.d2)
        return weights

    def _source_terms(self) -> AcquisitionFn:
        """``x -> (M, n)`` improvements over each source's current reference front."""
        cells = [
            reference_cells(s.mean(self.x_obs) if self.x_obs.shape[0] else np.empty((0, self.n_objectives)),
                            self.v_ref, self.w_ref)
            for s in self.sources
        ]

        def terms(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.stack([source_improvement(s.mean(x), c) for s, c in zip(self.sources, cells)])

        return terms

    def _log_acquisition(self, models: list[MultiOutputGP]) -> tuple[AcquisitionFn, AcquisitionFn]:
        """Blend the target's log EIPV with the sources' improvements."""
        screen, acquisition = super()._log_acquisition(models)
        if not self.sources:
            return screen, acquisition
        weights = self._source_weights()
        self.last_source_weights = weights
        if not np.any(weights > 0.0):
            return screen, acquisition
        terms = self._source_terms()
        tw = self.config.target_weight

        def blended(base: AcquisitionFn) -> AcquisitionFn:
            return lambda x: log_taf_eipv(base(x), terms(x), weights, tw)

        return blended(screen), blended(acquisition)

    def suggest(self) -> NDArray[np.float64]:
        """Next point ``(1, d)``; from the sources alone before any target data."""
        if self.x_obs.shape[0] > 0:
            return super().suggest()
        if not self.sources:
            raise ValueError("Cannot suggest without observations or sources. Call bootstrap() first.")
        weights = self._source_weights()
        self.last_source_weights = weights
        terms = self._source_terms()
        tw = self.config.target_weight

        def source_only(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return log_taf_eipv(None, terms(x), weights, tw)

        return self._search(source_only, source_only)

    def result(self) -> MOBORunResult:
        """The MOBO result, with the transfer state added to ``final_model_state``."""
        result = super().result()
        result.final_model_state.update(
            {
                "sources": [s.name for s in self.sources],
                "weight_mode": self.config.weight_mode,
                "last_source_weights": self.last_source_weights.tolist(),
                "target_weight": self.config.target_weight,
            }
        )
        return result


def run_mobo_taf_eipv(
    objective: MultiObjective,
    config: TAFEIPVConfig,
    n_iter: int = 25,
    population: TAFPlusPopulation | None = None,
) -> MOBORunResult:
    """Run ``config.n_init`` random points (may be 0), then ``n_iter`` TAF-EIPV steps."""
    optimizer = TAFEIPVSequentialOptimizer(config, population)
    optimizer.bootstrap(objective)
    for _ in range(n_iter):
        x_next = optimizer.suggest()
        optimizer.observe(x_next, np.asarray(objective(x_next), dtype=np.float64))
    return optimizer.result()
