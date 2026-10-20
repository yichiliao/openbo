"""Tests for the multi-objective optimizers (mobo_botorch) and TAF-EHVI (mobo_taf)."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
import torch

from openbo.acquisition.taf_mo_ehvi import (
    MOSourceTaskSurrogate,
    build_source_hvi_term,
    compute_taf_r_pareto_weights,
    compute_taf_r_ranking_weights,
    epanechnikov_weight,
    mo_meta_features,
    pareto_relation,
    taf_mo_ehvi_acquisition,
)
from openbo.optimizers.mobo_botorch import (
    MOBoTorchConfig,
    MOBoTorchSequentialOptimizer,
    compute_hypervolume,
    pareto_front,
    run_mobo_botorch,
)
from openbo.optimizers.mobo_taf import (
    MOTAFConfig,
    MOTAFSequentialOptimizer,
    _load_mo_source_surrogates,
    run_mobo_taf,
)

D = 3
M = 2
REF = [-1.0, -1.0]
BOUNDS = [(0.0, 1.0)] * D


def objective(x: np.ndarray) -> np.ndarray:
    """Two competing quadratics; both maximized."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    f1 = 1.0 - np.sum((x - 0.3) ** 2, axis=1)
    f2 = 1.0 - np.sum((x - 0.7) ** 2, axis=1)
    return np.stack([f1, f2], axis=1)


def write_source(root, name, shift=0.0, seed=0, n=12):
    rng = np.random.default_rng(seed)
    for sub in ("gp_states", "trajectories"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    x = rng.random((n, D))
    y = objective(x + shift)
    (root / "trajectories" / f"{name}.json").write_text(
        json.dumps(
            {
                "x_values": x.tolist(),
                "y_values": y.tolist(),
                "pareto_front": pareto_front(y).tolist(),
            }
        ),
        encoding="utf-8",
    )
    (root / "gp_states" / f"{name}.json").write_text(
        json.dumps(
            {
                "gp_state": {
                    "kernel_type": "matern52",
                    "lengthscale": [0.3] * D,
                    "variance": 1.0,
                    "noise": 1e-4,
                }
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------- hypervolume utilities


def test_hypervolume_of_known_front():
    y = np.array([[0.0, 0.0]])
    assert compute_hypervolume(y, np.array([-1.0, -1.0])) == pytest.approx(1.0)


def test_hypervolume_ignores_points_not_dominating_reference():
    y = np.array([[-2.0, 0.5], [0.0, 0.0]])
    assert compute_hypervolume(y, np.array([-1.0, -1.0])) == pytest.approx(1.0)


def test_hypervolume_of_empty_or_all_dominated_is_zero():
    assert compute_hypervolume(np.empty((0, 2)), np.array([-1.0, -1.0])) == 0.0
    assert compute_hypervolume(np.array([[-5.0, -5.0]]), np.array([-1.0, -1.0])) == 0.0


def test_hypervolume_rejects_mismatched_reference():
    with pytest.raises(ValueError):
        compute_hypervolume(np.zeros((3, 2)), np.array([-1.0, -1.0, -1.0]))


def test_pareto_front_keeps_only_non_dominated():
    y = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.1, 0.1]])
    front = pareto_front(y)
    assert front.shape[0] == 3
    assert not any(np.allclose(row, [0.1, 0.1]) for row in front)


# ---------------------------------------------------------------- Pareto relation


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([1.0, 1.0], [0.0, 0.0], 1),
        ([0.0, 0.0], [1.0, 1.0], -1),
        ([1.0, 0.0], [0.0, 1.0], 0),
        ([1.0, 1.0], [1.0, 1.0], 0),
        ([1.0, 1.0], [1.0, 0.0], 1),
    ],
)
def test_pareto_relation(a, b, expected):
    assert pareto_relation(np.array(a), np.array(b)) == expected


# ---------------------------------------------------------------- TAF-R weighting


class _FixedMeanSource:
    """Source surrogate stub whose posterior mean is a fixed matrix."""

    def __init__(self, name, mu):
        self.name = name
        self._mu = np.asarray(mu, dtype=np.float64)
        self.pareto_front = self._mu
        self.meta_features = np.zeros(1 + 2 * M)
        self.reference_front = None

    def posterior_mean(self, x):
        return self._mu

    def front(self):
        return self.pareto_front


TARGET_Y = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.9, 0.9], [0.2, 0.3]])
TARGET_X = np.random.default_rng(0).random((5, D))

# Both TAF-R variants (objective-wise ranking agreement -- the default -- and
# Pareto-dominance agreement, kept for ablation) share the degenerate contracts below.
BOTH_TAF_R = pytest.mark.parametrize(
    "weight_fn",
    [compute_taf_r_ranking_weights, compute_taf_r_pareto_weights],
    ids=["ranking", "pareto"],
)


@BOTH_TAF_R
def test_taf_r_gives_no_sources_an_empty_vector(weight_fn):
    w = weight_fn([], TARGET_X, TARGET_Y, rho=1.0)
    assert w.shape == (0,)


@BOTH_TAF_R
def test_taf_r_falls_back_to_uniform_below_two_observations(weight_fn):
    sources = [_FixedMeanSource("a", TARGET_Y[:1]), _FixedMeanSource("b", TARGET_Y[:1])]
    w = weight_fn(sources, TARGET_X[:1], TARGET_Y[:1], rho=1.0)
    assert w == pytest.approx([0.5, 0.5])


@BOTH_TAF_R
def test_taf_r_ranks_copy_above_negation_and_zeroes_flat(weight_fn):
    sources = [
        _FixedMeanSource("copy", TARGET_Y),
        _FixedMeanSource("negated", -TARGET_Y),
        _FixedMeanSource("flat", np.zeros_like(TARGET_Y)),
    ]
    w = weight_fn(sources, TARGET_X, TARGET_Y, rho=1.0)
    assert w[0] > 0.0
    # An exactly inverted source carries anti-information.
    assert w[1] == 0.0
    # A source that orders nothing carries no information; it must not keep the weight
    # its formula-distance would imply (the source_strict guard in both variants).
    assert w[2] == 0.0
    assert w.sum() == pytest.approx(1.0)


@BOTH_TAF_R
def test_taf_r_weight_decreases_monotonically_with_disagreement(weight_fn):
    half = TARGET_Y.copy()
    half[:2] = -half[:2]
    sources = [
        _FixedMeanSource("copy", TARGET_Y),
        _FixedMeanSource("half", half),
        _FixedMeanSource("negated", -TARGET_Y),
    ]
    w = weight_fn(sources, TARGET_X, TARGET_Y, rho=1.0)
    assert w[0] >= w[1] >= w[2]


@BOTH_TAF_R
def test_taf_r_returns_zeros_when_every_source_is_rejected(weight_fn):
    sources = [_FixedMeanSource("n1", -TARGET_Y), _FixedMeanSource("n2", -TARGET_Y)]
    w = weight_fn(sources, TARGET_X, TARGET_Y, rho=1.0)
    assert np.all(w == 0.0)


def test_taf_r_ranking_matches_worked_example():
    """d_s = 1/2 for the email's example: obj-1 rankings agree, obj-2 rankings disagree.

    Target f_t(x1)=[0.5,0.7], f_t(x2)=[0.6,0.8]; source f_s(x1)=[0.1,0.4],
    f_s(x2)=[0.4,0.1]. A perfect copy rides along so the normalized weights expose the
    raw distances: w_example / w_copy must equal epa(0.5) / epa(0.0).
    """
    y = np.array([[0.5, 0.7], [0.6, 0.8]])
    x = TARGET_X[:2]
    sources = [
        _FixedMeanSource("example", np.array([[0.1, 0.4], [0.4, 0.1]])),
        _FixedMeanSource("copy", y),
    ]
    w = compute_taf_r_ranking_weights(sources, x, y, rho=1.0)
    expected_ratio = epanechnikov_weight(0.5, 1.0) / epanechnikov_weight(0.0, 1.0)
    assert w[0] / w[1] == pytest.approx(expected_ratio)
    assert w.sum() == pytest.approx(1.0)


def test_taf_r_ranking_uses_non_dominated_pairs_the_pareto_variant_discards():
    """The motivating edge case: observations on a trade-off curve are mutually
    non-dominated, so the Pareto variant collects zero evidence and returns the null
    result -- even for a source in perfect objective-wise agreement. The ranking variant
    scores M rankings per pair and separates a copy from an inverted source cleanly."""
    tradeoff = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
    x = TARGET_X[:4]
    sources = [
        _FixedMeanSource("copy", tradeoff),
        _FixedMeanSource("negated", -tradeoff),
    ]
    w_ranking = compute_taf_r_ranking_weights(sources, x, tradeoff, rho=1.0)
    assert w_ranking == pytest.approx([1.0, 0.0])
    w_pareto = compute_taf_r_pareto_weights(sources, x, tradeoff, rho=1.0)
    assert np.all(w_pareto == 0.0)


def test_taf_r_ranking_skips_target_ties_from_distance_and_denominator():
    """A source order at a target tie is ignored. Only objective 2 is target-strict here,
    where both sources agree, so their equal raw weights normalize to one half each."""
    y = np.array([[1.0, 0.0], [1.0, 1.0]])
    x = TARGET_X[:2]
    sources = [
        _FixedMeanSource("resolves_tie", np.array([[0.2, 0.1], [0.9, 0.8]])),
        _FixedMeanSource("copy", y),
    ]
    w = compute_taf_r_ranking_weights(sources, x, y, rho=1.0)
    assert w == pytest.approx([0.5, 0.5])


def test_taf_r_ranking_rejects_all_target_ties():
    """An all-tied target supplies no pairwise ranking evidence to transfer."""
    y = np.ones((2, M))
    source = _FixedMeanSource("strict", np.array([[0.0, 0.0], [1.0, 1.0]]))
    w = compute_taf_r_ranking_weights([source], TARGET_X[:2], y, rho=1.0)
    assert w == pytest.approx([0.0])


def test_taf_r_ranking_rejects_wrong_mean_shape():
    """A source mean of shape (n,) would broadcast silently; it must raise instead."""
    bad = _FixedMeanSource("bad", TARGET_Y)
    bad._mu = TARGET_Y[:, 0].copy()  # (n,) instead of (n, M)
    with pytest.raises(ValueError, match="posterior mean has shape"):
        compute_taf_r_ranking_weights([bad], TARGET_X, TARGET_Y, rho=1.0)


def test_mo_meta_features_shape():
    x = np.random.default_rng(1).random((7, D))
    y = objective(x)
    feats = mo_meta_features(x, y)
    assert feats.shape == (1 + 2 * M,)
    assert feats[0] == float(D)


# ---------------------------------------------------------------- source term


def _linear_source(scale=1.0):
    def mean_fn(x: torch.Tensor) -> torch.Tensor:
        a = x.sum(dim=-1, keepdim=True) * scale
        return torch.cat([a, -a], dim=-1)

    return MOSourceTaskSurrogate(
        name="lin",
        mean_fn=mean_fn,
        pareto_front=np.array([[0.0, 0.0], [0.5, -0.5]]),
        meta_features=np.zeros(1 + 2 * M),
    )


def test_source_term_is_differentiable_and_deterministic():
    term = build_source_hvi_term(_linear_source(), np.array(REF))
    x = torch.rand(4, 1, D, dtype=torch.double, requires_grad=True)
    v1 = term(x)
    assert v1.shape == (4,)
    assert torch.isfinite(v1).all()
    v1.sum().backward()
    assert torch.isfinite(x.grad).all()
    # Zero-variance posterior => no Monte-Carlo noise between calls.
    with torch.no_grad():
        assert torch.equal(term(x.detach()), term(x.detach()))


def test_source_term_rejects_front_that_cannot_dominate_reference():
    src = _linear_source()
    src.pareto_front = np.array([[-5.0, -5.0]])
    with pytest.raises(ValueError):
        build_source_hvi_term(src, np.array(REF))


@pytest.mark.parametrize("m_objectives", [2, 3], ids=["M=2", "M=3"])
def test_source_term_matches_bruteforce_hypervolume_improvement(m_objectives):
    """The EHVI ground truth: for a deterministic source model, exp(qLogEHVI(x)) must
    equal HV(front ∪ {mu(x)}) - HV(front) computed by brute force, up to the tau_max
    smoothing error. Cross-checks the box-decomposition path inside qLogEHVI against the
    independent Hypervolume.compute path, on improving AND non-improving points."""
    if m_objectives == 2:
        def mean_fn(x):
            a = x[..., 0:1] * 2.0 - 0.5
            b = x[..., 1:2] * 2.0 - 0.5
            return torch.cat([a, b], dim=-1)

        front = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
        ref = np.array([-0.2, -0.2])
    else:
        def mean_fn(x):
            a = x[..., 0:1]
            b = x[..., 1:2]
            return torch.cat([a, b, 1.2 - a - b], dim=-1)

        front = np.array([[0.6, 0.3, 0.3], [0.2, 0.7, 0.3], [0.3, 0.3, 0.6]])
        ref = np.zeros(3)

    src = MOSourceTaskSurrogate(
        name="det", mean_fn=mean_fn, pareto_front=front,
        meta_features=np.zeros(1 + 2 * m_objectives),
    )
    term = build_source_hvi_term(src, ref)

    x = torch.tensor(np.random.default_rng(5).random((32, 1, D)), dtype=torch.double)
    with torch.no_grad():
        got = np.exp(term(x).numpy())
    mu = mean_fn(x.squeeze(1)).numpy()

    hv_front = compute_hypervolume(front, ref)
    n_improving = 0
    for k in range(32):
        hvi = compute_hypervolume(np.vstack([front, mu[k][None, :]]), ref) - hv_front
        if hvi > 1e-9:
            n_improving += 1
            assert got[k] == pytest.approx(hvi, rel=1e-3)
        else:
            # Smoothing must not leak spurious improvement onto dominated points.
            assert got[k] < 1e-9
    # The check is vacuous unless both regimes actually occur.
    assert 0 < n_improving < 32


def test_taf_mo_ehvi_blend_matches_log_of_weighted_average():
    def t(x):
        return torch.full((x.shape[0],), np.log(0.4), dtype=torch.double)

    def s(x):
        return torch.full((x.shape[0],), np.log(0.1), dtype=torch.double)

    x = torch.rand(3, 1, D, dtype=torch.double)
    got = taf_mo_ehvi_acquisition(x, t, [s], np.array([0.5]), target_weight=1.0)
    expected = np.log((1.0 * 0.4 + 0.5 * 0.1) / (1.0 + 0.5))
    assert got.detach().numpy() == pytest.approx(expected)


def test_taf_mo_ehvi_drops_non_positive_weight_sources_from_both_sides():
    def t(x):
        return torch.full((x.shape[0],), np.log(0.4), dtype=torch.double)

    def s(x):
        return torch.full((x.shape[0],), np.log(0.1), dtype=torch.double)

    x = torch.rand(2, 1, D, dtype=torch.double)
    got = taf_mo_ehvi_acquisition(x, t, [s], np.array([0.0]), target_weight=1.0)
    # A zero-weight source must not dilute the denominator: result is the target alone.
    assert got.detach().numpy() == pytest.approx(np.log(0.4))


def test_taf_mo_ehvi_requires_at_least_one_active_term():
    x = torch.rand(2, 1, D, dtype=torch.double)
    with pytest.raises(ValueError):
        taf_mo_ehvi_acquisition(x, None, [], np.zeros(0), target_weight=0.0)


# ---------------------------------------------------------------- source loading


def test_loader_returns_empty_for_missing_directory(tmp_path):
    assert _load_mo_source_surrogates(tmp_path / "nope") == []


def test_loader_skips_unpaired_and_malformed_artifacts(tmp_path):
    write_source(tmp_path, "good", seed=1)
    # trajectory without a gp_state partner is simply never discovered
    (tmp_path / "trajectories" / "orphan.json").write_text("{}", encoding="utf-8")
    # present but malformed
    (tmp_path / "gp_states" / "bad.json").write_text(
        json.dumps({"gp_state": {"kernel_type": "matern52"}}), encoding="utf-8"
    )
    (tmp_path / "trajectories" / "bad.json").write_text(
        json.dumps({"x_values": [[0, 0, 0]], "y_values": [[0, 0]]}), encoding="utf-8"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sources = _load_mo_source_surrogates(tmp_path, expected_m=M, expected_d=D)
    assert [s.name for s in sources] == ["good"]
    assert any("bad" in str(w.message) for w in caught)


def test_loader_skips_dimension_mismatch(tmp_path):
    write_source(tmp_path, "good", seed=2)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert _load_mo_source_surrogates(tmp_path, expected_m=M, expected_d=D + 1) == []


def test_loader_replays_mean_constant(tmp_path):
    """A stored mean_constant must shift the replayed prior: far away from the data the
    posterior mean reverts to the fitted constant (in standardized space), not to 0.
    Artifacts without the field keep the previous zero-mean behavior."""
    write_source(tmp_path, "src", seed=11)
    baseline = _load_mo_source_surrogates(tmp_path, expected_m=M, expected_d=D)[0]

    gp_path = tmp_path / "gp_states" / "src.json"
    payload = json.loads(gp_path.read_text(encoding="utf-8"))
    payload["gp_state"]["lengthscale"] = [0.02] * D  # kill data influence far away
    payload["gp_state"]["mean_constant"] = 1.5
    gp_path.write_text(json.dumps(payload), encoding="utf-8")
    shifted = _load_mo_source_surrogates(tmp_path, expected_m=M, expected_d=D)[0]

    baseline_payload = dict(payload)
    baseline_payload["gp_state"] = dict(payload["gp_state"])
    del baseline_payload["gp_state"]["mean_constant"]
    gp_path.write_text(json.dumps(baseline_payload), encoding="utf-8")
    zero_mean = _load_mo_source_surrogates(tmp_path, expected_m=M, expected_d=D)[0]

    # Recreate write_source's data to know the standardization scale.
    rng = np.random.default_rng(11)
    y = objective(rng.random((12, D)))

    x_far = np.full((1, D), 25.0)
    mu_shifted = shifted.posterior_mean(x_far)
    mu_zero = zero_mean.posterior_mean(x_far)
    # Standardize untransforms mean_constant c to mean_y + c * std_y per objective.
    np.testing.assert_allclose(
        (mu_shifted - mu_zero).reshape(-1), 1.5 * y.std(axis=0, ddof=1), rtol=1e-6
    )
    # And the default path (no mean_constant) is unchanged vs the original artifact:
    # both revert to mean_y far from the data.
    assert baseline is not None


# ---------------------------------------------------------------- end-to-end loops


def test_run_mobo_botorch_shapes_and_monotone_hypervolume():
    res = run_mobo_botorch(objective, BOUNDS, REF, n_init=5, n_iter=2, seed=0)
    assert res.x_obs.shape == (7, D)
    assert res.y_obs.shape == (7, M)
    assert res.hypervolume_history.shape == (7,)
    assert np.all(np.diff(res.hypervolume_history) >= -1e-9)
    assert res.pareto_front.shape[1] == M


def test_run_mobo_botorch_requires_initial_points():
    with pytest.raises(ValueError):
        run_mobo_botorch(objective, BOUNDS, REF, n_init=0, n_iter=1, seed=0)


def test_run_mobo_taf_runs_and_records_transfer_state(tmp_path):
    write_source(tmp_path, "srcA", shift=0.0, seed=3)
    write_source(tmp_path, "srcB", shift=0.25, seed=4)
    res = run_mobo_taf(
        objective, BOUNDS, REF, tmp_path, n_init=5, n_iter=2,
        taf_weight_mode="taf_r", seed=0,
    )
    assert res.y_obs.shape == (7, M)
    assert res.final_state["n_sources"] == 2
    assert res.final_state["taf_weight_mode"] == "taf_r"
    assert len(res.final_state["last_source_weights"]) == 2


def test_taf_with_no_sources_is_identical_to_plain_mobo(tmp_path):
    """The degenerate case must BE plain MOBO, not merely approximate it."""
    (tmp_path / "gp_states").mkdir(parents=True, exist_ok=True)
    baseline = run_mobo_botorch(objective, BOUNDS, REF, n_init=5, n_iter=2, seed=0)
    degenerate = run_mobo_taf(objective, BOUNDS, REF, tmp_path, n_init=5, n_iter=2, seed=0)
    np.testing.assert_allclose(degenerate.x_obs, baseline.x_obs)
    np.testing.assert_allclose(
        degenerate.hypervolume_history, baseline.hypervolume_history
    )


def test_invalid_weight_mode_is_rejected(tmp_path):
    (tmp_path / "gp_states").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        MOTAFSequentialOptimizer(
            MOTAFConfig(
                bounds=BOUNDS, ref_point=REF, taf_run_dir=tmp_path, taf_weight_mode="nope"
            )
        )


def test_taf_weight_mode_dispatches_to_ranking_or_pareto_variant(tmp_path):
    """"taf_r" must route to objective-wise ranking agreement and "taf_r_pareto" to the
    dominance variant. On a trade-off-curve target with a perfectly agreeing source the
    two are behaviorally distinguishable: ranking accepts the source, dominance finds no
    evidence and rejects it."""
    (tmp_path / "gp_states").mkdir(parents=True, exist_ok=True)
    tradeoff = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])

    def weights_for(mode):
        opt = MOTAFSequentialOptimizer(
            MOTAFConfig(
                bounds=BOUNDS, ref_point=REF, taf_run_dir=tmp_path, taf_weight_mode=mode
            )
        )
        opt.x_obs = np.random.default_rng(3).random((4, D))
        opt.y_obs = tradeoff
        opt.source_surrogates = [_FixedMeanSource("copy", tradeoff)]
        return opt._current_source_weights()

    assert weights_for("taf_r") == pytest.approx([1.0])
    assert weights_for("taf_r_pareto") == pytest.approx([0.0])


def test_non_dominating_source_is_warn_skipped_at_construction(tmp_path):
    """A frame-valid source whose Pareto front never strictly dominates the reference
    point must be warn-skipped (loader contract), not abort optimizer construction."""
    write_source(tmp_path, "good", seed=6)
    write_source(tmp_path, "dead", seed=7)
    traj_path = tmp_path / "trajectories" / "dead.json"
    payload = json.loads(traj_path.read_text(encoding="utf-8"))
    front = np.asarray(payload["pareto_front"], dtype=np.float64)
    front[:, 0] = REF[0]  # first objective pinned at the reference point -> no strict dominance
    payload["pareto_front"] = front.tolist()
    traj_path.write_text(json.dumps(payload), encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt = MOTAFSequentialOptimizer(
            MOTAFConfig(bounds=BOUNDS, ref_point=REF, taf_run_dir=tmp_path)
        )
    assert [s.name for s in opt.source_surrogates] == ["good"]
    assert len(opt._source_terms) == 1
    assert any("dead" in str(w.message) for w in caught)


def test_source_meta_features_must_cover_every_source(tmp_path):
    write_source(tmp_path, "srcA", seed=5)
    write_source(tmp_path, "srcB", seed=6)
    with pytest.raises(ValueError, match="must cover every source"):
        MOTAFSequentialOptimizer(
            MOTAFConfig(
                bounds=BOUNDS,
                ref_point=REF,
                taf_run_dir=tmp_path,
                source_meta_features={"srcA": np.zeros(1 + 2 * M)},
            )
        )


def test_population_decay_matches_taf_plus_definition():
    """gamma(t) per Liao et al. CHI'24 Eq. 6: flat, then linear, then clamped at 0."""
    from openbo.optimizers.mobo_taf import population_decay

    # No decay configured -> always 1.0.
    assert population_decay(99, 0, 0.0) == 1.0
    # Flat until d1.
    assert population_decay(1, 2, 0.3) == 1.0
    assert population_decay(2, 2, 0.3) == 1.0
    # Linear afterwards.
    assert population_decay(3, 2, 0.3) == pytest.approx(0.7)
    assert population_decay(4, 2, 0.3) == pytest.approx(0.4)
    # Clamped at zero, never negative.
    assert population_decay(9, 2, 0.3) == 0.0
    assert population_decay(50, 2, 0.3) == 0.0


def test_decay_shrinks_source_weights_over_suggestions(tmp_path):
    write_source(tmp_path, "srcA", seed=7)
    write_source(tmp_path, "srcB", shift=0.2, seed=8)
    opt = MOTAFSequentialOptimizer(
        MOTAFConfig(
            bounds=BOUNDS, ref_point=REF, taf_run_dir=tmp_path, n_init=5,
            num_restarts=2, raw_samples=16,
            decay_start_iter=1, decay_rate=0.5, seed=0,
        )
    )
    opt.bootstrap(objective)
    factors = []
    for _ in range(3):
        x = opt.suggest()
        factors.append(opt.result().final_state["last_decay_factor"])
        opt.observe(x, objective(x))
    # Monotone non-increasing, and fully decayed by the third suggestion.
    assert factors[0] >= factors[1] >= factors[2]
    assert factors[-1] == 0.0


def test_fully_decayed_sources_fall_back_to_plain_mobo(tmp_path):
    """Once gamma reaches 0 the acquisition must be the bare qLogNEHVI object."""
    write_source(tmp_path, "srcA", seed=9)
    opt = MOTAFSequentialOptimizer(
        MOTAFConfig(
            bounds=BOUNDS, ref_point=REF, taf_run_dir=tmp_path, n_init=5,
            num_restarts=2, raw_samples=16,
            decay_start_iter=0, decay_rate=1.0, seed=0,
        )
    )
    opt.bootstrap(objective)
    opt.n_suggestions = 5  # well past full decay
    train_x, train_y, _ = opt._unit_train_tensors()
    model = opt._fit_model(train_x, train_y)
    acq = opt._build_acquisition(model, train_x)
    from botorch.acquisition.multi_objective.logei import (
        qLogNoisyExpectedHypervolumeImprovement,
    )

    assert isinstance(acq, qLogNoisyExpectedHypervolumeImprovement)


def test_warmup_of_one_makes_exactly_the_first_suggestion_source_only(tmp_path):
    """Regression for the warmup off-by-one: k=1 must yield ONE source-only suggestion.

    The pre-fix comparison (`n_suggestions < k` on a pre-incremented counter) made k=1 a
    silent no-op -- the same coupling bug the fork's bo_taf fixed with its n_suggestions
    counter.
    """
    write_source(tmp_path, "srcA", seed=10)
    opt = MOTAFSequentialOptimizer(
        MOTAFConfig(
            bounds=BOUNDS, ref_point=REF, taf_run_dir=tmp_path, n_init=5,
            num_restarts=2, raw_samples=16,
            source_only_warmup_iters=1, seed=0,
        )
    )
    opt.bootstrap(objective)
    x = opt.suggest()
    assert opt.last_target_weight == 0.0  # first suggestion: prior only
    opt.observe(x, objective(x))
    x = opt.suggest()
    assert opt.last_target_weight > 0.0  # second suggestion: target term active
    opt.observe(x, objective(x))


def test_failed_suggest_does_not_consume_a_warmup_slot(tmp_path):
    (tmp_path / "gp_states").mkdir(parents=True, exist_ok=True)
    opt = MOTAFSequentialOptimizer(
        MOTAFConfig(bounds=BOUNDS, ref_point=REF, taf_run_dir=tmp_path, seed=0)
    )
    with pytest.raises(ValueError):
        opt.suggest()  # no observations yet
    assert opt.n_suggestions == 0


def test_per_objective_gp_state_is_honoured(tmp_path):
    """Sources may carry one hyperparameter set per objective; mismatched lists fail."""
    rng = np.random.default_rng(11)
    x = rng.random((10, D))
    y = objective(x)
    for sub in ("gp_states", "trajectories"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    (tmp_path / "trajectories" / "src.json").write_text(
        json.dumps({"x_values": x.tolist(), "y_values": y.tolist(),
                    "pareto_front": pareto_front(y).tolist()}), encoding="utf-8")
    per_obj = [
        {"kernel_type": "matern52", "lengthscale": [0.2] * D, "variance": 1.0, "noise": 1e-4},
        {"kernel_type": "rbf", "lengthscale": [0.6] * D, "variance": 0.5, "noise": 1e-3},
    ]
    (tmp_path / "gp_states" / "src.json").write_text(
        json.dumps({"gp_state": {"objectives": per_obj}}), encoding="utf-8")
    sources = _load_mo_source_surrogates(tmp_path, expected_m=M, expected_d=D)
    assert [s.name for s in sources] == ["src"]
    mu = sources[0].posterior_mean(rng.random((4, D)))
    assert mu.shape == (4, M)
    assert np.all(np.isfinite(mu))

    # Wrong list length -> warn-and-skip, not crash.
    (tmp_path / "gp_states" / "src.json").write_text(
        json.dumps({"gp_state": {"objectives": per_obj[:1]}}), encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _load_mo_source_surrogates(tmp_path, expected_m=M, expected_d=D) == []
    assert any("src" in str(w.message) for w in caught)


def test_reference_point_must_be_multi_objective():
    with pytest.raises(ValueError):
        MOBoTorchSequentialOptimizer(MOBoTorchConfig(bounds=BOUNDS, ref_point=[-1.0]))


def test_observe_rejects_wrong_objective_count():
    opt = MOBoTorchSequentialOptimizer(MOBoTorchConfig(bounds=BOUNDS, ref_point=REF))
    with pytest.raises(ValueError):
        opt.observe(np.zeros((2, D)), np.zeros((2, M + 1)))
