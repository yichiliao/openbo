"""Source loading, softplus overflow and logging for TAF."""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

from openbo.acquisition.taf import SourceTaskSurrogate, taf_m_acquisition
from openbo.models.gp_scratch import GPScratch
from openbo.optimizers.bo_taf import _load_source_surrogates, run_bo_taf
from openbo.test_functions.registry import get_function_spec

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_source(run_dir: Path, relative: str, seed: int, *, trajectory: bool = True) -> float:
    """Write one source task in the layout of ``scripts/train_taf.py`` under ``relative``.

    Returns the best value of its trajectory, which differs between seeds and so identifies the task.
    """
    rng = np.random.default_rng(seed)
    x = rng.random((5, 2))
    y = rng.normal(size=5) + 10.0 * seed
    gp = GPScratch(optimize_hyperparameters=False)
    gp.fit(x, y)
    gp_payload = {
        "gp_state": {
            "kernel_type": gp.kernel_type,
            "lengthscale": [float(v) for v in np.asarray(gp.lengthscale).reshape(-1)],
            "variance": float(gp.variance),
            "noise": float(gp.noise),
            "standardize_targets": bool(gp.standardize_targets),
            "optimize_noise": bool(gp.optimize_noise),
        }
    }
    gp_path = run_dir / "gp_states" / relative
    gp_path.parent.mkdir(parents=True, exist_ok=True)
    gp_path.write_text(json.dumps(gp_payload), encoding="utf-8")
    if trajectory:
        traj_path = run_dir / "trajectories" / relative
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_path.write_text(json.dumps({"x_values": x.tolist(), "y_values": y.tolist()}), encoding="utf-8")
    return float(np.max(y))


def test_loader_reads_nested_task_folders(tmp_path: Path) -> None:
    """An older branin_train_v1 kept most of its tasks in a subfolder; all of them must load."""
    _write_source(tmp_path, "task_000.json", 0)
    _write_source(tmp_path, "sub/task_001.json", 1)
    _write_source(tmp_path, "sub/task_002.json", 2)
    assert sorted(s.name for s in _load_source_surrogates(tmp_path)) == ["task_000", "task_001", "task_002"]


def test_loader_warns_about_a_task_without_a_trajectory(tmp_path: Path) -> None:
    _write_source(tmp_path, "task_000.json", 0)
    _write_source(tmp_path, "task_001.json", 1, trajectory=False)
    with pytest.warns(UserWarning, match="task_001.json.*no matching trajectory"):
        sources = _load_source_surrogates(tmp_path)
    assert [s.name for s in sources] == ["task_000"]


def test_committed_branin_run_loads_all_twelve_sources() -> None:
    run_dir = REPO_ROOT / "meta-bo-training" / "taf-gps" / "branin_train_v1"
    if not run_dir.exists():
        pytest.skip("the committed TAF run is not available")
    assert len(_load_source_surrogates(run_dir)) == 12


def test_softplus_source_improvement_does_not_overflow() -> None:
    """exp(z) overflows for z > 709; softplus(z) is then z, and the acquisition must stay finite."""
    x = np.array([[0.1, 0.1], [0.6, 0.4], [0.9, 0.8]])
    y = np.array([-1.0, 0.3, 0.2])
    gp = GPScratch(optimize_hyperparameters=False)
    gp.fit(x, y)
    # A far-away reference makes (mean - reference) / temperature about 1e8.
    source = SourceTaskSurrogate(
        name="far", gp=gp, best_y=0.3, meta_features=np.zeros(2), reference_y=-1.0e6
    )
    kwargs = dict(
        x=np.array([[0.2, 0.8], [0.5, 0.5]]),
        target_gp=None,
        target_best_y=0.3,
        source_surrogates=[source],
        source_weights=np.array([1.0]),
        target_weight=0.0,
        source_improvement_temperature=0.01,
    )
    soft = taf_m_acquisition(source_improvement_mode="softplus", **kwargs)
    relu = taf_m_acquisition(source_improvement_mode="relu", **kwargs)
    assert np.all(np.isfinite(soft))
    np.testing.assert_allclose(soft, relu, rtol=1e-9)


def test_taf_does_not_print_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_source(tmp_path, "task_000.json", 0)
    spec = get_function_spec("branin")
    run_bo_taf(objective=spec.objective, bounds=spec.bounds, taf_run_dir=tmp_path, n_init=0, n_iter=4, seed=0)
    assert capsys.readouterr().out == ""


def test_tasks_with_the_same_name_in_different_folders_keep_their_own_data(tmp_path: Path) -> None:
    """Pairing uses the relative path, so a same-named task never takes another folder's trajectory."""
    expected = [
        _write_source(tmp_path, "task.json", 1),
        _write_source(tmp_path, "a/task.json", 2),
        _write_source(tmp_path, "b/task.json", 3),
    ]
    sources = _load_source_surrogates(tmp_path)
    assert [s.name for s in sources] == ["task", "task", "task"]  # the name is the file stem only
    assert sorted(s.best_y for s in sources) == pytest.approx(sorted(expected))


def test_a_gp_without_a_trajectory_in_its_own_folder_is_not_paired_with_another_folder(tmp_path: Path) -> None:
    _write_source(tmp_path, "a/task.json", 1, trajectory=False)
    own = _write_source(tmp_path, "b/task.json", 2)
    with pytest.warns(UserWarning, match="'a/task.json'.*no matching trajectory"):
        sources = _load_source_surrogates(tmp_path)
    assert [s.best_y for s in sources] == pytest.approx([own])


def test_a_backup_copy_inside_gp_states_loads_twice_and_warns(tmp_path: Path) -> None:
    _write_source(tmp_path, "task.json", 1)
    for folder in ("gp_states", "trajectories"):
        (tmp_path / folder / "backup").mkdir()
        shutil.copy(tmp_path / folder / "task.json", tmp_path / folder / "backup" / "task.json")
    with pytest.warns(UserWarning, match="same data as .*counts twice") as record:
        sources = _load_source_surrogates(tmp_path)
    assert len(sources) == 2  # still loaded: the warning makes it visible, it does not change the result
    message = str(record[0].message)
    assert "'task.json'" in message and "'backup/task.json'" in message


def test_distinct_tasks_in_nested_folders_do_not_warn(tmp_path: Path) -> None:
    _write_source(tmp_path, "task_000.json", 1)
    _write_source(tmp_path, "sub/task_001.json", 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(_load_source_surrogates(tmp_path)) == 2


def test_only_gp_states_is_scanned_and_malformed_sources_are_skipped(tmp_path: Path) -> None:
    _write_source(tmp_path, "task_000.json", 1)
    # Other folders and files of the run are ignored, without a warning.
    (tmp_path / "gp_predictions").mkdir()
    (tmp_path / "gp_predictions" / "other.json").write_text("{}", encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(_load_source_surrogates(tmp_path)) == 1
    # A malformed source must not prevent the valid sources from loading.
    (tmp_path / "gp_states" / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "trajectories" / "broken.json").write_text("{}", encoding="utf-8")
    with pytest.warns(UserWarning, match="broken.*malformed artifacts"):
        assert len(_load_source_surrogates(tmp_path)) == 1
