"""The committed teaching notebooks are valid, non-empty and were saved without errors."""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

NOTEBOOKS = sorted(Path(__file__).resolve().parents[1].glob("notebooks/*.ipynb"))
FILLED = [p for p in NOTEBOOKS if nbformat.read(p, as_version=4).cells]


def test_there_are_notebooks() -> None:
    assert NOTEBOOKS, "notebooks/ holds no .ipynb files"


@pytest.mark.parametrize("path", FILLED, ids=lambda p: p.name)
def test_notebook_is_valid_and_has_clean_outputs(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert code_cells and any(c.cell_type == "markdown" for c in notebook.cells)
    # Saved after a full execution: every code cell ran, none ended in an error.
    assert all(c.execution_count is not None for c in code_cells), f"{path.name} has cells that were never executed"
    errors = [o for c in code_cells for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors, f"{path.name} was saved with an error output: {errors[0].get('ename')}"
    text = path.read_text(encoding="utf-8")
    assert "/Users/" not in text and "/home/" not in text, f"{path.name} contains a local path in its outputs"
