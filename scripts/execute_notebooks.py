"""Execute the teaching notebooks in `notebooks/` and save them with their outputs.

Run it after changing code that a notebook uses, so the stored outputs match the code:

    uv run python scripts/execute_notebooks.py                 # every notebook that has cells
    uv run python scripts/execute_notebooks.py notebooks/01_bo_from_scratch.ipynb

Each notebook is executed from the `notebooks/` folder in a fresh `python3` kernel and
rewritten in place with its outputs. The script stops at the first notebook that raises an error and
leaves that file untouched. Needs `nbformat` and `nbclient` (in the `dev` group).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def execute(path: Path, timeout: int) -> float:
    """Execute one notebook in place and return the seconds it took."""
    notebook = nbformat.read(path, as_version=4)
    start = time.perf_counter()
    NotebookClient(notebook, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    return time.perf_counter() - start


def main() -> None:
    """Parse arguments and execute the requested notebooks."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="*", type=Path, help="Notebook files (default: all with at least one cell).")
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds allowed per cell.")
    args = parser.parse_args()

    paths = args.notebooks or sorted(
        p for p in Path("notebooks").glob("*.ipynb") if nbformat.read(p, as_version=4).cells
    )
    if not paths:
        parser.error("no notebooks to execute.")
    for path in paths:
        print(f"executing {path} ...", flush=True)
        print(f"  done in {execute(path, args.timeout):.0f} s", flush=True)


if __name__ == "__main__":
    main()
