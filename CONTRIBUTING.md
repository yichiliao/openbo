# Contributing to OpenBO

Start with the [student walkthrough](docs/guide/students.md) for installation and
the [project map](docs/guide/project-structure.md) for implemented methods and stubs.
Run commands from the repository root. Contributions should keep objectives in
maximization form and preserve the documented array shapes and budget conventions.

## Before opening a pull request

```sh
uv sync --locked
uv run ruff check src scripts server_scripts examples tests
uv run mypy src/openbo --ignore-missing-imports
uv run python scripts/check_docs.py --execute-notebooks
uv run python examples/student_quickstart.py --n-evals 5
uv run pytest tests
uv build
```

The full suite includes numerical optimization, server sessions and small neural
training runs. To iterate on one method, first run its relevant `tests/*_test.py`
files. Add regression coverage when changing numerical behavior, persistence or
an API; a short regression should reproduce the original problem.

CI installs the locked environment and runs the full suite on Python 3.11–3.15
on Linux and Windows. It also checks the runnable teaching material, documentation
links, syntax/lint and wheel/sdist builds. Free-threaded Python and accelerator
execution are not currently covered. A green CI run does not establish that a
paper's results have been reproduced; benchmark changes need comparable seeds,
budgets, train/test splits and objective conventions.

Use `uv sync --locked --python 3.15` to check that interpreter locally, then
`uv python pin 3.15` to retain the selection for `uv run`. Do not commit a local
interpreter preference unless you intend to change the repository default.

## Documentation and experiment artifacts

- Keep the README short; put instructions in `docs/guide/` and link them from its index.
- Update the method guide when flags, defaults, protocols or result fields change.
- Put ordinary generated results in ignored `test_results/`; do not overwrite the
  maintainer's archived results or trained checkpoints while testing.
- Keep teaching notebooks runnable from a fresh kernel. Clear outputs before
  committing; include the objective sign, shapes and evaluation budget.
- Include the problem, resulting behavior and validation in pull requests. Avoid
  unrelated reformatting and dependency upgrades.
