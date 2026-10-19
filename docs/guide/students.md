# Start here: OpenBO for students

[Guide index](README.md) · [Repository README](../../README.md)

OpenBO lets you study how an optimizer chooses experiments when each experiment
is expensive. Start with the small CPU examples below; no GPU, server, training
archive or previous BO experience is needed.

## 1. Install and check your environment

Install [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/),
then open a terminal in your checkout. On Windows, PowerShell works for all the
single-line commands on this page. Use the repository URL supplied by your course;
the maintainer's repository is shown here.

```sh
git clone https://github.com/yichiliao/openbo.git
cd openbo
uv sync --locked --python 3.14
uv run python -c "import sys, openbo; print(sys.version); print(openbo.__file__)"
```

Python **3.11 through 3.15** are included in the CI matrix. Choose a version with
`uv sync --locked --python 3.14` or `uv sync --locked --python 3.15`.
The repository's `.python-version` selects 3.14 when you do not specify a version;
to keep a different version for subsequent commands, run `uv python pin 3.14`
(this changes your local `.python-version`; do not commit that course preference).

For Python 3.15, uv automatically selects CPU PyTorch from its
[official wheel index](https://download.pytorch.org/whl/cpu).
[PyTorch documents this installation requirement](https://pytorch.org/blog/pytorch-2-14-release-blog/).
Other packages still come from PyPI. These examples use regular CPython and CPU
execution; free-threaded builds and GPU execution are outside the CI matrix.
If installing with pip instead of uv on 3.15, install PyTorch first with
`python -m pip install "torch>=2.14" --index-url https://download.pytorch.org/whl/cpu`,
then `python -m pip install -e .`.

Keep using `uv run` from the repository root. You do not need to activate the
environment or adjust PowerShell's execution policy. The first install downloads
scientific Python packages and PyTorch, so allow time and disk space for it.

## 2. Run your first comparison

```sh
uv run python examples/student_quickstart.py
```

This gives random search and BO the same 12 evaluations of Branin. It prints their
best values and saves:

- `test_results/student_quickstart/comparison.png`: best value versus evaluations.
- `test_results/student_quickstart/trajectories.json`: every input and observation.

Both methods start with the same three points for a given seed. Change the seed
and output folder to keep a second run:

```sh
uv run python examples/student_quickstart.py --seed 1 --output-dir test_results/student_seed1
```

One short run does not establish which algorithm is better. Compare multiple
seeds, budgets and functions. BO spends more computation choosing each point;
it is useful when evaluating the real objective costs much more.

## 3. Understand what the code is doing

An objective takes input rows `x` and returns observed scores `y`. BO repeats:

1. Fit a **surrogate** to observations collected so far.
2. Score candidate inputs with an **acquisition function**.
3. Evaluate the real objective at the selected input and add the observation.

A **Gaussian process (GP)** predicts both a mean and uncertainty. **Expected
improvement (EI)** balances promising predicted scores and uncertain regions.
The NumPy implementation is deliberately small enough to read alongside these
steps: [optimizer](../../src/openbo/optimizers/bo_scratch.py),
[GP](../../src/openbo/models/gp_scratch.py), [EI](../../src/openbo/acquisition/ei.py).

All objectives are **maximized**. Branin is usually described as a minimization
problem; OpenBO returns *negative* Branin, so a score closer to zero is better.
Synthetic inputs are normalized to `[0, 1]^d`.

| Array | Shape | Meaning |
| --- | --- | --- |
| `x` | `(n, d)` | `n` experiments with `d` input parameters |
| scalar `y` | `(n,)` | One score for each experiment |
| multi-objective `y` | `(n, M)` | `M` scores for each experiment |

For a loss or cost, return its negative. Convert physical parameter ranges into
the normalized cube before using the server or the synthetic examples. Use
`np.array([[0.2, 0.7]])` for one 2D experiment; the outer list is the batch axis.
Pass finite values; a failed experiment needs to be handled before calling
`observe`.

`n_init` is the random initial budget; `n_iter` counts subsequent suggestions.
Their sum is the total budget for the ordinary BO Python APIs. NAF and TAF+ use
their own budget conventions, described in their guides. Existing scalar
`best_y_history` arrays have one entry per observation *call*, so the initial
batch is one entry. To plot against individual evaluations, use
`np.maximum.accumulate(result.y_obs)`, as the quickstart does.

## 4. Learn interactively

Open the notebooks in VS Code or another Jupyter frontend and choose the
repository's `.venv` Python kernel. The first four notebooks are runnable lessons:

1. [BO from scratch](../../notebooks/01_bo_from_scratch.ipynb): the optimization loop and budgets.
2. [GP posterior](../../notebooks/02_gp_posterior_visualized.ipynb): mean and uncertainty.
3. [Acquisition functions](../../notebooks/03_ei_ucb_pi.ipynb): calculate EI, PI and UCB explicitly.
4. [BoTorch BO](../../notebooks/04_botorch_bo.ipynb): run the library-backed baseline.

Each notebook explains the conventions and can run independently. The later
meta-BO and preferential-BO notebooks remain placeholders; use the guides below
for implemented transfer methods. PI/UCB package modules are also placeholders;
lesson 3 implements the formulas in the notebook.

## 5. Choose a next step

| Goal | Where to go |
| --- | --- |
| Run and plot single-objective comparisons | [Single-function guide](single-function.md) |
| Make related tasks and avoid train/test leakage | [Families and splits](families.md) |
| Reuse saved source-task GPs | [TAF](taf.md) |
| Explore trade-offs and Pareto fronts | [Multi-objective BO](mobo.md) |
| Adapt across successive users | [ConBO](conbo.md) |
| Transfer users' data with objective weights and decay | [TAF+](taf-plus.md) |
| Train a learned acquisition function | [NAF](naf.md) |
| Connect an experiment running in another application | [Server protocols](servers.md) |

Transfer learning uses **sources** (previous tasks) to help a **target** (new
task). Train sources on the training split and evaluate on held-out tasks.
Similar source rankings do not guarantee helpful transfer; compare with ordinary
BO on the same targets. Multi-objective optimization keeps a **Pareto front** of
trade-offs and measures its coverage with **hypervolume** relative to a fixed
pessimistic reference point. Larger hypervolume is better; keep the reference
point unchanged when comparing runs.

## Common problems

| Symptom | What to check |
| --- | --- |
| `uv` is not recognized | Restart the terminal after installing uv; follow its installation instructions. |
| `No module named openbo` | Run `uv sync --locked` in the checkout and use `uv run python`. |
| Python 3.15 cannot install PyTorch | Use the current checkout's `uv sync --locked --python 3.15`; the PyTorch index configuration is required. |
| Missing config or source artifacts | Run from the repository root; TAF needs both matching `gp_states/` and `trajectories/`. |
| PowerShell rejects multiline guide commands | Bash uses `\` for continuation. Use a single line, as on this page. |
| Plot appears inverted | Every objective is maximized; minimization objectives must be negated. |
| Notebook imports fail | Select `.venv` as its kernel, then restart the kernel. |

Run `uv run pytest tests` to verify behavior, or start with
`uv run pytest tests/gp_scratch_test.py tests/acquisition_test.py` for a smaller
check. The full suite includes optimizer fits and neural training tests and takes
longer. See [CONTRIBUTING](../../CONTRIBUTING.md) for development checks.
