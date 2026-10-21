# OpenBO

OpenBO is a research and teaching Python library for Bayesian optimization (BO): standard BO (from scratch in NumPy, and with BoTorch), transfer and meta-BO (TAF, TAF+, continual BO, MetaBO-style neural acquisition functions) and multi-objective BO, including multi-objective transfer (TAF-MOBO). Every optimizer has a Python ask/tell interface, benchmark scripts and a WebSocket server for human-in-the-loop studies.

**Website: [https://www.csie.ntu.edu.tw/~yichi/openbo-website/](https://www.csie.ntu.edu.tw/~yichi/openbo-website/index.html)**, with the overview, [how to call each method](https://www.csie.ntu.edu.tw/~yichi/openbo-website/methods.html), [interactive demos](https://www.csie.ntu.edu.tw/~yichi/openbo-website/demos.html), [benchmark results](https://www.csie.ntu.edu.tw/~yichi/openbo-website/benchmarks.html) and [how to cite](https://www.csie.ntu.edu.tw/~yichi/openbo-website/publications.html).

## Installation

OpenBO needs Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/yichiliao/openbo.git
cd openbo
uv sync                # create .venv and install the locked dependencies
uv run pytest tests    # optional: the full test suite takes a few minutes
```

Run commands with `uv run ...` (or `source .venv/bin/activate` first), from the repository root: default config and output paths are relative to it.

## A first run

```python
from openbo.optimizers.bo_scratch import run_bo_scratch
from openbo.test_functions.registry import get_function_spec

spec = get_function_spec("branin")  # y = -Branin(x), x in [0, 1]^2; every objective is maximized
result = run_bo_scratch(spec.objective, bounds=[(0.0, 1.0)] * spec.dim, n_init=5, n_iter=20, seed=0)
print(result.best_y_history[-1], spec.optimum)
```

## Guide

The full command reference is in [`docs/guide/`](docs/guide/README.md):

| Topic | Guide |
| --- | --- |
| Benchmarks on one test function | [single-function.md](docs/guide/single-function.md) |
| Families of related tasks, train/test splits | [families.md](docs/guide/families.md) |
| TAF (transfer acquisition function) | [taf.md](docs/guide/taf.md) |
| Multi-objective BO | [mobo.md](docs/guide/mobo.md) |
| TAF-MOBO (transfer for multi-objective BO): TAF-EIPV, on OpenBO's MOBO (recommended) | [taf-eipv.md](docs/guide/taf-eipv.md) |
| TAF-MOBO: TAF-EHVI, on BoTorch | [taf-ehvi.md](docs/guide/taf-ehvi.md) |
| Continual BO (ConBO) | [conbo.md](docs/guide/conbo.md) |
| TAF+ | [taf-plus.md](docs/guide/taf-plus.md) |
| NAF (MetaBO) | [naf.md](docs/guide/naf.md) |
| WebSocket servers and their protocols | [servers.md](docs/guide/servers.md) |
| Output folders and archived benchmarks | [results.md](docs/guide/results.md) |
| Project structure and stubs | [project-structure.md](docs/guide/project-structure.md) |

Design notes for each method (decisions, deviations from the papers, open questions) are in [`docs/plans/`](docs/plans/).

## License and citation

OpenBO is licensed under the MIT License; see [LICENSE](LICENSE). If you use OpenBO in your research, please cite it:

```bibtex
@software{liao2026openbo,
  author    = {Yi-Chi Liao and Mark Colley and George Mo and Chih-Hao Hsu and Kuan-Lin Lee},
  title     = {{OpenBO}: An Open-Source Library for Bayesian Optimization},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/yichiliao/openbo}
}
```

Please also cite the paper behind each method you use: they are listed on the website's [Publications](https://www.csie.ntu.edu.tw/~yichi/openbo-website/publications.html) page, with BibTeX entries in [`relevant_papers/references.bib`](relevant_papers/references.bib).

OpenBO is maintained by Yi-Chi Liao (yichi@csie.ntu.edu.tw). Visit my group website: <https://www.csie.ntu.edu.tw/~yichi/>.
