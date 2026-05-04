# OpenMetaBO

OpenMetaBO is a research + teaching Python library for Bayesian optimization (BO) and meta-BO.

## Current scope

This repository currently includes a first vertical slice:

- synthetic test functions (`Branin`, `Sphere`) in normalized input space `[0, 1]^d`
- random search baseline
- BO from scratch (NumPy GP + EI)
- BO with BoTorch (`SingleTaskGP` + `LogExpectedImprovement`)
- simple benchmark runner and plotting script

The code is intentionally simple and teaching-oriented.

## Quickstart

Run tests:

```bash
uv run pytest
```

Run benchmarks:

```bash
uv run python scripts/run_benchmark.py --method random
uv run python scripts/run_benchmark.py --method bo_scratch
uv run python scripts/run_benchmark.py --method bo_botorch
```

Compare methods and save a plot:

```bash
uv run python scripts/plot_results.py
```

This writes `benchmark_y_values.png` with line plots of `iteration` vs observed `y`.

## Project structure

- `src/metabo/` - core library code (test functions, models, acquisition, optimizers, benchmarks)
- `scripts/` - runnable scripts for benchmarking, aggregation, and plotting
- `tests/` - unit tests for functions, GP, acquisitions, and optimizer smoke tests
- `configs/` - lightweight config files for benchmark/method settings
- `notebooks/` - teaching notebooks for BO concepts and methods
