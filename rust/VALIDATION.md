# Validation record

Validated on **2026-10-21**, Windows x86-64 MSVC, against upstream
`a22d861401a157e2379bdb44355323819ea05414` plus the bug fixes described in
[PORTING.md](PORTING.md#corrections-made-during-the-port).

## Reproduce

From `rust/`:

```sh
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
cargo test --locked --doc
cargo doc --locked --no-deps
cargo build --locked --release
cargo run --locked --release -- quickstart
cargo run --locked --release --example student
cargo package --locked --allow-dirty
```

From the parent Python checkout with its development dependencies installed:

```sh
uv run python rust/tools/validate_python_parity.py --build
uv run pytest tests -q
uv run ruff check src/openbo/models/gp_scratch.py src/openbo/models/kernels.py src/openbo/optimizers/naf.py tests/gp_port_regression_test.py tests/naf_validation_test.py rust/tools
```

The parity runner accepts `--binary PATH` and `--report PATH`; its default JSON
report is `test_results/rust_parity.json`. Python is needed only for these optional
cross-language checks and for exporting existing Python policies.

## Evidence

- **Rust regression suite:** core objectives and task families, scalar and
  correlated GP behavior, acquisition geometry and sampling, optimizers, source
  interchange, population replay, neural gradients/training/resume, CLI workflows
  and real WebSocket connections. Tests run without Python or external datasets.
- **Cross-language comparison:** 68 numerical groups, **2,639 scalar values**
  compared with the current Python implementation. This includes seven scalar
  functions, three MetaBO normalization formulas, three multi-objective functions,
  16 scalar GP configurations (RBF/Matérn, constant/nonconstant targets,
  standardization on/off, zero/positive noise), joint covariance, EI, hypervolume
  in two through four dimensions, and meta-feature weights.
- **Trained policy interchange:** the archived Python Branin policy was exported
  and imported successfully. All **48 recorded logits** passed the export's
  float32/float64 tolerances; maximum absolute difference was
  `0.0003397758346181945` (absolute tolerance `0.0005`, relative `0.00002`).
- **Python regressions:** **542 tests passed**, including the new GP/kernel and
  neural checkpoint/input validation cases. The suite emitted 68 existing
  numerical/benchmark warnings and had no failures. Python version: 3.14.6.
- **Static checks:** Rust formatting and Clippy with warnings denied, Python Ruff
  on changed code/tools, and Mypy on changed implementation modules with missing
  third-party stubs ignored. Local documentation links and notebook syntax are
  checked by `scripts/check_docs.py`.

The tests include analytical EI and hypervolume cases, numerical GP reference
values, finite-difference neural/likelihood gradients, Monte Carlo checks against
closed forms, a real PPO learning test, exact native checkpoint continuation,
malformed input rejection, isolated concurrent WebSocket sessions, and retries
after a deliberately failed population save.

The CLI tests execute the student comparison, scalar/MO optimization, split →
source training → inspection → transfer, NAF training → resume → normalized
benchmark, policy import/rejection and a persisted two-user ConBO sequence.

## Limits

This record establishes behavior covered by the tests. It is not a proof that
every numerical input or experimental setup is correct. In particular:

- Optimized hyperparameters, RNG streams, acquisition search and neural training
  are native algorithms; their complete trajectories are not expected to match
  Python. The differences are explicit in [PORTING.md](PORTING.md).
- CI is configured for stable Rust on Linux, Windows and macOS. Those hosted jobs
  must run after pushing; local validation does not establish their results.
- Release builds and small workflows are exercised locally. Large paper-scale
  benchmark reproduction, GPU acceleration and Python training-checkpoint resume
  are outside this validation record.
- Rust runs independently of Python. Python 3.15 compatibility of the parent
  project is separate from native Rust validation; this run used Python 3.14.6.

The locked dependency graph and declared minimum Rust version make the build
constraints reviewable. Keep `Cargo.lock` when reproducing this record.
