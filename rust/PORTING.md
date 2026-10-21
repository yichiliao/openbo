# Port coverage and numerical backends

This crate ports the implemented OpenBO workflows from upstream
[`3ceda2d34225d2f1601fc57815cd821fbc78f37d`](https://github.com/yichiliao/openbo/commit/3ceda2d34225d2f1601fc57815cd821fbc78f37d).
It is a separate native library and command-line program in `rust/`. Python stays
available in the parent project. Start with [README.md](README.md); the test record
is in [VALIDATION.md](VALIDATION.md).

## Coverage map

| Python area | Native implementation | Scope |
| --- | --- | --- |
| `test_functions/synthetic`, registry and transforms | `core` | Branin, sphere, Ackley, Rastrigin, Rosenbrock, Hartmann-6, McCormick; Gaussian noise, capping, normalization and variants |
| Task families and train/test splits | `core` | Persisted variants, disjoint seeded splits, bounded reachable optima, simulated sphere users |
| MetaBO families | `core` | Normalized Branin, log Goldstein–Price, Hartmann-3; sampled and grid task families |
| Multi-objective problems | `core` | OKA2, VLMOP3, DTLZ1a, three spheres and correlated SLF-GP objective draws |
| NumPy scalar GP and kernels | `core::GaussianProcess` | ARD RBF/Matérn-5/2, constant means, outcome scaling, marginal/joint latent predictions, noise variance, likelihood fitting and state interchange |
| Multi-output GP and slice sampling | `acquisition::MultiOutputGP` | Independent, multitask and SLF covariance models, MAP fitting, posterior joint predictions and slice-sampled hyperparameters |
| EI and transfer acquisition | `acquisition` | EI/logEI, PI, UCB, TAF-M/R, Pareto ranking, PoE and TAF+ with objective weights and decay |
| Pareto geometry and MO acquisitions | `acquisition` | Exact hypervolume and cells, Monte Carlo EHVI/NEHVI, IEIPV, moment-matched CEIPV and printed-paper EIPV form |
| Random, scratch and BoTorch BO wrappers | `optimizers` | Unified scalar/MO ask/tell; native EI or Monte Carlo NEHVI; seeded candidate pools and bounded refinement |
| Scalar/MO TAF and TAF+ | `optimizers` | Source loading, weights, source warmup, decay, target fallback, per-user objective GPs and histories |
| TAF-EIPV and TAF-MOBO server | `acquisition`, `optimizers`, `server` | Source gain against current target designs, pooled per-objective ranking weights, stable log blend, native target IEIPV/CEIPV, source-only starts and total-budget server sessions |
| TAF+ workflow | `optimizers` | Positive simplex weight grid, rating-based weight selection, leave-one-user-out decay search |
| Population BNN and generative replay | `neural` | Actual heteroscedastic NLL/backpropagation, Adam, MC dropout, mean/variance replay and uncertainty filtering |
| ConBO | `optimizers::ConboPopulation` | Successive users, reduced initialization, population/user mixed EI, online adaptation, replay, reset/continued training and native persistence |
| Neural acquisition functions | `neural` | Feature construction, adaptive candidates, policy/value networks, PPO, GAE, deterministic parallel rollouts, checkpoint resume and best-policy selection |
| Server adapters | `server` | JSON WebSocket ask/tell for native optimizers, ConBO and NAF, range/budget validation, isolated sessions and completed-user persistence |
| Scripts and plots | CLI | Consolidated experiments, source training/inspection, families, NAF training/import, ConBO, weight/decay workflows, client/server and SVG convergence plots |

PI and UCB are implemented as library utilities even though the corresponding
Python modules are placeholders. The Rust CLI is consolidated; it does not mirror
every Python script or flag. SLF-GP draws, custom noisy objectives, user simulations,
parallel NAF settings and advanced acquisition choices are available through the
library. The CLI's built-in NAF trainer uses the supplied MetaBO families.

## What is numerically comparable

Fixed-input objective formulas, scalar GP posteriors with fixed hyperparameters,
EI, transfer weights and hypervolume have explicit Python comparison tests.
Multi-output covariance definitions, CEIPV terms and neural forward passes also
have fixed reference tests. All optimization maximizes objectives; prediction
variances are latent variances and GP noise is an observation variance.

These comparisons have stated floating-point tolerances. They do not imply
identical learning curves. The native backend choices are:

- **Linear algebra:** float64 `nalgebra` factorizations, with numerical jitter and
  validation before committing fitted model state. Scalar hyperparameter search
  uses bounded numerical gradients/BFGS; multi-output fitting uses bounded
  coordinate MAP search. Neither invokes SciPy or GPyTorch.
- **Acquisition search:** seeded low-discrepancy candidates, followed by bounded
  coordinate refinement from several starts for scalar methods. The `grid`
  selector scans candidates without refinement. Search differs from SciPy or
  BoTorch L-BFGS-B and BoTorch's acquisition optimizer.
- **BoTorch selectors:** `bo_botorch` runs native EI; `mobo_botorch` runs native
  Monte Carlo noisy expected hypervolume improvement. It samples baseline and
  candidate values jointly. It does not reproduce BoTorch qLogNEHVI's smoothing,
  log-space machinery, GPU execution or fitted model trajectories.
- **Randomness:** ChaCha-based streams and native digitally shifted Sobol points
  (Halton fallback in higher dimensions) differ from NumPy, SciPy and PyTorch.
  Matching a seed across languages does not match samples or trajectories.
- **MO defaults:** explicit `mobo_ceipv_slf`/`mobo_ceipv_mt` selectors choose their
  named covariance model. Generic `mobo_ceipv` and `mobo_ieipv` use `config.model`;
  its native default is independent. Slice averaging uses native sampling
  settings rather than the paper's full experimental protocol.
- **Neural training:** native float64 networks and Adam train without libtorch.
  A Python float32 policy can pass inference parity after export, while subsequent
  training and dropout random streams differ. No GPU backend is provided.

Use repeated seeds and matched evaluation budgets for experiments. Reproducing
published benchmark numbers requires a separate experimental study. Historical
archives are retained in the Python project and have not been relabeled as Rust
results.

## Persistence and interchange

Source populations use paired `gp_states/<name>.json` and
`trajectories/<name>.json`, including nested folders. The loader understands
Python scalar states and per-objective states. It conditions on the paired data,
preserves means/scaling/hyperparameters, reports malformed pairs, and skips them.
Read `warnings` when inspecting an imported population. Custom source coordinates
must follow the same normalization convention as the target experiment.

Rust result files contain objective rows, histories for every evaluation and
backend metadata; standard optimizer results also retain their configuration.
They are a native schema, not replacements for every archived Python result file.
`suggestion_seconds` and `total_suggestion_seconds` measure successful suggestion
calls. CLI `elapsed_seconds` measures the complete run (for ConBO, a user plus
population training). These wall-clock values are not deterministic seed outputs.

Native neural and ConBO checkpoints are JSON with validation on load. A native
NAF training checkpoint includes Adam and RNG state for exact native resume.
Python `.pt` optimizer/training checkpoints are not native resume checkpoints.
The optional [export tool](tools/export_python_naf.py) converts a trusted Python
policy to portable tensors and records Python logits. `import-naf` verifies those
logits before writing the native policies. It does not claim cross-language PPO
training-state resume or import Python ConBO BNN weights.

The server supports native JSON/YAML and the common Python server configuration
keys, including engine inference for the repository's dedicated server YAMLs.
Native defaults and search controls still apply; unsupported MO acquisitions are
rejected. The shared JSON message loop is documented in [README.md](README.md#servers).
Multiple ordinary sessions are isolated. ConBO serializes active users within a
server process; separate processes must use separate population directories.

## Corrections made during the port

The Python implementation received regression-tested corrections alongside the
port:

- GP refits commit only after success, preserve the original model on invalid or
  nonfinite intermediate state, accept zero-noise likelihoods, and retain a better
  starting solution when bounded hyperparameter optimization cannot improve it.
- Stationary kernels validate finite inputs/hyperparameters and handle extreme
  distances without `inf * 0` NaNs or overflowing subtraction.
- PPO best-policy checkpoints retain the policy that produced the measured
  rollout return, before the next update. NAF observations/episode steps reject
  invalid shapes, nonfinite values and exhausted budgets before mutation.

Native regression tests also cover rejected observations preserving pending
experiments, overflow-safe midpoint/rating calculations, source and checkpoint
validation, correct transfer decay timing, Python YAML engine selection, explicit
CLI configuration overrides, and persistence failures that can be retried.

Upstream placeholder aggregation/plotting modules, empty teaching notebooks,
MetaBO GP-prior/real-world task datasets and website assets are not new Rust
features. The native student guide and executable example are included instead.
