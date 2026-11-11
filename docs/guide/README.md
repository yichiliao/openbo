# OpenBO guide

[← README](../../README.md) · [Website](https://www.csie.ntu.edu.tw/~yichi/openbo-website/index.html)

The command reference for OpenBO: every script, flag, server protocol and archived run. The [website](https://www.csie.ntu.edu.tw/~yichi/openbo-website/index.html) has the overview, what each method does, and a short "how to call it" for each one ([Methods](https://www.csie.ntu.edu.tw/~yichi/openbo-website/methods.html)). Run every command from the repository root, with `uv run` or an activated `.venv`.

**Benchmarks on test functions**

- [Single-function benchmarks](single-function.md): run and compare optimizers on one test function, noise, 2D search plots, log-regret plots.
- [Families of related tasks](families.md): task variants, train/test splits, family benchmarks and plots.

**Methods**

- [TAF: transfer from source tasks](taf.md): train source GPs, test TAF on held-out tasks, and the end-to-end TAF workflow through the servers.
- [Multi-objective BO](mobo.md): CEIPV/IEIPV with correlated GPs and a qLogNEHVI baseline; benchmark and server.
- TAF-MOBO, transfer for multi-objective BO from earlier tasks or users, in two implementations (one [server](servers.md#taf-mobo-server-protocol) serves both):
  - [TAF-EIPV](taf-eipv.md) (recommended): the original TAF on OpenBO's multi-objective BO, starting from the sources alone.
  - [TAF-EHVI](taf-ehvi.md): on BoTorch's qLogNEHVI, contributed by Mark Colley.
- [Continual BO (ConBO)](conbo.md): a population model across a sequence of users; Python, benchmark and server.
- [TAF+](taf-plus.md): transfer across users with several objectives; workflow, simulations and server.
- [NAF (MetaBO)](naf.md): train a neural acquisition function with PPO, benchmark it, serve it.
- [NAF+](naf-plus.md): a neural AF that reads objective weights, with a novelty detector that falls back to EI (after CHI 2026).

**Integration**

- [WebSocket servers](servers.md): starting each server, config files, the JSON protocol, per-server differences, client loops and fake clients.

**Reference**

- [Reproducing the TAF+ and ConBO papers](paper-reproduction.md): every simulation of both papers, 5 seeds, and how to rerun them.
- [Teaching notebooks](../../notebooks/): five notebooks from a hand-written BO loop to TAF, ConBO and NAF.
- [Results and archived benchmarks](results.md): where outputs go, and what each archive in `benchmark_results/` holds.
- [Project structure](project-structure.md): every package, script and folder, and what is still a stub.
