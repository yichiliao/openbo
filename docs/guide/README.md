# OpenBO guide

[← README](../../README.md) · [Website](https://www.csie.ntu.edu.tw/~yichi/openbo-website/index.html)

The command reference for OpenBO: every script, flag, server protocol and archived run. The [website](https://www.csie.ntu.edu.tw/~yichi/openbo-website/index.html) has the overview, what each method does, and a short "how to call it" for each one ([Methods](https://www.csie.ntu.edu.tw/~yichi/openbo-website/methods.html)). Run every command from the repository root, with `uv run` or an activated `.venv`.

**Benchmarks on test functions**

- **[Student walkthrough](students.md)**: install, run a small comparison, learn the conventions and work through the first four notebooks.

- [Single-function benchmarks](single-function.md): run and compare optimizers on one test function, noise, 2D search plots, log-regret plots.
- [Families of related tasks](families.md): task variants, train/test splits, family benchmarks and plots.

**Methods**

- [TAF: transfer from source tasks](taf.md): train source GPs, test TAF on held-out tasks, and the end-to-end TAF workflow through the servers.
- [Multi-objective BO](mobo.md): CEIPV/IEIPV with correlated GPs; benchmark and server.
- [Continual BO (ConBO)](conbo.md): a population model across a sequence of users; Python, benchmark and server.
- [TAF+](taf-plus.md): transfer across users with several objectives; workflow, simulations and server.
- [NAF (MetaBO)](naf.md): train a neural acquisition function with PPO, benchmark it, serve it.

**Integration**

- [WebSocket servers](servers.md): starting each server, config files, the JSON protocol, per-server differences, client loops and fake clients.

**Reference**

- [Results and archived benchmarks](results.md): where outputs go, and what each archive in `benchmark_results/` holds.
- [Project structure](project-structure.md): every package, script and folder, and what is still a stub.
