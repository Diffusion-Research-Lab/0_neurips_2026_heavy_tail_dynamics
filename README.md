## Heavy-Tail Flow Matching

Official code release for **Heavy-Tail Flow Matching** (NeurIPS 2026).

This repository provides:
- reference implementations of heavy-tailed bridges / noise models,
- training code for flow matching baselines and heavy-tail variants,
- evaluation metrics for heavy-tailed generative modeling,
- scripts to reproduce the main figures and tables.

### Authors

- Hamza Cherkaoui

### Results reproduction

To reproduce benchmarks locally:

    bash benchmarks/01_setup.sh --check
    bash benchmarks/03_launcher.sh --run --venv-dir .venv

To reproduce benchmarks on Slurm:

    sbatch benchmarks/02_launcher.slurm

Detailed benchmark instructions are available in:

    benchmarks/00_README.md
