## FlowBench Framework

Official code release for **FlowBench**.

[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/hcherkaoui/neurips_2026_heaytail_flow_matching/actions/workflows/ci.yml/badge.svg)](https://github.com/hcherkaoui/neurips_2026_heaytail_flow_matching/actions/workflows/ci.yml)
[![flake8](https://github.com/hcherkaoui/neurips_2026_heaytail_flow_matching/actions/workflows/flake8.yml/badge.svg)](https://github.com/hcherkaoui/neurips_2026_heaytail_flow_matching/actions/workflows/flake8.yml)
[![Smoke](https://github.com/hcherkaoui/neurips_2026_heaytail_flow_matching/actions/workflows/smoke.yml/badge.svg)](https://github.com/hcherkaoui/neurips_2026_heaytail_flow_matching/actions/workflows/smoke.yml)
![maintenance-status](https://img.shields.io/badge/maintenance-active-brightgreen.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FlowBench provides:
- heavy-tailed samplers and generative model baselines,
- training/evaluation utilities for flow and diffusion models,
- benchmark scripts for synthetic settings.

### Install

```bash
pip install -e .
```

Or:

```bash
make install
```

Developer install:

```bash
pip install -e .[dev]
```

Benchmark install:

```bash
pip install -e .[dev,bench]
```

Optional vendor backends:

```bash
bash scripts/fetch_vendor.sh
```

### Quickstart

Check imports:

```bash
python -c "import genkit, labkit; print(genkit.__name__, labkit.__name__)"
```

Run smoke test:

```bash
python benchmarks/_smoke_test.py
```

Run tiny example:

```bash
python examples/02_visu_2d.py --blank
```

### Testing

```bash
pytest -q
```

Or:

```bash
make test
```

### Benchmarks

Blank/local pipeline:

```bash
make bench-blank
```

Full local run:

```bash
make bench-run
```

Slurm:

```bash
sbatch benchmarks/02_launcher.slurm
```

### Developer Commands

```bash
make install
make install-dev
make install-bench
make install-vendor
make test
make smoke
make bench-blank
make bench-run
make bench-clean
make lint
```
