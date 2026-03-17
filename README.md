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
- benchmark scripts for synthetic and long-tail settings.

### Install

```bash
pip install -e .[dev]
pip install -r benchmarks/requirements.txt
```

Or:

```bash
make install
```

### Quickstart

Check imports:

```bash
python -c "import genkit, labkit; print(genkit.__name__, labkit.__name__)"
```

Run smoke test:

```bash
python benchmarks/smoke_test.py
```

Run tiny example:

```bash
python examples/spiral_example.py --smoke
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
bash benchmarks/03_launcher.sh --blank
```

Full local run:

```bash
bash benchmarks/03_launcher.sh --run
```

Slurm:

```bash
sbatch benchmarks/02_launcher.slurm
```

### Developer Commands

```bash
make install
make test
make smoke
flake8 --ignore E501 -j1
```
