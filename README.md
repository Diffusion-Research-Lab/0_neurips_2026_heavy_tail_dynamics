## FlowBench

[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/hcherkaoui/flowbench/actions/workflows/ci.yml/badge.svg)](https://github.com/hcherkaoui/flowbench/actions/workflows/ci.yml)
[![flake8](https://github.com/hcherkaoui/flowbench/actions/workflows/flake8.yml/badge.svg)](https://github.com/hcherkaoui/flowbench/actions/workflows/flake8.yml)
[![codecov](https://codecov.io/gh/hcherkaoui/flowbench/graph/badge.svg?branch=master&token=YOUR_PRIVATE_CODECOV_BADGE_TOKEN)](https://codecov.io/gh/hcherkaoui/flowbench)
[![Smoke](https://github.com/hcherkaoui/flowbench/actions/workflows/smoke.yml/badge.svg)](https://github.com/hcherkaoui/flowbench/actions/workflows/smoke.yml)
![maintenance-status](https://img.shields.io/badge/maintenance-active-brightgreen.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


```bash
███████ ██       ██████  ██     ██       ██████  ███████ ███    ██  ██████ ██   ██
██      ██      ██    ██ ██     ██       ██   ██ ██      ████   ██ ██      ██   ██
█████   ██      ██    ██ ██  █  ██ █████ ██████  █████   ██ ██  ██ ██      ███████
██      ██      ██    ██ ██ ███ ██       ██   ██ ██      ██  ██ ██ ██      ██   ██
██      ███████  ██████   ███ ███        ██████  ███████ ██   ████  ██████ ██   ██
```


FlowBench provides:
- heavy-tailed samplers and generative baselines for flow-matching and diffusion models,
- lightweight training, inspection, and evaluation utilities to compare learned generative dynamics,
- benchmark loaders and sweep tooling for reproducible experiments on synthetic and real datasets.

For a private Codecov badge, replace `YOUR_PRIVATE_CODECOV_BADGE_TOKEN` with the private badge token from the Codecov repo settings.

### Install

```bash
make setup-local
```

### Quickstart

Run the local checks:

```bash
make check
```

Run benchmarks locally:

```bash
make run-local
```

Submit the benchmark on Jean Zay after a setup:

```bash
make setup-jz && make run-jz
```

### Minimal Example

```python
from genkit.datasets import fetch_synthetic_data
from genkit.flow import GaussianFlowLinear
from genkit.nn import MLPModel
from genkit.training import train

x_train, _, _ = fetch_synthetic_data(
    target_data="balanced_bimodal_gaussian",
    dim=10,
    n_samples=20_000,
)

net = MLPModel(dim=10, width=32, depth=3, time_dim=16)
generator = GaussianFlowLinear(net=net, dim=10, n_steps=128)

generator, stats = train(
    generator,
    target_data=x_train,
    n_epochs=128,
    batch_size=1024,
    lr=1e-3,
    device="cpu",
    num_workers=0,
)

x_gen = generator.sample(n_samples=2048)
print(x_gen.shape, list(stats))
```
