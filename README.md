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
- benchmark loaders and sweep tooling for reproducible experiments on synthetic and image datasets.

### Benchmark

To reproduce the NeurIPS 2026 paper results, see `benchmarks/README.md`.

### Minimal Example

```python
from genkit.datasets import fetch_synthetic_data
from genkit.flow_matching import GaussianFlowLinear
from genkit.nn import MLPModel
from genkit.training import train

dim = 2
x_train, _, _ = fetch_synthetic_data(
    target_data="balanced_bimodal_gaussian",
    dim=dim,
    n_samples=10_000,
)

net = MLPModel(dim=dim, width=64, depth=4, time_dim=32)
generator = GaussianFlowLinear(net=net, dim=dim, n_steps=128)

generator, _stats_ = train(
    generator,
    target_data=x_train,
    n_epochs=128,
    batch_size=1024,
    lr=1e-3,
    device="cpu",
    num_workers=0,
)

x_gen = generator.sample(n_samples=2048)
print(x_gen.shape)
```

### Citation

If you use this code in your research, please cite:

```bibtex
@article{cherkaoui2026heavy,
  title={Do Heavy Tails Help Diffusion? On the Subtle Trade-off Between Initialization and Training},
  author={Cherkaoui, Hamza and Halconruy, H{\'e}l{\`e}ne and Ocello, Antonio},
  journal={arXiv preprint arXiv:2605.13175},
  year={2026}
}
```
