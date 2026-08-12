## Heavy Tail Dynamics

[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
![maintenance-status](https://img.shields.io/badge/maintenance-active-brightgreen.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Heavy Tail Dynamics provides:
- heavy-tailed samplers and generative baselines for flow-matching and diffusion models,
- lightweight training, inspection, and evaluation utilities to compare learned generative dynamics,
- benchmark loaders and sweep tooling for reproducible experiments on synthetic and image datasets.

### Benchmark

To reproduce the NeurIPS 2026 paper results, see `benchmarks/README.md`.

On Jean Zay, benchmark-specific outputs are stored under `$WORK/0_neurips_2026_heavy_tail_dynamics_assets`; shared datasets remain under `$WORK/jz_datasets`.

Install the independent benchmark utility, model, and Jean Zay dataset packages from their public GitHub repositories over HTTPS before installing this benchmark:

```bash
pip install "benchtools @ git+https://github.com/Diffusion-Research-Lab/benchtools.git"
pip install "gendynamics @ git+https://github.com/Diffusion-Research-Lab/gendynamics.git"
pip install "jeanzaydata @ git+https://github.com/Diffusion-Research-Lab/jeanzaydata.git"
pip install -e ".[dev,bench]"
```

### Minimal Example

```python
from gendynamics.datasets import fetch_synthetic_data
from gendynamics.flow_matching import GaussianFlowLinear
from gendynamics.nn import MLPModel
from gendynamics.training import train

dim = 2
x_train, _, _ = fetch_synthetic_data(
    target_data="checker",
    n_samples=10_000,
)

net = MLPModel(input_dim=dim, width=64, depth=4, time_dim=32)
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
