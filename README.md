# Heavy Tail Dynamics

Diffusion and flow-matching experiments on heavy-tailed distributions. For paper reproduction and Jean Zay commands, see [benchmarks/README.md](benchmarks/README.md).

Install the shared packages, then this project:

```bash
pip install "benchtools @ git+https://github.com/Diffusion-Research-Lab/benchtools.git" \
  "gendynamics @ git+https://github.com/Diffusion-Research-Lab/gendynamics.git" \
  "jeanzaydata @ git+https://github.com/Diffusion-Research-Lab/jeanzaydata.git"
pip install -e ".[dev,bench]"
```

Citation:

```bibtex
@article{cherkaoui2026heavy,
  title={Do Heavy Tails Help Diffusion? On the Subtle Trade-off Between Initialization and Training},
  author={Cherkaoui, Hamza and Halconruy, H{\'e}l{\`e}ne and Ocello, Antonio},
  journal={arXiv preprint arXiv:2605.13175},
  year={2026}
}
```
