## Cauda

**Cauda** is a small PyTorch-first package for generative modeling with diffusion and flow methods tailored to heavy-tailed data.

### Installation

Recommended pip virtual environment setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

To test the installation you can run the unit tests:

```bash
pytest
```

In order to check the PEP 8 compliance level of the package:

```bash
flake8 --ignore=E501 --count cauda
```

### Usage

Examples are located in the `examples` directory. Run:

```bash
python examples/spiral_example.py
```

### License

This project is licensed under BSD 3-Clause License. See the [LICENSE](./LICENCE) file for details.
