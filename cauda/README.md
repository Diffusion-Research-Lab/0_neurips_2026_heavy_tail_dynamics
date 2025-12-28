## Cauda

**Cauda** is a small PyTorch-first package for generative modeling with diffusion and flow methods tailored to heavy-tailed data.

### Installation

To install the required dependencies and the package, run the command::

    pip install -r requirements.txt
    python setup.py install


To test the installation you can run the unit-tests, run the command::

    pytest  # run the unit-tests


In order to check the PEP 8 compliance level of the package, run the command::

    flake8 --ignore=E501 --count cauda


### Usage

Examples are located in the `examples` directory. Here is how you can run a benchmark for Gaussian Processes::

    python examples/1_benchmark_gp.py


### License

This project is licensed under BSD 3-Clause License. See the [LICENSE](./LICENCE) file for details.
