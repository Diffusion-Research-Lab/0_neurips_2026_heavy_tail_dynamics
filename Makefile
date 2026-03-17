PYTHON ?= python
PIP ?= pip
PYTEST ?= pytest
FLAKE8 ?= flake8

.PHONY: install dev_checks bench_checks checks full_checks

install:
	$(PIP) install -e .[dev]
	$(PIP) install -r benchmarks/requirements.txt

dev_checks:
	$(PYTEST)
	$(FLAKE8) --ignore E501

bench_checks:
	cd examples && $(PYTHON) spiral.py --blank && $(PYTHON) bimodal_gaussian.py --blank
	cd benchmarks && bash 03_launcher.sh --blank

checks:
	$(MAKE) dev_checks
	cd examples && $(PYTHON) spiral.py --blank && $(PYTHON) bimodal_gaussian.py --blank
	cd benchmarks && bash 03_launcher.sh --blank

full_checks: checks
