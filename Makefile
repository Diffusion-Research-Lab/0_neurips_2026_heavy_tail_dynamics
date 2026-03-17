PYTHON ?= python
PIP ?= pip
PYTEST ?= pytest
FLAKE8 ?= flake8
JUPYTER ?= jupyter

NOTEBOOKS := $(wildcard sandbox/*.ipynb)

.PHONY: install dev_checks bench_checks notebook_checks checks full_checks

install:
	$(PIP) install -e .[dev]
	$(PIP) install -r benchmarks/requirements.txt

dev_checks:
	$(PYTEST)
	$(FLAKE8) --ignore E501

bench_checks:
	cd examples && $(PYTHON) spiral.py --blank && $(PYTHON) bimodal_gaussian.py --blank
	cd benchmarks && bash 03_launcher.sh --blank

notebook_checks:
	@if [ -z "$(NOTEBOOKS)" ]; then \
		echo "No notebooks found under sandbox/"; \
	else \
		for nb in $(NOTEBOOKS); do \
			$(JUPYTER) nbconvert --to notebook --execute --inplace "$$nb"; \
		done; \
	fi

checks:
	$(PYTEST)
	$(FLAKE8) --ignore E501
	cd examples && $(PYTHON) spiral.py --blank && $(PYTHON) bimodal_gaussian.py --blank
	cd sandbox && for nb in *.ipynb; do $(JUPYTER) nbconvert --to notebook --execute --inplace "$$nb"; done
	cd benchmarks && bash 03_launcher.sh --blank

full_checks: checks
