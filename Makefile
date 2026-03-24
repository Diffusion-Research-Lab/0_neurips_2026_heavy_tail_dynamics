PYTHON ?= python
PIP ?= pip
PYTEST ?= pytest
FLAKE8 ?= flake8
BASH ?= bash

.PHONY: install install-dev install-bench install-vendor bench-setup transfer-send transfer-fetch transfer-supp test lint smoke bench-blank bench-run bench-clean checks full_checks

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e .[dev]

install-bench:
	$(PIP) install -e .[dev,bench]

install-vendor:
	$(BASH) scripts/fetch_vendor.sh

bench-setup:
	$(BASH) benchmarks/01_setup.sh

transfer-send:
	$(BASH) scripts/transfer.sh send

transfer-fetch:
	$(BASH) scripts/transfer.sh fetch

transfer-supp:
	$(BASH) scripts/transfer.sh supp

test:
	$(PYTEST) -q

lint:
	$(FLAKE8) --ignore E501 --exclude src/genkit/_vendor

smoke:
	$(PYTHON) benchmarks/_smoke_test.py
	$(PYTHON) examples/01_visu_1d_path.py --blank
	$(PYTHON) examples/02_visu_2d.py --blank
	$(PYTHON) examples/03_shariatan_et_al.py

bench-blank:
	$(BASH) benchmarks/03_launcher.sh --blank

bench-run:
	$(BASH) benchmarks/03_launcher.sh --run

bench-clean:
	$(BASH) benchmarks/03_launcher.sh --clean

checks: test lint smoke bench-blank

full_checks: checks
