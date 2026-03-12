PYTHON ?= python
PIP ?= pip

.PHONY: install test smoke

install:
	$(PIP) install -e .[dev]
	$(PIP) install -r benchmarks/requirements.txt

test:
	pytest -q

smoke:
	$(PYTHON) benchmarks/smoke_test.py
