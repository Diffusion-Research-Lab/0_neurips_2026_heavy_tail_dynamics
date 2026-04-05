PYTHON ?= python
BASH ?= bash
VENV_DIR ?= .venv
PYTHONPATH_EXPORT = export PYTHONPATH="$(CURDIR)/src$${PYTHONPATH:+:$$PYTHONPATH}"

.DEFAULT_GOAL := help

.PHONY: setup-local setup-server run-local run-server check send fetch supp help

setup-local:
	$(BASH) benchmarks/01_setup.sh --venv-dir "$(VENV_DIR)"
	PYTHON="$(CURDIR)/$(VENV_DIR)/bin/python" $(BASH) scripts/fetch_vendor.sh

setup-server:
	$(BASH) benchmarks/01_setup.sh --venv-dir "$(VENV_DIR)" --use-jz-module
	PYTHON="$(CURDIR)/$(VENV_DIR)/bin/python" $(BASH) scripts/fetch_vendor.sh
	$(BASH) -lc 'source "$(VENV_DIR)/bin/activate"; $(PYTHONPATH_EXPORT); $(PYTHON) -c "from genkit.datasets import list_datasets, fetch_real_data; [fetch_real_data(name, val_size=0.15, test_size=0.15) for name in list_datasets()]"'

run-local:
	$(BASH) benchmarks/03_launcher.sh --run --venv-dir "$(CURDIR)/$(VENV_DIR)"

run-server:
	sbatch --export=ALL,VENV_DIR="$(CURDIR)/$(VENV_DIR)" benchmarks/02_launcher.slurm

check:
	$(BASH) -lc 'source "$(VENV_DIR)/bin/activate"; $(PYTHONPATH_EXPORT); flake8 --ignore E501 --exclude src/genkit/_vendor src tests benchmarks examples'
	$(BASH) -lc 'source "$(VENV_DIR)/bin/activate"; $(PYTHONPATH_EXPORT); pytest -v'
	$(BASH) -lc 'source "$(VENV_DIR)/bin/activate"; $(PYTHONPATH_EXPORT); python benchmarks/_smoke_test.py'
	$(BASH) -lc 'source "$(VENV_DIR)/bin/activate"; $(PYTHONPATH_EXPORT); python examples/01_visu_1d_path.py --blank'
	$(BASH) -lc 'source "$(VENV_DIR)/bin/activate"; $(PYTHONPATH_EXPORT); python examples/02_visu_2d.py --blank'
	$(BASH) -lc 'source "$(VENV_DIR)/bin/activate"; $(PYTHONPATH_EXPORT); python examples/03_shariatan_et_al.py --blank'
	$(BASH) -lc 'source "$(VENV_DIR)/bin/activate"; $(PYTHONPATH_EXPORT); bash benchmarks/03_launcher.sh --blank'

send:
	$(BASH) scripts/transfer.sh send

fetch:
	$(BASH) scripts/transfer.sh fetch

supp:
	$(BASH) scripts/transfer.sh supp

help:
	@printf "Available targets:\n"
	@printf "  %-14s %s\n" "setup-local" "Fetch vendors and install the local benchmark environment"
	@printf "  %-14s %s\n" "setup-server" "Fetch vendors, install the Jean Zay environment, and prefetch real datasets"
	@printf "  %-14s %s\n" "run-local" "Run benchmarks locally via benchmarks/03_launcher.sh --run"
	@printf "  %-14s %s\n" "run-server" "Submit the Slurm benchmark job via benchmarks/02_launcher.slurm"
	@printf "  %-14s %s\n" "check" "Run lint, tests, blank examples, smoke, and blank benchmarks"
	@printf "  %-14s %s\n" "send" "Send the project tree to the remote benchmark host"
	@printf "  %-14s %s\n" "fetch" "Fetch benchmark figures and tables from the remote host"
	@printf "  %-14s %s\n" "supp" "Build the supplementary code archive"
	@printf "  %-14s %s\n" "help" "Print this help message"
