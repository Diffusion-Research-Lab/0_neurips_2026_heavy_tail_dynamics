PYTHON ?= python
BASH ?= bash
VENV_DIR ?= .venv
JZ_MODULE ?= pytorch-gpu/py3/2.8.0
ARRAY_BLANK ?= 0-0
ARRAY_ALPHASTABLE ?= 0-7
ARRAY_DIMENSION ?= 0-15
ARRAY_MODECOLLAPSE ?= 0-5
VENV_PYTHON = $(CURDIR)/$(VENV_DIR)/bin/python
PYTHONPATH_EXPORT = export PYTHONPATH="$(CURDIR)/src$${PYTHONPATH:+:$$PYTHONPATH}"

.DEFAULT_GOAL := help

.PHONY: setup-local setup-jz run-local run-jz check send fetch supp help

setup-local:
	$(BASH) benchmarks/launchers/setup.sh --venv-dir "$(VENV_DIR)"
	PYTHON="$(VENV_PYTHON)" $(BASH) scripts/fetch_vendor.sh

setup-jz:
	$(BASH) benchmarks/launchers/setup.sh --venv-dir "$(VENV_DIR)" --use-jz-module
	PYTHON="$(VENV_PYTHON)" $(BASH) scripts/fetch_vendor.sh
	$(BASH) -lc 'module purge || true; module load "$(JZ_MODULE)"; source "$(VENV_DIR)/bin/activate"; $(PYTHONPATH_EXPORT); "$(VENV_PYTHON)" -c "from genkit.datasets import fetch_real_data, get_dataset_metadata, list_datasets; [fetch_real_data(name, val_size=0.15, test_size=0.15) for name in list_datasets() if get_dataset_metadata(name)[\"dataset_type\"] == \"real\"]"'

run-local:
	$(BASH) benchmarks/launchers/local.sh --run --venv-dir "$(CURDIR)/$(VENV_DIR)"

run-jz:
	@set -e; \
	for cfg in benchmarks/configs/*.yaml; do \
	  case "$$(basename "$$cfg")" in \
	    00_blank.yaml) array="$(ARRAY_BLANK)" ;; \
	    01_alphastable_baseline.yaml) array="$(ARRAY_ALPHASTABLE)" ;; \
	    02_dimension_effect.yaml) array="$(ARRAY_DIMENSION)" ;; \
	    03_modecollapsing_effect.yaml) array="$(ARRAY_MODECOLLAPSE)" ;; \
	    *) array="$(ARRAY_ALPHASTABLE)" ;; \
	  esac; \
	  echo "Submitting $$cfg with --array=$$array"; \
	  sbatch --array="$$array" --export=ALL,VENV_DIR="$(CURDIR)/$(VENV_DIR)" benchmarks/launchers/slurm.sh --config "$$cfg"; \
	done

check:
	$(BASH) -lc 'source "$(VENV_DIR)/bin/activate"; $(PYTHONPATH_EXPORT); flake8 --ignore E501 --exclude src/genkit/_vendor src tests benchmarks examples'
	$(BASH) -lc 'source "$(VENV_DIR)/bin/activate"; $(PYTHONPATH_EXPORT); pytest -v'
	$(BASH) -lc 'source "$(VENV_DIR)/bin/activate"; $(PYTHONPATH_EXPORT); python -m benchmarks.runner.main --config benchmarks/configs/00_blank.yaml'
	$(BASH) -lc 'source "$(VENV_DIR)/bin/activate"; $(PYTHONPATH_EXPORT); python examples/01_visu_1d_path.py --blank'
	$(BASH) -lc 'source "$(VENV_DIR)/bin/activate"; $(PYTHONPATH_EXPORT); python examples/02_visu_2d.py --blank'
	$(BASH) -lc 'source "$(VENV_DIR)/bin/activate"; $(PYTHONPATH_EXPORT); python examples/03_shariatan_et_al.py --blank'
	$(BASH) -lc 'source "$(VENV_DIR)/bin/activate"; $(PYTHONPATH_EXPORT); rm -rf _figures/ _00_results/ && python -c "from pathlib import Path; import shutil; [shutil.rmtree(path) for path in Path(\".\").rglob(\"__pycache__\") if path.is_dir()]; [path.unlink() for pattern in (\"*.pyc\", \"*.pyo\") for path in Path(\".\").rglob(pattern) if path.is_file()]"'

send:
	$(BASH) scripts/transfer.sh send

fetch:
	$(BASH) scripts/transfer.sh fetch

supp:
	$(BASH) scripts/transfer.sh supp

help:
	@printf "Available targets:\n"
	@printf "  %-14s %s\n" "setup-local" "Fetch vendors and install the local benchmark environment"
	@printf "  %-14s %s\n" "setup-jz" "Fetch vendors, install the Jean Zay environment, and prefetch real datasets"
	@printf "  %-14s %s\n" "run-local" "Run benchmarks locally via benchmarks/launchers/local.sh --run"
	@printf "  %-14s %s\n" "run-jz" "Submit all benchmark configs via Slurm with per-config array sizes"
	@printf "  %-14s %s\n" "check" "Run lint, tests, blank examples, smoke, and blank benchmarks"
	@printf "  %-14s %s\n" "send" "Send the project tree to the remote benchmark host"
	@printf "  %-14s %s\n" "fetch" "Fetch benchmark result directories into benchmarks/analysis/data"
	@printf "  %-14s %s\n" "supp" "Build the supplementary code archive"
	@printf "  %-14s %s\n" "help" "Print this help message"
