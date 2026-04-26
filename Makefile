BASH            ?= bash
VENV_DIR        ?= .venv
JZ_MODULE       ?= pytorch-gpu/py3/2.8.0
LOG_DIR         ?= logs
LOG_PREFIX      ?= bench
ANALYSIS_DATA_DIR ?= benchmarks/data
ALPHASTABLE_CONFIG ?= benchmarks/configs/01_alphastable_baseline.yaml
HRRR_CONFIG     ?= benchmarks/configs/02_hrrr_unet.yaml
ALPHA_ARRAY     ?= 0-49
HRRR_ARRAY      ?= 0-4
VENV_PYTHON      = $(CURDIR)/$(VENV_DIR)/bin/python
TOOLS_BIN        = $(CURDIR)/.tools/bin
PATH_EXPORT      = export PATH="$(TOOLS_BIN):$$PATH"
PYTHONPATH_EXPORT = export PYTHONPATH="$(CURDIR)/src$${PYTHONPATH:+:$$PYTHONPATH}"
ACTIVATE         = source "$(VENV_DIR)/bin/activate"; $(PATH_EXPORT); $(PYTHONPATH_EXPORT)
RUN_LOG_ARGS     = --output="$(CURDIR)/$(LOG_DIR)/$(LOG_PREFIX)_%A_%a.out" \
                   --error="$(CURDIR)/$(LOG_DIR)/$(LOG_PREFIX)_%A_%a.err"
EVAL_LOG_ARGS    = --output="$(CURDIR)/$(LOG_DIR)/$(LOG_PREFIX)_eval_%A_%a.out" \
                   --error="$(CURDIR)/$(LOG_DIR)/$(LOG_PREFIX)_eval_%A_%a.err"
SBATCH_EXPORT    = --export=ALL,VENV_DIR="$(CURDIR)/$(VENV_DIR)"

.DEFAULT_GOAL := help

.PHONY: setup prefetch-data run evaluate check send fetch supp help

setup:
	$(BASH) scripts/setup.sh --venv-dir "$(VENV_DIR)" --use-jz-module
	PYTHON="$(VENV_PYTHON)" $(BASH) scripts/fetch.vendor.sh
	$(BASH) -lc 'type module >/dev/null 2>&1 && { module purge || true; module load "$(JZ_MODULE)"; }; $(ACTIVATE); python scripts/prefetch_datasets.py'

run:
	@mkdir -p "$(LOG_DIR)"
	sbatch --array=$(ALPHA_ARRAY) $(RUN_LOG_ARGS) $(SBATCH_EXPORT) scripts/run.slurm.sh --config "$(ALPHASTABLE_CONFIG)"
	sbatch --array=$(HRRR_ARRAY) $(RUN_LOG_ARGS) $(SBATCH_EXPORT) scripts/run.slurm.sh --config "$(HRRR_CONFIG)" --skip-existing

evaluate:
	@mkdir -p "$(LOG_DIR)"
	@test -d "$(ANALYSIS_DATA_DIR)" || { echo "Missing: $(ANALYSIS_DATA_DIR)" >&2; exit 2; }
	sbatch --array=$(ALPHA_ARRAY) $(EVAL_LOG_ARGS) $(SBATCH_EXPORT) scripts/evaluate.slurm.sh \
	  --batch-dir "$$(ls -d "$(ANALYSIS_DATA_DIR)"/*_01_alphastable_baseline | tail -n 1)"
	sbatch --array=$(HRRR_ARRAY) $(EVAL_LOG_ARGS) $(SBATCH_EXPORT) scripts/evaluate.slurm.sh \
	  --batch-dir "$$(ls -d "$(ANALYSIS_DATA_DIR)"/*_02_hrrr_unet | tail -n 1)"

check:
	$(BASH) -lc '$(ACTIVATE); flake8 --ignore E501 --exclude src/genkit/_vendor src tests benchmarks examples'
	$(BASH) -lc '$(ACTIVATE); pytest -v'
	$(BASH) -lc '$(ACTIVATE); python -m benchmarks.main --config benchmarks/configs/00_blank.yaml'
	$(BASH) -lc '$(ACTIVATE); python examples/01_visu_1d_path.py --blank'
	$(BASH) -lc '$(ACTIVATE); python examples/02_visu_2d.py --blank'

send:
	$(BASH) scripts/transfer.sh send

fetch:
	$(BASH) scripts/transfer.sh fetch

supp:
	$(BASH) scripts/transfer.sh supp

help:
	@printf "Available targets:\n"
	@printf "  %-14s %s\n" "setup"      "Install the Jean Zay environment and prefetch real datasets"
	@printf "  %-14s %s\n" "run"        "Submit all benchmark configs via Slurm"
	@printf "  %-14s %s\n" "evaluate"   "Submit sharded evaluation for latest alpha-stable and HRRR batches"
	@printf "  %-14s %s\n" "check"         "Run lint, tests, smoke, and blank benchmarks"
	@printf "  %-14s %s\n" "send"          "Send the project tree to the remote benchmark host"
	@printf "  %-14s %s\n" "fetch"         "Fetch benchmark result directories into benchmarks/data"
	@printf "  %-14s %s\n" "supp"          "Build the supplementary code archive"
	@printf "  %-14s %s\n" "help"          "Print this help message"
