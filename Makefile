BASH            ?= bash
VENV_DIR        ?= .venv
JZ_MODULE       ?= pytorch-gpu/py3/2.8.0
LOG_DIR         ?= logs
LOG_PREFIX      ?= htfm
ANALYSIS_DATA_DIR ?= benchmarks/data
JOBID           ?=
TASK            ?=
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

.PHONY: setup-local setup-jz prefetch-data run-local run-jz evaluate-jz inspect-jz log-jz check send fetch supp help

setup-local:
	$(BASH) scripts/setup.sh --venv-dir "$(VENV_DIR)"
	PYTHON="$(VENV_PYTHON)" $(BASH) scripts/fetch.vendor.sh

setup-jz:
	$(BASH) scripts/setup.sh --venv-dir "$(VENV_DIR)" --use-jz-module
	PYTHON="$(VENV_PYTHON)" $(BASH) scripts/fetch.vendor.sh
	$(MAKE) prefetch-data

prefetch-data:
	$(BASH) -lc 'type module >/dev/null 2>&1 && { module purge || true; module load "$(JZ_MODULE)"; }; $(ACTIVATE); python scripts/prefetch_datasets.py'

run-local:
	$(BASH) scripts/run.local.sh --run --venv-dir "$(CURDIR)/$(VENV_DIR)"

run-jz:
	@mkdir -p "$(LOG_DIR)"
	sbatch --array=0-0  $(RUN_LOG_ARGS) $(SBATCH_EXPORT) scripts/run.slurm.sh --config benchmarks/configs/00_blank.yaml
	sbatch --array=0-15 $(RUN_LOG_ARGS) $(SBATCH_EXPORT) scripts/run.slurm.sh --config benchmarks/configs/01_alphastable_baseline.yaml

evaluate-jz:
	@mkdir -p "$(LOG_DIR)"
	@test -d "$(ANALYSIS_DATA_DIR)" || { echo "Missing: $(ANALYSIS_DATA_DIR)" >&2; exit 2; }
	sbatch --array=0-15 $(EVAL_LOG_ARGS) $(SBATCH_EXPORT) scripts/evaluate.slurm.sh \
	  --batch-dir "$$(ls -d "$(ANALYSIS_DATA_DIR)"/*_01_alphastable_baseline | tail -n 1)"

inspect-jz:
	@test -n "$(JOBID)" || { echo "Usage: make inspect-jz JOBID=<id>" >&2; exit 2; }
	sacct -j "$(JOBID)" --format=JobID,JobName%20,State,ExitCode,Elapsed,NodeList%30

log-jz:
	@test -n "$(JOBID)" || { echo "Usage: make log-jz JOBID=<id> [TASK=<index>]" >&2; exit 2; }
	@if [ -n "$(TASK)" ]; then \
	  for f in "$(LOG_DIR)/$(LOG_PREFIX)_$(JOBID)_$(TASK).out" \
	           "$(LOG_DIR)/$(LOG_PREFIX)_$(JOBID)_$(TASK).err"; do \
	    [ -f "$$f" ] && { echo "==> $$f <=="; tail -n 80 "$$f"; } || true; \
	  done; \
	else \
	  ls -1 "$(LOG_DIR)/$(LOG_PREFIX)_$(JOBID)"_*.{out,err} 2>/dev/null || true; \
	fi

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
	@printf "  %-14s %s\n" "setup-local"   "Install the local benchmark environment"
	@printf "  %-14s %s\n" "setup-jz"      "Install the Jean Zay environment and prefetch real datasets"
	@printf "  %-14s %s\n" "prefetch-data" "Download and cache all real datasets under \$$WORK or \$$HOME"
	@printf "  %-14s %s\n" "run-local"     "Run benchmarks locally"
	@printf "  %-14s %s\n" "run-jz"        "Submit all benchmark configs via Slurm"
	@printf "  %-14s %s\n" "evaluate-jz"   "Submit sharded evaluation for the latest alphastable batch"
	@printf "  %-14s %s\n" "inspect-jz"    "Inspect a Jean Zay array job via sacct (JOBID=...)"
	@printf "  %-14s %s\n" "log-jz"        "List or tail Jean Zay logs (JOBID=..., TASK=...)"
	@printf "  %-14s %s\n" "check"         "Run lint, tests, smoke, and blank benchmarks"
	@printf "  %-14s %s\n" "send"          "Send the project tree to the remote benchmark host"
	@printf "  %-14s %s\n" "fetch"         "Fetch benchmark result directories into benchmarks/data"
	@printf "  %-14s %s\n" "supp"          "Build the supplementary code archive"
	@printf "  %-14s %s\n" "help"          "Print this help message"
