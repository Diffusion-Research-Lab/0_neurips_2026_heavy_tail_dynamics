BASH            ?= bash
VENV_DIR        ?= .venv
JZ_MODULE       ?= pytorch-gpu/py3/2.8.0
LOG_DIR         ?= logs
LOG_PREFIX      ?= bench
ANALYSIS_DATA_DIR ?= benchmarks/data
ALPHASTABLE_PILOT_CONFIG ?= benchmarks/configs/01_alphastable_pilot.yaml
ALPHASTABLE_BENCH_CONFIG ?= benchmarks/configs/02_alphastable_bench.yaml
IMAGE_PILOT_CONFIG ?= benchmarks/configs/03_image_pilot.yaml
IMAGE_BENCH_CONFIG ?= benchmarks/configs/04_image_bench.yaml
ALPHA_PILOT_ARRAY ?= 0-269
ALPHA_BENCH_ARRAY ?= 0-99
IMAGE_PILOT_ARRAY ?= 0-39
IMAGE_BENCH_ARRAY ?= 0-9
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

.PHONY: setup dataset prefetch-data run pilot bench evaluate check send fetch supp help

setup:
	$(BASH) scripts/setup.sh --venv-dir "$(VENV_DIR)" --use-jz-module
	PYTHON="$(VENV_PYTHON)" $(BASH) scripts/fetch.vendor.sh
	$(MAKE) dataset

dataset:
	$(BASH) -lc 'type module >/dev/null 2>&1 && { module purge || true; module load "$(JZ_MODULE)"; }; $(ACTIVATE); python scripts/prefetch_datasets.py'

run: pilot bench

pilot:
	@mkdir -p "$(LOG_DIR)"
	sbatch --array=$(ALPHA_PILOT_ARRAY) $(RUN_LOG_ARGS) $(SBATCH_EXPORT) scripts/run.slurm.sh --config "$(ALPHASTABLE_PILOT_CONFIG)" --skip-existing
	sbatch --array=$(IMAGE_PILOT_ARRAY) $(RUN_LOG_ARGS) $(SBATCH_EXPORT) scripts/run.slurm.sh --config "$(IMAGE_PILOT_CONFIG)" --skip-existing

bench:
	@mkdir -p "$(LOG_DIR)"
	sbatch --array=$(ALPHA_BENCH_ARRAY) $(RUN_LOG_ARGS) $(SBATCH_EXPORT) scripts/run.slurm.sh --config "$(ALPHASTABLE_BENCH_CONFIG)" --skip-existing
	sbatch --array=$(IMAGE_BENCH_ARRAY) $(RUN_LOG_ARGS) $(SBATCH_EXPORT) scripts/run.slurm.sh --config "$(IMAGE_BENCH_CONFIG)" --skip-existing

evaluate:
	@mkdir -p "$(LOG_DIR)"
	@test -d "$(ANALYSIS_DATA_DIR)" || { echo "Missing: $(ANALYSIS_DATA_DIR)" >&2; exit 2; }
	sbatch --array=$(ALPHA_PILOT_ARRAY) $(EVAL_LOG_ARGS) $(SBATCH_EXPORT) scripts/evaluate.slurm.sh \
	  --batch-dir "$$(ls -d "$(ANALYSIS_DATA_DIR)"/*_01_alphastable_pilot | tail -n 1)"
	sbatch --array=$(ALPHA_BENCH_ARRAY) $(EVAL_LOG_ARGS) $(SBATCH_EXPORT) scripts/evaluate.slurm.sh \
	  --batch-dir "$$(ls -d "$(ANALYSIS_DATA_DIR)"/*_02_alphastable_bench | tail -n 1)"
	sbatch --array=$(IMAGE_PILOT_ARRAY) $(EVAL_LOG_ARGS) $(SBATCH_EXPORT) scripts/evaluate.slurm.sh \
	  --batch-dir "$$(ls -d "$(ANALYSIS_DATA_DIR)"/*_03_image_pilot | tail -n 1)"
	sbatch --array=$(IMAGE_BENCH_ARRAY) $(EVAL_LOG_ARGS) $(SBATCH_EXPORT) scripts/evaluate.slurm.sh \
	  --batch-dir "$$(ls -d "$(ANALYSIS_DATA_DIR)"/*_04_image_bench | tail -n 1)"

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
	@printf "  %-14s %s\n" "dataset"    "Prefetch cacheable real datasets"
	@printf "  %-14s %s\n" "run"        "Submit all benchmark configs via Slurm"
	@printf "  %-14s %s\n" "pilot"      "Submit pilot configs via Slurm"
	@printf "  %-14s %s\n" "bench"      "Submit benchmark configs via Slurm"
	@printf "  %-14s %s\n" "evaluate"   "Submit sharded evaluation for latest benchmark batches"
	@printf "  %-14s %s\n" "check"         "Run lint, tests, smoke, and blank benchmarks"
	@printf "  %-14s %s\n" "send"          "Send the project tree to the remote benchmark host"
	@printf "  %-14s %s\n" "fetch"         "Fetch benchmark result directories into benchmarks/data"
	@printf "  %-14s %s\n" "supp"          "Build the supplementary code archive"
	@printf "  %-14s %s\n" "help"          "Print this help message"
