SHELL := /bin/bash

ROOT := $(abspath ..)
PYTHON ?= $(ROOT)/.venv/bin/python
JZ_MODULE ?= pytorch-gpu/py3/2.8.0
SBATCH_ARGS ?= --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --partition=gpu_p13 --qos=qos_gpu-t3 --account=jcx@v100 --time=08:00:00
LOG_DIR ?= logs
SCRIPTS ?= 0_heteroscedastic.py 1_dlpm_loss_power.py 2_dlpm_mc_score.py 3_lora_tail_correction_model.py 4_contracted_space_gfl.py 5_bulk_tail_matched_dlpm_source.py 6_tail_conditioned_gfl.py 7_scheduled_kappa_dlpm.py 8_feynman_kac_guidance.py

.PHONY: run run-local

run:
	@mkdir -p "$(LOG_DIR)"
	sbatch --job-name=flowbench_prototypes --output="$(LOG_DIR)/%x_%j.out" --error="$(LOG_DIR)/%x_%j.err" $(SBATCH_ARGS) --wrap='cd "$(CURDIR)" && make run-local SCRIPTS="$(SCRIPTS)"'

run-local:
	@set -euo pipefail; \
	module purge || true; \
	conda deactivate 2>/dev/null || true; \
	module load "$(JZ_MODULE)"; \
	for script in $(SCRIPTS); do \
	  echo "[prototypes] $$script"; \
	  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$(ROOT):$(ROOT)/src$${PYTHONPATH:+:$$PYTHONPATH}" "$(PYTHON)" "$$script"; \
	done
