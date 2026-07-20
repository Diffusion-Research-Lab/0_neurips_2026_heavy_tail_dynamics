SHELL := /bin/bash

ROOT := $(abspath ..)
PYTHON ?= $(ROOT)/.venv/bin/python
JZ_MODULE ?= pytorch-gpu/py3/2.8.0
SBATCH_ARGS ?= --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --partition=gpu_p13 --qos=qos_gpu-t3 --account=jcx@v100 --time=15:00:00
LOG_DIR ?= logs
SCRIPTS ?= 0_heteroscedastic.py 1_dlpm_loss_power.py 2_training_size_tce.py 3_contracted_space_gfl.py 4_tail_conditioned_gfl.py 5_scheduled_kappa_dlpm.py 6_feynman_kac_guidance.py 7_tce_regularized_ddpm.py

.PHONY: run run-local

run:
	@mkdir -p "$(LOG_DIR)"
	sbatch --job-name=flowbench_prototypes --output="$(LOG_DIR)/%x_%j.out" --error="$(LOG_DIR)/%x_%j.err" $(SBATCH_ARGS) --wrap='cd "$(CURDIR)" && make run-local SCRIPTS="$(SCRIPTS)"'

run-local:
	@set -uo pipefail; \
	module purge || true; \
	conda deactivate 2>/dev/null || true; \
	module load "$(JZ_MODULE)" || exit "$$?"; \
	status=0; \
	failed_scripts=""; \
	for script in $(SCRIPTS); do \
	  echo "[prototypes] $$script"; \
	  if PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$(ROOT):$(ROOT)/src$${PYTHONPATH:+:$$PYTHONPATH}" "$(PYTHON)" "$$script"; then \
	    echo "[prototypes] $$script: ok"; \
	  else \
	    rc="$$?"; \
	    status=1; \
	    failed_scripts="$$failed_scripts $$script"; \
	    echo "[prototypes] $$script: failed with exit $$rc" >&2; \
	  fi; \
	done; \
	if [ "$$status" -ne 0 ]; then \
	  echo "[prototypes] failed scripts:$$failed_scripts" >&2; \
	fi; \
	exit "$$status"
