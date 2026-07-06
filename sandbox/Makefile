SHELL := /bin/bash

ROOT := $(abspath ..)
PYTHON ?= $(ROOT)/.venv/bin/python
JZ_MODULE ?= pytorch-gpu/py3/2.8.0
GPU_DEV_ARGS ?= --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --partition=gpu_p13 --qos=qos_gpu-t3 --account=jcx@v100 --time=10:00:00
SCRIPTS ?= 0_heteroscedastic.py 1_dlpm_loss_power.py 2_dlpm_mc_score.py 3_lora_tail_correction_model.py 4_contracted_space_gfl.py

.PHONY: node run

node:
	srun $(GPU_DEV_ARGS) --pty bash -lc 'module purge || true; module load "$(JZ_MODULE)"; exec bash'

run:
	@set -euo pipefail; \
	if command -v module >/dev/null 2>&1; then \
	  module purge || true; \
	  conda deactivate 2>/dev/null || true; \
	  module load "$(JZ_MODULE)"; \
	fi; \
	for script in $(SCRIPTS); do \
	  echo "[sandbox] $$script"; \
	  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$(ROOT):$(ROOT)/src$${PYTHONPATH:+:$$PYTHONPATH}" "$(PYTHON)" "$$script"; \
	done
