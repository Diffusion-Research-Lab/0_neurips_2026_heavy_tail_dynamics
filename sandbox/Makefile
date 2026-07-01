SHELL := /bin/bash

ROOT := $(abspath ..)
PYTHON ?= $(ROOT)/.venv/bin/python
JZ_MODULE ?= pytorch-gpu/py3/2.8.0
GPU_DEV_ARGS ?= --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --partition=gpu_p13 --qos=qos_gpu-t3 --account=jcx@v100 --time=12:00:00
SCRIPTS ?= 0_heteroscedastic.py 1_dlpm_loss_power.py 2_dlpm_mc_score.py 3_guidance.py 4_lora_tail_correction_model.py 5_smc_tail_correction_model.py 6_contracted_space_gfl.py 7_error.py

.PHONY: node run

node:
	srun $(GPU_DEV_ARGS) --pty bash -lc 'module purge || true; module load "$(JZ_MODULE)"; exec bash'

run:
	@set -euo pipefail; \
	for script in $(SCRIPTS); do \
	  echo "[sandbox] $$script"; \
	  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$(ROOT):$(ROOT)/src$${PYTHONPATH:+:$$PYTHONPATH}" "$(PYTHON)" "$$script"; \
	done
