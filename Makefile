BASH            ?= bash
PYTHON          ?= python
VENV_DIR        ?= .venv
JZ_MODULE       ?= pytorch-gpu/py3/2.8.0
LOG_DIR         ?= logs
BENCH_MAIN      ?= benchmarks/01_main.py
BENCH_EVAL      ?= benchmarks/02_evaluate.py
BENCH_UTILS     ?= benchmarks/utils.py
PILOT_ANALYSIS  ?= benchmarks/03_pilot_analysis.py
BENCH_PLOTTING  ?= benchmarks/04_plotting_bench.py
ARTIFACT_DIR      ?= benchmarks/artifacts
LEGACY_ARTIFACT_DIR ?= benchmarks/data
PILOT_CONFIG_DIR ?= benchmarks/configs/pilot
BENCH_TEMPLATE_DIR ?= benchmarks/configs/templates
BENCH_CONFIG_DIR ?= benchmarks/configs/bench
PILOT_SYNTH_SHARDS ?= 15
PILOT_REAL_SHARDS ?= 8
PILOT_IMAGE_SHARDS ?= 15
EVAL_PILOT_SYNTH_SHARDS ?= 6
EVAL_PILOT_REAL_SHARDS ?= 3
EVAL_PILOT_IMAGE_SHARDS ?= 6
ifndef FLOWBENCH_DATA
  ifdef WORK
    FLOWBENCH_DATA := $(WORK)/flowbench_data
  endif
endif
export FLOWBENCH_DATA
VENV_PYTHON      = $(CURDIR)/$(VENV_DIR)/bin/python
PYTHONPATH_EXPORT = export PYTHONPATH="$(CURDIR):$(CURDIR)/src$${PYTHONPATH:+:$$PYTHONPATH}"
ACTIVATE         = source "$(VENV_DIR)/bin/activate"; $(PYTHONPATH_EXPORT)
RUN_LOG_ARGS     = --output="$(CURDIR)/$(LOG_DIR)/htfm_run_%A_%a.out" \
                   --error="$(CURDIR)/$(LOG_DIR)/htfm_run_%A_%a.err"
DATASET_LOG_ARGS = --output="$(CURDIR)/$(LOG_DIR)/htfm_dataset_%j.out" \
                   --error="$(CURDIR)/$(LOG_DIR)/htfm_dataset_%j.err"
EVAL_LOG_ARGS    = --output="$(CURDIR)/$(LOG_DIR)/htfm_evaluate_%A_%a.out" \
                   --error="$(CURDIR)/$(LOG_DIR)/htfm_evaluate_%A_%a.err"
SBATCH_EXPORT    = --export=ALL,VENV_DIR="$(CURDIR)/$(VENV_DIR)"
JZ_GPU_ARGS      ?= --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --partition=gpu_p13 --qos=qos_gpu-t3 --account=jcx@v100
PILOT_SBATCH_ARGS ?= $(JZ_GPU_ARGS) --time=04:00:00
BENCH_SBATCH_ARGS ?= $(JZ_GPU_ARGS) --time=20:00:00
EVAL_SBATCH_ARGS ?= $(JZ_GPU_ARGS) --time=02:00:00
DATASET_SBATCH_ARGS ?= --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --partition=gpu_p13 --qos=qos_gpu-t3 --account=jcx@v100 --time=01:00:00
EVAL_PILOT_ARGS ?= --n-eval-samples 2048 --n-eval-repeats 4 --inspect-samples 1024 --probe-size 128 --sample-batch-size 64 --max-fid-dim 2048 --max-mmd-dim 2048 --max-mmd-samples 4096 --max-inspect-dim 1024
EVAL_BENCH_TABULAR_ARGS ?= --n-eval-samples 2048 --n-eval-repeats 4 --inspect-samples 1024 --probe-size 128 --sample-batch-size 64 --max-fid-dim 2048 --max-mmd-dim 2048 --max-mmd-samples 4096 --max-inspect-dim 1024
EVAL_BENCH_IMAGE_ARGS ?= --n-eval-samples 4096 --n-eval-repeats 6 --inspect-samples 1024 --probe-size 128 --sample-batch-size 64 --max-fid-dim 2048 --max-mmd-dim 2048 --max-mmd-samples 4096 --max-inspect-dim 1024
PILOT_SYNTH_CONFIG ?= $(PILOT_CONFIG_DIR)/synth.yaml
PILOT_REAL_CONFIG ?= $(PILOT_CONFIG_DIR)/real.yaml
PILOT_IMAGE_CONFIG ?= $(PILOT_CONFIG_DIR)/image.yaml
PREFETCH_CONFIG_INPUTS = $(PILOT_CONFIG_DIR) $(BENCH_TEMPLATE_DIR) $(BENCH_CONFIG_DIR)

.DEFAULT_GOAL := help

.PHONY: setup dataset dataset-tabular dataset-tabular-init dataset-hrrr dataset-lvis dataset-cifar100-lt dataset-cifar100-lt-init dataset-imagenet-lt dataset-imagenet-lt-init pilot analyze-pilot bench evaluate-pilot evaluate-bench plotting-bench check send supp help

setup:
	$(BASH) scripts/setup.sh --venv-dir "$(VENV_DIR)" --use-jz-module
	USE_JZ_MODULE=1 JZ_MODULE="$(JZ_MODULE)" PYTHON="$(VENV_PYTHON)" $(BASH) scripts/fetch.vendor.sh

dataset:
	$(MAKE) dataset-tabular
	$(MAKE) dataset-hrrr
	$(MAKE) dataset-lvis
	$(MAKE) dataset-cifar100-lt
	$(MAKE) dataset-imagenet-lt

dataset-tabular: dataset-tabular-init
	@mkdir -p "$(LOG_DIR)"
	@if $(VENV_PYTHON) scripts/prefetch.datasets.py --check-only --only-dataset kddcup $(PREFETCH_CONFIG_INPUTS); then \
	  echo "[skip-sbatch] kddcup: all variants already cached"; \
	else \
	  sbatch --job-name=htfm_dataset_kddcup \
	    --output="$(CURDIR)/$(LOG_DIR)/htfm_dataset_kddcup_%j.out" \
	    --error="$(CURDIR)/$(LOG_DIR)/htfm_dataset_kddcup_%j.err" \
	    $(DATASET_SBATCH_ARGS) $(SBATCH_EXPORT) scripts/dataset.slurm.sh \
	    --only-dataset kddcup $(PREFETCH_CONFIG_INPUTS) ; \
	fi

dataset-tabular-init:
	@$(BASH) scripts/init.kddcup.sh

dataset-hrrr:
	@mkdir -p "$(LOG_DIR)"
	@if $(VENV_PYTHON) scripts/prefetch.datasets.py --check-only --only-dataset hrrr $(PREFETCH_CONFIG_INPUTS); then \
	  echo "[skip-sbatch] hrrr: all variants already cached"; \
	else \
	  sbatch --job-name=htfm_dataset_hrrr \
	    --output="$(CURDIR)/$(LOG_DIR)/htfm_dataset_hrrr_%j.out" \
	    --error="$(CURDIR)/$(LOG_DIR)/htfm_dataset_hrrr_%j.err" \
	    $(DATASET_SBATCH_ARGS) $(SBATCH_EXPORT) scripts/dataset.slurm.sh \
	    --only-dataset hrrr $(PREFETCH_CONFIG_INPUTS) ; \
	fi

dataset-lvis:
	@mkdir -p "$(LOG_DIR)"
	@if $(VENV_PYTHON) scripts/prefetch.datasets.py --check-only --only-dataset lvis $(PREFETCH_CONFIG_INPUTS); then \
	  echo "[skip-sbatch] lvis: all variants already cached"; \
	else \
	  sbatch --job-name=htfm_dataset_lvis \
	    --output="$(CURDIR)/$(LOG_DIR)/htfm_dataset_lvis_%j.out" \
	    --error="$(CURDIR)/$(LOG_DIR)/htfm_dataset_lvis_%j.err" \
	    $(DATASET_SBATCH_ARGS) $(SBATCH_EXPORT) scripts/dataset.slurm.sh \
	    --only-dataset lvis $(PREFETCH_CONFIG_INPUTS) ; \
	fi

dataset-cifar100-lt: dataset-cifar100-lt-init
	@mkdir -p "$(LOG_DIR)"
	@if $(VENV_PYTHON) scripts/prefetch.datasets.py --check-only --only-dataset cifar100_lt $(PREFETCH_CONFIG_INPUTS); then \
	  echo "[skip-sbatch] cifar100_lt: all variants already cached"; \
	else \
	  sbatch --job-name=htfm_dataset_cifar100_lt \
	    --output="$(CURDIR)/$(LOG_DIR)/htfm_dataset_cifar100_lt_%j.out" \
	    --error="$(CURDIR)/$(LOG_DIR)/htfm_dataset_cifar100_lt_%j.err" \
	    $(DATASET_SBATCH_ARGS) $(SBATCH_EXPORT) scripts/dataset.slurm.sh \
	    --only-dataset cifar100_lt $(PREFETCH_CONFIG_INPUTS) ; \
	fi

dataset-cifar100-lt-init:
	@$(BASH) scripts/init.cifar100_lt.sh

dataset-imagenet-lt: dataset-imagenet-lt-init
	@mkdir -p "$(LOG_DIR)"
	@if $(VENV_PYTHON) scripts/prefetch.datasets.py --check-only --only-dataset imagenet_lt $(PREFETCH_CONFIG_INPUTS); then \
	  echo "[skip-sbatch] imagenet_lt: all variants already cached"; \
	else \
	  sbatch --job-name=htfm_dataset_imagenet_lt \
	    --output="$(CURDIR)/$(LOG_DIR)/htfm_dataset_imagenet_lt_%j.out" \
	    --error="$(CURDIR)/$(LOG_DIR)/htfm_dataset_imagenet_lt_%j.err" \
	    $(DATASET_SBATCH_ARGS) $(SBATCH_EXPORT) scripts/dataset.slurm.sh \
	    --only-dataset imagenet_lt $(PREFETCH_CONFIG_INPUTS) ; \
	fi

dataset-imagenet-lt-init:
	@$(BASH) scripts/init.imagenet_lt.sh

define latest_batch
$(VENV_PYTHON) "$(BENCH_UTILS)" latest-batch --root "$(ARTIFACT_DIR)" --pattern "$(1)"
endef

pilot_array = 0-$(shell expr $(1) - 1)

define require_bench_configs
	test -d "$(BENCH_CONFIG_DIR)" || { echo "Missing: $(BENCH_CONFIG_DIR). Run 'make analyze-pilot' first." >&2; exit 2; }; \
	find "$(BENCH_CONFIG_DIR)" -name '*.yaml' -print -quit | grep -q . || { echo "No bench configs under $(BENCH_CONFIG_DIR). Run 'make analyze-pilot' first." >&2; exit 2; }
endef

define config_run_name
$(VENV_PYTHON) "$(BENCH_UTILS)" run-name --config "$(1)"
endef

define config_run_count
$(VENV_PYTHON) "$(BENCH_UTILS)" count-runs --config "$(1)"
endef

pilot:
	@mkdir -p "$(LOG_DIR)"
	sbatch --job-name=htfm_run --array=$(call pilot_array,$(PILOT_SYNTH_SHARDS)) $(PILOT_SBATCH_ARGS) $(RUN_LOG_ARGS) $(SBATCH_EXPORT) scripts/run.slurm.sh --config "$(PILOT_SYNTH_CONFIG)" --skip-existing
	sbatch --job-name=htfm_run --array=$(call pilot_array,$(PILOT_REAL_SHARDS)) $(PILOT_SBATCH_ARGS) $(RUN_LOG_ARGS) $(SBATCH_EXPORT) scripts/run.slurm.sh --config "$(PILOT_REAL_CONFIG)" --skip-existing
	sbatch --job-name=htfm_run --array=$(call pilot_array,$(PILOT_IMAGE_SHARDS)) $(PILOT_SBATCH_ARGS) $(RUN_LOG_ARGS) $(SBATCH_EXPORT) scripts/run.slurm.sh --config "$(PILOT_IMAGE_CONFIG)" --skip-existing

analyze-pilot:
	$(BASH) -lc '$(ACTIVATE); python "$(PILOT_ANALYSIS)"'

bench: analyze-pilot
	@mkdir -p "$(LOG_DIR)"
	@$(require_bench_configs)
	@find "$(BENCH_CONFIG_DIR)" -name '*.yaml' | sort | while read -r config; do \
	  count="$$($(call config_run_count,$$config))"; \
	  array_end="$$((count - 1))"; \
	  echo "[bench] submit $$config ($$count runs)"; \
	  sbatch --job-name=htfm_run --array="0-$$array_end" $(BENCH_SBATCH_ARGS) $(RUN_LOG_ARGS) $(SBATCH_EXPORT) scripts/run.slurm.sh --config "$$config" --skip-existing; \
	done

evaluate-pilot:
	@mkdir -p "$(LOG_DIR)"
	@test -d "$(ARTIFACT_DIR)" || { echo "Missing: $(ARTIFACT_DIR)" >&2; exit 2; }
	sbatch --job-name=htfm_evaluate --array=$(call pilot_array,$(EVAL_PILOT_SYNTH_SHARDS)) $(EVAL_SBATCH_ARGS) $(EVAL_LOG_ARGS) $(SBATCH_EXPORT) scripts/evaluate.slurm.sh \
	  --batch-dir "$$($(call latest_batch,*_pilot_synth))" $(EVAL_PILOT_ARGS)
	sbatch --job-name=htfm_evaluate --array=$(call pilot_array,$(EVAL_PILOT_REAL_SHARDS)) $(EVAL_SBATCH_ARGS) $(EVAL_LOG_ARGS) $(SBATCH_EXPORT) scripts/evaluate.slurm.sh \
	  --batch-dir "$$($(call latest_batch,*_pilot_real))" $(EVAL_PILOT_ARGS)
	sbatch --job-name=htfm_evaluate --array=$(call pilot_array,$(EVAL_PILOT_IMAGE_SHARDS)) $(EVAL_SBATCH_ARGS) $(EVAL_LOG_ARGS) $(SBATCH_EXPORT) scripts/evaluate.slurm.sh \
	  --batch-dir "$$($(call latest_batch,*_pilot_image))" $(EVAL_PILOT_ARGS)

evaluate-bench: analyze-pilot
	@mkdir -p "$(LOG_DIR)"
	@test -d "$(ARTIFACT_DIR)" || { echo "Missing: $(ARTIFACT_DIR)" >&2; exit 2; }
	@$(require_bench_configs)
	@find "$(BENCH_CONFIG_DIR)" -name '*.yaml' | sort | while read -r config; do \
	  run_name="$$($(call config_run_name,$$config))"; \
	  batch_dir="$$($(call latest_batch,*_$$run_name) 2>/dev/null || true)"; \
	  if [[ -z "$$batch_dir" ]]; then echo "[evaluate-bench] skip $$config -> no batch found under $(ARTIFACT_DIR)"; continue; fi; \
	  if echo "$$config" | grep -q '/image/'; then eval_args='$(EVAL_BENCH_IMAGE_ARGS)'; else eval_args='$(EVAL_BENCH_TABULAR_ARGS)'; fi; \
	  echo "[evaluate-bench] submit $$config -> $$batch_dir"; \
	  sbatch --job-name=htfm_evaluate $(EVAL_SBATCH_ARGS) $(EVAL_LOG_ARGS) $(SBATCH_EXPORT) scripts/evaluate.slurm.sh --batch-dir "$$batch_dir" $$eval_args; \
	done

plotting-bench:
	$(BASH) -lc '$(ACTIVATE); python "$(BENCH_PLOTTING)" --artifact-root "$(LEGACY_ARTIFACT_DIR)" --table-root benchmarks/tables --figure-root benchmarks/figures'

check:
	$(BASH) -lc '$(ACTIVATE); flake8 --ignore E501 --exclude src/genkit/_vendor src tests benchmarks examples'
	$(BASH) -lc '$(ACTIVATE); pytest -v'
	$(BASH) -lc '$(ACTIVATE); python "$(BENCH_MAIN)" --config tests/fixtures/benchmark_smoke.yaml'
	$(BASH) -lc '$(ACTIVATE); python examples/01_visu_1d_path.py --blank'
	$(BASH) -lc '$(ACTIVATE); python examples/02_visu_2d.py --blank'

send:
	$(BASH) scripts/transfer.sh send

supp:
	$(BASH) scripts/transfer.sh supp

help:
	@printf "Available targets:\n"
	@printf "  %-14s %s\n" "setup"      "Install Jean Zay environment"
	@printf "  %-14s %s\n" "dataset"    "Submit all processed real-data cache jobs via Slurm"
	@printf "  %-14s %s\n" "dataset-tabular" "Run KDD init on login node + submit KDDCup cache job"
	@printf "  %-14s %s\n" "dataset-hrrr" "Submit HRRR cache job"
	@printf "  %-14s %s\n" "dataset-lvis" "Submit LVIS cache job"
	@printf "  %-14s %s\n" "dataset-cifar100-lt" "Run init + submit CIFAR-100-LT cache job"
	@printf "  %-14s %s\n" "dataset-imagenet-lt" "Run init + submit ImageNet-LT cache job"
	@printf "  %-14s %s\n" "pilot"      "Submit pilot configs via Slurm"
	@printf "  %-14s %s\n" "analyze-pilot" "Generate per-dataset winners, reports, and bench configs"
	@printf "  %-14s %s\n" "bench"      "Generate and submit explicit benchmark configs via Slurm"
	@printf "  %-14s %s\n" "evaluate-pilot" "Submit pilot evaluation only"
	@printf "  %-14s %s\n" "evaluate-bench" "Submit benchmark evaluation only"
	@printf "  %-14s %s\n" "plotting-bench" "Generate benchmark tables and figures from current eval artifacts"
	@printf "  %-14s %s\n" "check"         "Run lint, tests, local smoke, and examples"
	@printf "  %-14s %s\n" "send"          "Send the project tree to the remote benchmark host"
	@printf "  %-14s %s\n" "supp"          "Build the supplementary code archive"
	@printf "  %-14s %s\n" "help"          "Print this help message"
