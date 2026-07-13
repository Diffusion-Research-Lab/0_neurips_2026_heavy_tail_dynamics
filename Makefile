# Local shell and environment.
BASH                      ?= bash
SHELL                     := $(BASH)
VENV_DIR                  ?= .venv
JZ_MODULE                 ?= pytorch-gpu/py3/2.8.0


# Python entry points.
BENCH_MAIN                ?= benchmarks/01_main.py
BENCH_UTILS               ?= benchmarks/utils.py
PILOT_ANALYSIS            ?= benchmarks/03_pilot_analysis.py
BENCH_PLOTTING            ?= benchmarks/04_plotting_bench.py
SHARIATAN_BENCH           ?= benchmarks/05_shariatan_et_al.py
IMAGENET128_VIZ_CONFIGS   ?= benchmarks/06_make_imagenet128_viz.py
IMAGENET128_VIZ_SCRIPT    ?= benchmarks/07_visualize_imagenet128.py


# Output directories.
LOG_DIR                   ?= logs
ARTIFACT_DIR              ?= benchmarks/artifacts
TABLE_DIR                 ?= benchmarks/tables
FIGURE_DIR                ?= benchmarks/figures


# Config roots.
PILOT_CONFIG_DIR          ?= benchmarks/configs/pilot
BENCH_TEMPLATE_DIR        ?= benchmarks/configs/templates
BENCH_CONFIG_DIR          ?= benchmarks/configs/bench
VIZ_CONFIG_DIR            ?= benchmarks/configs/viz
IMAGENET128_VIZ_CONFIG_DIR ?= $(VIZ_CONFIG_DIR)/imagenet_lt_96


# Dataset cache controls. Override DATASETS to submit a subset.
DATASETS                  ?= hrrr lvis cifar100_lt imagenet_lt
INIT_DATASETS             ?= cifar100_lt imagenet_lt
DATASET_OVERWRITE         ?= 0
DATASET_HRRR_ENSURE       ?= 1
DATASET_CPUS              ?= 20
HRRR_ENSURE_ARGS          ?= --workers 4
HRRR_STATUS               ?= $(CURDIR)/$(LOG_DIR)/hrrr_ensure_status.json


# Slurm array sizes.
PILOT_SYNTH_SHARDS        ?= 15
PILOT_IMAGE_SHARDS        ?= 15
EVAL_PILOT_SYNTH_SHARDS   ?= 4
EVAL_PILOT_IMAGE_SHARDS   ?= 4


# Jean Zay data root fallback.
ifndef FLOWBENCH_DATA
  ifdef WORK
    FLOWBENCH_DATA := $(WORK)/flowbench_data
  endif
endif
export FLOWBENCH_DATA


# Python helpers.
VENV_BIN                  = $(CURDIR)/$(VENV_DIR)/bin
VENV_PYTHON               = $(VENV_BIN)/python
PROJECT_PYTHONPATH        = $(CURDIR):$(CURDIR)/src
PYTHON_ENV                = PYTHONPATH="$(PROJECT_PYTHONPATH)$${PYTHONPATH:+:$$PYTHONPATH}"
# Run project Python commands without activating the virtual environment.
RUN_PYTHON                = $(PYTHON_ENV) "$(VENV_PYTHON)"


# Shared Slurm plumbing.
ARRAY_LOG_ARGS            = --output="$(CURDIR)/$(LOG_DIR)/%x_%A_%a.out" \
                            --error="$(CURDIR)/$(LOG_DIR)/%x_%A_%a.err"
SBATCH_EXPORT             = --export=ALL,VENV_DIR="$(CURDIR)/$(VENV_DIR)"


# Slurm resource presets.
JZ_GPU_ARGS               ?= --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --partition=gpu_p13 --qos=qos_gpu-t3 --account=jcx@v100
JZ_GPU_DEV_ARGS           ?= --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --partition=gpu_p13 --qos=qos_gpu-dev --account=jcx@v100
PILOT_SBATCH_ARGS         ?= $(JZ_GPU_ARGS) --time=20:00:00
BENCH_SBATCH_ARGS_SYNTH   ?= $(JZ_GPU_ARGS) --time=04:00:00
BENCH_SBATCH_ARGS_IMAGE   ?= --nodes=1 --ntasks=1 --cpus-per-task=15 --gres=gpu:1 --partition=gpu_p13 --qos=qos_gpu-t3 --account=jcx@v100 --time=20:00:00
IMAGENET128_VIZ_SBATCH_ARGS ?= --nodes=1 --ntasks=1 --cpus-per-task=15 --gres=gpu:1 --partition=gpu_p13 --qos=qos_gpu-t3 --account=jcx@v100 --time=30:00:00
EVAL_PILOT_SBATCH_ARGS    ?= $(JZ_GPU_DEV_ARGS) --time=00:50:00
EVAL_SBATCH_ARGS          ?= $(JZ_GPU_ARGS) --time=06:00:00
DATASET_SBATCH_ARGS       ?= --nodes=1 --ntasks=1 --cpus-per-task=$(DATASET_CPUS) --gres=gpu:1 --partition=gpu_p13 --qos=qos_gpu-t3 --account=jcx@v100 --time=20:00:00


# Evaluation options.
PILOT_SELECTION_ARGS      ?= --selection-only --selection-split val --selection-repeats 4 --selection-batch-size 64
EVAL_BENCH_SYNTH_ARGS     ?= --n-eval-samples 2048 --n-eval-repeats 4 --sample-batch-size 64 --max-mmd-samples 4096
EVAL_BENCH_IMAGE_ARGS     ?= --n-eval-samples 4096 --n-eval-repeats 6 --sample-batch-size 64 --max-mmd-samples 256
SHARIATAN_ARGS            ?=


# Pilot configs and dataset prefetch inputs.
PILOT_SYNTH_CONFIG        ?= $(PILOT_CONFIG_DIR)/synth.yaml
PILOT_IMAGE_CONFIG        ?= $(PILOT_CONFIG_DIR)/image.yaml
DATASET_CONFIG_INPUTS     ?= $(PILOT_IMAGE_CONFIG) $(BENCH_TEMPLATE_DIR)/image_bench.yaml $(BENCH_CONFIG_DIR)/image $(VIZ_CONFIG_DIR)


.DEFAULT_GOAL := help

.PHONY: setup dataset dataset-hrrr pilot analyze-pilot bench bench-shariatan imagenet128-viz-configs \
        bench-imagenet128-viz visualize-imagenet128 evaluate-pilot evaluate-bench analyze-bench \
        check send supp help


# Environment bootstrap.
setup:
	$(BASH) scripts/setup.sh --venv-dir "$(VENV_DIR)" --use-jz-module
	USE_JZ_MODULE=1 JZ_MODULE="$(JZ_MODULE)" PYTHON="$(VENV_PYTHON)" $(BASH) scripts/fetch.vendor.sh


# Dataset cache submission.
dataset-hrrr:
	@mkdir -p "$(LOG_DIR)"
	@set -euo pipefail; \
	  if ! command -v module >/dev/null 2>&1; then \
	    for init_script in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do \
	      if [ -r "$$init_script" ]; then . "$$init_script"; break; fi; \
	    done; \
	  fi; \
	  if ! command -v module >/dev/null 2>&1; then \
	    echo "[hrrr-login] 'module' command is required to load $(JZ_MODULE)." >&2; \
	    exit 1; \
	  fi; \
	  module purge || true; \
	  conda deactivate 2>/dev/null || true; \
	  module load "$(JZ_MODULE)"; \
	  hrrr_args="$(HRRR_ENSURE_ARGS)"; \
	  if [ -n "$${HRRR_MIN_SAMPLES:-}" ]; then hrrr_args="$$hrrr_args --min-samples $$HRRR_MIN_SAMPLES"; fi; \
	  if [ -n "$${HRRR_MIN_COVERAGE:-}" ]; then hrrr_args="$$hrrr_args --min-coverage $$HRRR_MIN_COVERAGE"; fi; \
	  echo "[hrrr-login] ensuring raw HRRR tensor on the login node with $(JZ_MODULE)"; \
	  $(RUN_PYTHON) scripts/ensure_hrrr.py --status-file "$(HRRR_STATUS)" $$hrrr_args

ifeq ($(DATASET_HRRR_ENSURE),1)
ifneq (,$(filter hrrr,$(DATASETS)))
dataset: dataset-hrrr
endif
endif

dataset:
	@mkdir -p "$(LOG_DIR)"
	@for dataset in $(DATASETS); do \
	  case " $(INIT_DATASETS) " in \
	    *" $$dataset "*) $(RUN_PYTHON) -m datakit init "$$dataset" ;; \
	  esac; \
	  submit=0; \
	  force_overwrite=0; \
	  if [ "$$dataset" = "hrrr" ] && [ "$(DATASET_HRRR_ENSURE)" = "1" ]; then \
	    if [ ! -f "$(HRRR_STATUS)" ]; then \
	      echo "[dataset] missing HRRR status file after dataset-hrrr: $(HRRR_STATUS)" >&2; \
	      exit 1; \
	    fi; \
	    if grep -q '"status": "built"' "$(HRRR_STATUS)"; then \
	      submit=1; \
	      force_overwrite=1; \
	    fi; \
	  fi; \
	  if [ "$$submit" != "1" ] && [ "$(DATASET_OVERWRITE)" = "1" ]; then \
	    submit=1; \
	    force_overwrite=1; \
	  elif [ "$$submit" != "1" ]; then \
	    $(RUN_PYTHON) scripts/prefetch.datasets.py --check-only --only-dataset "$$dataset" $(DATASET_CONFIG_INPUTS); \
	    status="$$?"; \
	    if [ "$$status" = "0" ]; then \
	      echo "[skip-sbatch] $$dataset: all variants already cached"; \
	    elif [ "$$status" = "2" ]; then \
	      submit=1; \
	    else \
	      exit "$$status"; \
	    fi; \
	  fi; \
	  if [ "$$submit" = "1" ]; then \
	    overwrite_arg=""; \
	    if [ "$$force_overwrite" = "1" ]; then overwrite_arg="--overwrite"; fi; \
	    sbatch --job-name=htfm_dataset_$${dataset} \
	      --output="$(CURDIR)/$(LOG_DIR)/htfm_dataset_$${dataset}_%j.out" \
	      --error="$(CURDIR)/$(LOG_DIR)/htfm_dataset_$${dataset}_%j.err" \
	      $(DATASET_SBATCH_ARGS) $(SBATCH_EXPORT) scripts/dataset.slurm.sh \
	      $$overwrite_arg --only-dataset "$$dataset" $(DATASET_CONFIG_INPUTS) ; \
	  fi; \
	done


# Small helpers used by benchmark targets.
define latest_batch
$(RUN_PYTHON) "$(BENCH_UTILS)" latest-batch --root "$(ARTIFACT_DIR)" --pattern "$(1)"
endef

pilot_array = 0-$(shell expr $(1) - 1)

define require_bench_configs
	test -d "$(BENCH_CONFIG_DIR)" || { echo "Missing: $(BENCH_CONFIG_DIR). Run 'make analyze-pilot' first." >&2; exit 2; }; \
	find "$(BENCH_CONFIG_DIR)" -name '*.yaml' -print -quit | grep -q . || { echo "No bench configs under $(BENCH_CONFIG_DIR). Run 'make analyze-pilot' first." >&2; exit 2; }
endef

define config_run_count
$(RUN_PYTHON) "$(BENCH_UTILS)" count-runs --config "$(1)"
endef


# Pilot and benchmark submission.
pilot:
	@mkdir -p "$(LOG_DIR)"
	sbatch --job-name=htfm_pilot --array=$(call pilot_array,$(PILOT_SYNTH_SHARDS)) $(PILOT_SBATCH_ARGS) $(ARRAY_LOG_ARGS) $(SBATCH_EXPORT) scripts/run.slurm.sh --config "$(PILOT_SYNTH_CONFIG)" --skip-existing
	sbatch --job-name=htfm_pilot --array=$(call pilot_array,$(PILOT_IMAGE_SHARDS)) $(PILOT_SBATCH_ARGS) $(ARRAY_LOG_ARGS) $(SBATCH_EXPORT) scripts/run.slurm.sh --config "$(PILOT_IMAGE_CONFIG)" --skip-existing

analyze-pilot:
	$(RUN_PYTHON) "$(PILOT_ANALYSIS)"


bench:
	@mkdir -p "$(LOG_DIR)"
	@$(require_bench_configs)
	@find "$(BENCH_CONFIG_DIR)" -name '*.yaml' | sort | while read -r config; do \
	  count="$$($(call config_run_count,$$config))"; \
	  array_end="$$((count - 1))"; \
	  case "$$config" in \
	    */image/*) bench_sbatch_args='$(BENCH_SBATCH_ARGS_IMAGE)' ;; \
	    *) bench_sbatch_args='$(BENCH_SBATCH_ARGS_SYNTH)' ;; \
	  esac; \
	  echo "[bench] submit $$config ($$count runs)"; \
	  sbatch --job-name=htfm_bench --array="0-$$array_end" $$bench_sbatch_args $(ARRAY_LOG_ARGS) $(SBATCH_EXPORT) scripts/run.slurm.sh --config "$$config" --skip-existing; \
	done

imagenet128-viz-configs:
	$(RUN_PYTHON) "$(IMAGENET128_VIZ_CONFIGS)" --bench-config-root "$(BENCH_CONFIG_DIR)" --output-root "$(IMAGENET128_VIZ_CONFIG_DIR)"

bench-imagenet128-viz: imagenet128-viz-configs
	@mkdir -p "$(LOG_DIR)"
	@$(RUN_PYTHON) scripts/prefetch.datasets.py --check-only --only-dataset imagenet_lt "$(IMAGENET128_VIZ_CONFIG_DIR)" || { \
	  status="$$?"; \
	  if [ "$$status" = "2" ]; then \
	    echo "[bench-imagenet128-viz] missing ImageNet-LT-96 processed cache." >&2; \
	    echo "[bench-imagenet128-viz] Run first: make dataset DATASETS=imagenet_lt DATASET_CONFIG_INPUTS=\"$(IMAGENET128_VIZ_CONFIG_DIR)\"" >&2; \
	  fi; \
	  exit "$$status"; \
	}
	@find "$(IMAGENET128_VIZ_CONFIG_DIR)" -name '*.yaml' | sort | while read -r config; do \
	  count="$$($(call config_run_count,$$config))"; \
	  array_end="$$((count - 1))"; \
	  echo "[bench-imagenet128-viz] submit $$config ($$count runs)"; \
	  sbatch --job-name=htfm_imagenet128_viz --array="0-$$array_end" $(IMAGENET128_VIZ_SBATCH_ARGS) $(ARRAY_LOG_ARGS) $(SBATCH_EXPORT) scripts/run.slurm.sh --config "$$config" --skip-existing; \
	done

visualize-imagenet128:
	$(RUN_PYTHON) "$(IMAGENET128_VIZ_SCRIPT)" --config-root "$(IMAGENET128_VIZ_CONFIG_DIR)" --artifact-root "$(ARTIFACT_DIR)" --output-dir "$(FIGURE_DIR)/imagenet_lt_96_viz"

bench-shariatan:
	$(RUN_PYTHON) "$(SHARIATAN_BENCH)" $(SHARIATAN_ARGS)

# Evaluation submission.
evaluate-pilot:
	@mkdir -p "$(LOG_DIR)"
	@test -d "$(ARTIFACT_DIR)" || { echo "Missing: $(ARTIFACT_DIR)" >&2; exit 2; }
	@pilot_synth_batch="$$($(call latest_batch,*_synth) 2>/dev/null || true)"; \
	test -n "$$pilot_synth_batch" || { echo "No synth pilot batch found under $(ARTIFACT_DIR)" >&2; exit 2; }; \
	sbatch --job-name=htfm_eval_pilot --array=$(call pilot_array,$(EVAL_PILOT_SYNTH_SHARDS)) $(EVAL_PILOT_SBATCH_ARGS) $(ARRAY_LOG_ARGS) $(SBATCH_EXPORT) scripts/evaluate.slurm.sh \
	  --batch-dir "$$pilot_synth_batch" $(PILOT_SELECTION_ARGS)
	@pilot_image_batch="$$($(call latest_batch,*_image) 2>/dev/null || true)"; \
	test -n "$$pilot_image_batch" || { echo "No image pilot batch found under $(ARTIFACT_DIR)" >&2; exit 2; }; \
	sbatch --job-name=htfm_eval_pilot --array=$(call pilot_array,$(EVAL_PILOT_IMAGE_SHARDS)) $(EVAL_PILOT_SBATCH_ARGS) $(ARRAY_LOG_ARGS) $(SBATCH_EXPORT) scripts/evaluate.slurm.sh \
	  --batch-dir "$$pilot_image_batch" $(PILOT_SELECTION_ARGS)

evaluate-bench:
	@mkdir -p "$(LOG_DIR)"
	@test -d "$(ARTIFACT_DIR)" || { echo "Missing: $(ARTIFACT_DIR)" >&2; exit 2; }
	@$(require_bench_configs)
	@find "$(BENCH_CONFIG_DIR)" -name '*.yaml' | sort | while read -r config; do \
	  batch_dir="$$($(RUN_PYTHON) "$(BENCH_UTILS)" latest-config-batch --root "$(ARTIFACT_DIR)" --config "$$config" 2>/dev/null || true)"; \
	  if [[ -z "$$batch_dir" ]]; then echo "[evaluate-bench] skip $$config -> no batch found under $(ARTIFACT_DIR)"; continue; fi; \
	  case "$$config" in \
	    */image/*) eval_args='$(EVAL_BENCH_IMAGE_ARGS)' ;; \
	    *) eval_args='$(EVAL_BENCH_SYNTH_ARGS)' ;; \
	  esac; \
	  echo "[evaluate-bench] submit $$config -> $$batch_dir"; \
	  sbatch --job-name=htfm_eval_bench $(EVAL_SBATCH_ARGS) $(ARRAY_LOG_ARGS) $(SBATCH_EXPORT) scripts/evaluate.slurm.sh --batch-dir "$$batch_dir" $$eval_args; \
	done


# Local reporting and checks.
analyze-bench:
	$(RUN_PYTHON) "$(BENCH_PLOTTING)" --artifact-root "$(ARTIFACT_DIR)" --table-root "$(TABLE_DIR)" --figure-root "$(FIGURE_DIR)"

check:
	$(PYTHON_ENV) "$(VENV_BIN)/flake8" --ignore E501 --exclude src/genkit/_vendor src tests benchmarks examples
	$(PYTHON_ENV) "$(VENV_BIN)/pytest" -v
	$(RUN_PYTHON) "$(BENCH_MAIN)" --config tests/fixtures/benchmark_smoke.yaml --fail-on-error
	$(RUN_PYTHON) examples/01_visu_1d_path.py --blank
	$(RUN_PYTHON) examples/02_visu_2d.py --blank


# Transfer and archive helpers.
send:
	$(BASH) scripts/transfer.sh send

supp:
	$(BASH) scripts/transfer.sh supp


# Target index.
help:
	@printf "Available targets:\n"
	@printf "  %-22s %s\n" "setup" "Install Jean Zay environment"
	@printf "  %-22s %s\n" "dataset" "Submit real image-cache jobs; override with DATASETS=hrrr"
	@printf "  %-22s %s\n" "dataset-hrrr" "Fetch/grow raw HRRR tensor on the login node"
	@printf "  %-22s %s\n" "pilot" "Submit pilot configs via Slurm"
	@printf "  %-22s %s\n" "analyze-pilot" "Generate per-dataset winners, reports, and bench configs"
	@printf "  %-22s %s\n" "bench" "Generate and submit explicit benchmark configs via Slurm"
	@printf "  %-22s %s\n" "imagenet128-viz-configs" "Generate ImageNet-LT-96 viz configs from selected bench configs"
	@printf "  %-22s %s\n" "bench-imagenet128-viz" "Submit ImageNet-LT-96 visualization training jobs"
	@printf "  %-22s %s\n" "visualize-imagenet128" "Sample visualization grids from ImageNet-LT-96 runs"
	@printf "  %-22s %s\n" "bench-shariatan" "Run standalone Shariatian et al. benchmark"
	@printf "  %-22s %s\n" "evaluate-pilot" "Submit pilot evaluation only"
	@printf "  %-22s %s\n" "evaluate-bench" "Submit benchmark evaluation only"
	@printf "  %-22s %s\n" "analyze-bench" "Generate benchmark tables and figures from current eval artifacts"
	@printf "  %-22s %s\n" "check" "Run lint, tests, local smoke, and examples"
	@printf "  %-22s %s\n" "send" "Send code and configs only to the remote benchmark host"
	@printf "  %-22s %s\n" "supp" "Build the supplementary code archive"
	@printf "  %-22s %s\n" "help" "Print this help message"
