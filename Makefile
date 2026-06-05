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


# Output directories.
LOG_DIR                   ?= logs
ARTIFACT_DIR              ?= benchmarks/artifacts
TABLE_DIR                 ?= benchmarks/tables
FIGURE_DIR                ?= benchmarks/figures


# Config roots.
PILOT_CONFIG_DIR          ?= benchmarks/configs/pilot
BENCH_TEMPLATE_DIR        ?= benchmarks/configs/templates
BENCH_CONFIG_DIR          ?= benchmarks/configs/bench


# Dataset cache controls. Override DATASETS to submit a subset.
DATASETS                  ?= hrrr lvis cifar100_lt imagenet_lt
INIT_DATASETS             ?= cifar100_lt imagenet_lt


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
PILOT_SBATCH_ARGS         ?= $(JZ_GPU_ARGS) --time=04:00:00
BENCH_SBATCH_ARGS_SYNTH   ?= $(JZ_GPU_ARGS) --time=04:00:00
BENCH_SBATCH_ARGS_IMAGE   ?= --nodes=1 --ntasks=1 --cpus-per-task=15 --gres=gpu:1 --partition=gpu_p13 --qos=qos_gpu-t3 --account=jcx@v100 --time=10:00:00
EVAL_PILOT_SBATCH_ARGS    ?= $(JZ_GPU_DEV_ARGS) --time=00:50:00
EVAL_SBATCH_ARGS          ?= $(JZ_GPU_ARGS) --time=06:00:00
DATASET_SBATCH_ARGS       ?= --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --partition=gpu_p13 --qos=qos_gpu-t3 --account=jcx@v100 --time=01:00:00


# Evaluation options.
PILOT_SELECTION_ARGS      ?= --selection-only --selection-split val --selection-repeats 8 --selection-batch-size 64
EVAL_BENCH_SYNTH_ARGS     ?= --n-eval-samples 2048 --n-eval-repeats 4 --sample-batch-size 64 --max-mmd-samples 4096
EVAL_BENCH_IMAGE_ARGS     ?= --n-eval-samples 4096 --n-eval-repeats 6 --sample-batch-size 64 --max-mmd-samples 256
SHARIATAN_ARGS            ?=


# Pilot configs and dataset prefetch inputs.
PILOT_SYNTH_CONFIG        ?= $(PILOT_CONFIG_DIR)/synth.yaml
PILOT_IMAGE_CONFIG        ?= $(PILOT_CONFIG_DIR)/image.yaml
PREFETCH_CONFIG_INPUTS    = $(PILOT_CONFIG_DIR) $(BENCH_TEMPLATE_DIR) $(BENCH_CONFIG_DIR)


.DEFAULT_GOAL := help

.PHONY: setup dataset pilot analyze-pilot bench bench-shariatan evaluate-pilot evaluate-bench \
        analyze-bench check send supp help


# Environment bootstrap.
setup:
	$(BASH) scripts/setup.sh --venv-dir "$(VENV_DIR)" --use-jz-module
	USE_JZ_MODULE=1 JZ_MODULE="$(JZ_MODULE)" PYTHON="$(VENV_PYTHON)" $(BASH) scripts/fetch.vendor.sh


# Dataset cache submission.
dataset:
	@mkdir -p "$(LOG_DIR)"
	@for dataset in $(DATASETS); do \
	  case " $(INIT_DATASETS) " in \
	    *" $$dataset "*) $(RUN_PYTHON) -m datakit init "$$dataset" ;; \
	  esac; \
	  if $(RUN_PYTHON) scripts/prefetch.datasets.py --check-only --only-dataset "$$dataset" $(PREFETCH_CONFIG_INPUTS); then \
	    echo "[skip-sbatch] $$dataset: all variants already cached"; \
	  else \
	    sbatch --job-name=htfm_dataset_$${dataset} \
	      --output="$(CURDIR)/$(LOG_DIR)/htfm_dataset_$${dataset}_%j.out" \
	      --error="$(CURDIR)/$(LOG_DIR)/htfm_dataset_$${dataset}_%j.err" \
	      $(DATASET_SBATCH_ARGS) $(SBATCH_EXPORT) scripts/dataset.slurm.sh \
	      --only-dataset "$$dataset" $(PREFETCH_CONFIG_INPUTS) ; \
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
	@printf "  %-22s %s\n" "dataset" "Submit image dataset cache jobs; override with DATASETS=hrrr"
	@printf "  %-22s %s\n" "pilot" "Submit pilot configs via Slurm"
	@printf "  %-22s %s\n" "analyze-pilot" "Generate per-dataset winners, reports, and bench configs"
	@printf "  %-22s %s\n" "bench" "Generate and submit explicit benchmark configs via Slurm"
	@printf "  %-22s %s\n" "bench-shariatan" "Run standalone Shariatian et al. benchmark"
	@printf "  %-22s %s\n" "evaluate-pilot" "Submit pilot evaluation only"
	@printf "  %-22s %s\n" "evaluate-bench" "Submit benchmark evaluation only"
	@printf "  %-22s %s\n" "analyze-bench" "Generate benchmark tables and figures from current eval artifacts"
	@printf "  %-22s %s\n" "check" "Run lint, tests, local smoke, and examples"
	@printf "  %-22s %s\n" "send" "Send code and configs only to the remote benchmark host"
	@printf "  %-22s %s\n" "supp" "Build the supplementary code archive"
	@printf "  %-22s %s\n" "help" "Print this help message"
