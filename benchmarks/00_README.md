## Benchmark Quickstart

### 1) Setup

From the repo root:

```bash
make install-bench
```

Or, if you want the benchmark environment, asset prefetch, and optional checks handled for you:

```bash
make bench-setup
```

From `benchmarks/`, the underlying setup script is:

```bash
bash 01_setup.sh
```

On Jean Zay:

```bash
bash 01_setup.sh --env-name genkit --use-jz-module --check
```

The `--check` mode runs:
- unit tests
- `examples/02_visu_2d.py --blank`
- the benchmark blank pipeline for `04_alpha_values_benchmark_*` and `05_loss_H_comparison_benchmark_*`

### 2) Run

Locally with bash:

```bash
make bench-run
```

Or smoke-only pipeline with the `*_blank.yml` configs:

```bash
make bench-blank
```

Or with Slurm:

```bash
sbatch 02_launcher.slurm
```

### 3) Clean artifacts

```bash
make bench-clean
```
