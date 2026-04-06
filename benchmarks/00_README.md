## Benchmark Quickstart

### 1) Setup

From the repo root:

```bash
make install-bench
```

Or, if you want the benchmark environment, asset prefetch, and optional checks handled for you:

```bash
make setup-local
```

From `benchmarks/`, the underlying setup script is:

```bash
bash 01_setup.sh
```

On Jean Zay:

```bash
bash 01_setup.sh --env-name genkit --use-jz-module --check
```

The `--check` mode runs unit tests only.

### 2) Run

Locally with bash:

```bash
make run-local
```

Smoke-only pipeline:

```bash
bash 03_launcher.sh --blank
```

Or with Slurm:

```bash
sbatch 02_launcher.slurm
```

The launcher now runs YAML configs from `benchmarks/configs/`:
- `--blank` runs `00_blank.yaml`
- `--run` runs all `*.yaml` files in lexical order (`00_...`, `01_...`, `03_...`, ...)

### 3) Clean artifacts

```bash
bash 03_launcher.sh --clean
```
