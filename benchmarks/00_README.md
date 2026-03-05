## Benchmark Quickstart

### 1) Setup

From `benchmarks/`:

```bash
bash 01_setup.sh --env-name cauda --use-jz-module
```

This creates `.venv-cauda`, installs dependencies, and installs `cauda`.
Dependencies are split in 3 layers:
- `cauda/requirements.txt`
- `labkit/requirements.txt`
- `benchmarks/requirements.txt` (benchmark-specific, e.g. `torchvision`)

### 2) Smoke check

```bash
bash 01_setup.sh --env-name cauda --use-jz-module --check
```

### 3) Run benchmark (local)

```bash
bash 03_launcher.sh --run
```

Smoke-only pipeline:

```bash
bash 03_launcher.sh --blank
```

Cleanup generated benchmark artifacts:

```bash
bash 03_launcher.sh --clean
```

### 4) Run benchmark (Slurm)

Submit from `benchmarks/`:

```bash
sbatch 02_launcher.slurm
```

Artifacts are separated as:
- `_results/<bench>/<run_id>/` for `config.yml`, `results.json`, `run.txt`
- `_figures/<bench>/<run_id>/` for figures
- `_tables/<bench>/<run_id>/` for tables
