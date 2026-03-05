## Benchmark Quickstart

### 1) Setup

From `benchmarks/`:

```bash
bash 01_setup.sh --env-name cauda --use-jz-module
```

This creates `.venv-cauda`, installs dependencies, and installs `cauda`.

### 2) Smoke check

```bash
bash 01_setup.sh --env-name cauda --use-jz-module --check
```

### 3) Run benchmark (local)

```bash
bash 03_launcher.sh --run
```

### 4) Run benchmark (Slurm)

Submit from `benchmarks/`:

```bash
sbatch 02_launcher.slurm
```
