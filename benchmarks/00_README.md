## Benchmark Quickstart

### 1) Setup

From `benchmarks/`:

```bash
bash 01_setup.sh --env-name cauda --use-jz-module
```

Or:

```bash
bash 01_setup.sh --env-name cauda --use-jz-module --check
```

### 2) Run

Locally with bash:

```bash
bash 03_launcher.sh --run
```

Or smoke-only pipeline:

```bash
bash 03_launcher.sh --blank
```

Or with Slurm:

```bash
sbatch 02_launcher.slurm
```

### 3) Clean artifacts

```bash
bash 03_launcher.sh --clean
```