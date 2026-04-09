## Benchmark Quickstart

### 1) Setup

From the repo root, locally:

```bash
make setup-local
```

Or on Jean Zay:

```bash
make setup-server
```

### 2) Run

Local smoke test:

```bash
bash benchmarks/03_launcher.sh --blank
```

Local full config sweep:

```bash
bash benchmarks/03_launcher.sh --run
```

Jean Zay single-job run:

```bash
sbatch benchmarks/02_launcher.slurm
```

Jean Zay recommended run for the big benchmark:

```bash
sbatch --array=0-7 benchmarks/02_launcher.slurm
```
