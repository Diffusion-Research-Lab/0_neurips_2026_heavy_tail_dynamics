## Benchmark Quickstart

### 1) Setup

From the repo root, locally:

```bash
make setup-local
```

Or on Jean Zay:

```bash
make setup-jz
```

### 2) Run

Local smoke test:

```bash
bash benchmarks/launchers/local.sh --blank
```

Local full config sweep:

```bash
make run-local
```

Jean Zay multi-jobss run:

```bash
make run-jz
```
