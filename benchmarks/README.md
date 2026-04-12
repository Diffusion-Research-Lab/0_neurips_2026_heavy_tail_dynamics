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

Jean Zay full run:

```bash
make run-jz
```

Override config or shard count when needed:

```bash
make run-jz CONFIG=benchmarks/configs/02_dimension_effect.yaml ARRAY=0-15
make run-jz CONFIG=benchmarks/configs/03_modecollapsing_effect.yaml ARRAY=0-5
```
