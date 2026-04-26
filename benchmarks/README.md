## Benchmark Quickstart

### 1) Setup

From the repo root, locally:

```bash
bash scripts/setup.sh --venv-dir .venv
```

Or on Jean Zay:

```bash
make setup
```

### 2) Run

Local smoke test:

```bash
bash scripts/run.local.sh --blank
```

Local full config sweep:

```bash
bash scripts/run.local.sh --run
```

Jean Zay benchmark run:

```bash
make run
```

Launch sharded evaluation for one fetched or remote batch:

```bash
make evaluate
```
