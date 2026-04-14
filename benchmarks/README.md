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
bash scripts/run.local.sh --blank
```

Local full config sweep:

```bash
make run-local
```

Jean Zay full run:

```bash
make run-jz
```

Inspect one submitted array job:

```bash
make inspect-jz JOBID=1985827
make log-jz JOBID=1985827
make log-jz JOBID=1985827 TASK=3
```

Launch sharded evaluation for one fetched or remote batch:

```bash
make evaluate-jz
make log-jz LOG_PREFIX=htfm_eval JOBID=1986001 TASK=3
```
