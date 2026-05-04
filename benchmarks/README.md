## Benchmark Quickstart

To setup the benchmark:

```bash
  make send
  ssh jz
  cd $WORK/src/flowbench/
  make setup
  make dataset
```

Dataset caches live under `flowbench_data/`. Benchmark run outputs live under `benchmarks/artifacts/`.

To launch the pilot:

```bash
  make pilot
  make evaluate-pilot
  make analyze-pilot
```

To launch the benchmark:

```bash
  make bench
  make evaluate-bench
```
