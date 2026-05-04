## Benchmark Quickstart

To setup the benchmark:

```bash
  make send
  ssh jz
  cd $WORK/src/flowbench/
  make setup
  make dataset
```

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
