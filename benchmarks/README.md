## Benchmarks

Run the steps in order. Wait for each Slurm stage to finish before starting the next one.

Local machine:

```bash
make send
ssh jz
cd "$WORK/src/flowbench"
```

Jean Zay login node:

```bash
make setup
make dataset
make pilot
make evaluate-pilot
make analyze-pilot
make bench
make evaluate-bench
make analyze-bench
make bench-shariatan
```
