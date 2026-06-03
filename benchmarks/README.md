## Benchmarks

Run each step after the previous Slurm jobs are complete.

```bash
make send
ssh jz
cd "$WORK/src/flowbench"

make setup
make dataset

make pilot
make evaluate-pilot
make analyze-pilot

make bench
make evaluate-bench
make plotting-bench

make bench-shariatan
```
