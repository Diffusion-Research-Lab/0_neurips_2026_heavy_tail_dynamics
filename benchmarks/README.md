# Benchmarks

Send the project to Jean Zay, then run the selected experiment there:

```bash
make send
ssh jz
cd "$WORK/src/Diffusion-Research-Lab/0_neurips_2026_heavy_tail_dynamics"
```

```bash
make install
make dataset
make pilot
make evaluate-pilot
make analyze-pilot
make bench
make evaluate-bench
make analyze-bench
```

Optional ImageNet-LT visualization:

```bash
make imagenet128-viz-configs
make dataset DATASETS=imagenet_lt
make bench-imagenet128-viz
make visualize-imagenet128
```

Optional Shariatian benchmark:

```bash
make bench-shariatan
```
