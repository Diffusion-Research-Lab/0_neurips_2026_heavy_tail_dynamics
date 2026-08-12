## Benchmarks

### Local

```bash
make send
ssh jz
cd "$WORK/src/Diffusion-Research-Lab/0_neurips_2026_heavy_tail_dynamics"
```

### Setup

```bash
make setup
make dataset
```

### Pilot

```bash
make pilot
make evaluate-pilot
make analyze-pilot
```

### Main

```bash
make bench
make evaluate-bench
make analyze-bench
```

### ImageNet-LT-96 Vizu

```bash
make imagenet128-viz-configs
make dataset DATASETS=imagenet_lt
make bench-imagenet128-viz
make visualize-imagenet128
```

### Shariatian

```bash
make bench-shariatan
```
