Final checkpoint only; entries are median ± std over finite generated evaluation repeats/trials. Heavy-tailed models are italicized.

**Dataset Recap**

| Dataset | Sample shape | Training samples | Eval samples |
|---|---:|---:|---:|
| CIFAR100-LT | 3 x 32 x 32 | 15,657 | 1,958 |
| HRRR | 1 x 100 x 100 | 81,505 | 10,189 |
| ImageNet-LT | 3 x 64 x 64 | 92,676 | 11,585 |
| LVIS | 3 x 64 x 64 | 80,136 | 10,017 |

**CIFAR100-LT**

| Model | Class recovery ↑ | MMD-RBF ↓ |
| --- | ---: | ---: |
| GF-Linear Euler | 76.0 ± 1.5% | **0.006 ± 0.002** |
| GF-Linear Heun | 78.0 ± 2.0% | 0.008 ± 0.002 |
| DDPM-V DDPM | 1.0 ± 0.0% | NaN |
| DDPM-V DDIM | **91.5 ± 1.8%** | 0.020 ± 0.006 |
| *DLPM alpha=1.7* | 87.0 ± 1.7% | 0.13 ± 0.010 |
| *DLPM alpha=1.9* | 85.5 ± 3.2% | 0.16 ± 0.018 |
| *TEDM nu=2.1* | 45.5 ± 1.7% | 0.51 ± 0.016 |
| *TEDM nu=3.0* | 53.0 ± 2.5% | 0.53 ± 0.020 |

**ImageNet-LT**

| Model | Class recovery ↑ | MMD-RBF ↓ |
| --- | ---: | ---: |
| GF-Linear Euler | 22.5 ± 0.6% | **0.006 ± 8.64e-04** |
| GF-Linear Heun | NaN | NaN |
| DDPM-V DDPM | 0.1 ± 0.0% | NaN |
| DDPM-V DDIM | **40.6 ± 0.6%** | 0.050 ± 0.006 |
| *DLPM alpha=1.7* | 30.5 ± 0.8% | 0.25 ± 0.026 |
| *DLPM alpha=1.9* | 27.8 ± 0.4% | 0.20 ± 0.013 |
| *TEDM nu=2.1* | 9.8 ± 0.5% | 0.62 ± 0.014 |
| *TEDM nu=3.0* | 11.2 ± 0.4% | 0.61 ± 0.016 |

**LVIS**

| Model | MMD-RBF ↓ |
| --- | ---: |
| GF-Linear Euler | **0.007 ± 0.001** |
| GF-Linear Heun | NaN |
| DDPM-V DDPM | NaN |
| DDPM-V DDIM | 0.049 ± 0.003 |
| *DLPM alpha=1.7* | 0.18 ± 0.013 |
| *DLPM alpha=1.9* | 0.18 ± 0.008 |
| *TEDM nu=2.1* | 0.60 ± 0.016 |
| *TEDM nu=3.0* | 0.60 ± 0.010 |

**DLPM Loss Power**

MMD-RBF values are read from the figure legend. TCE values are approximate read-offs from the vector PDF curves.

| Model | MMD-RBF ↓ | TCE@90 ↓ | TCE@99 ↓ |
| --- | ---: | ---: | ---: |
| *DLPM r=0.1* | 1.1e-02 | 3.34e-01 | 1.35e+00 |
| *DLPM r=0.3* | 1.3e-02 | 3.44e-01 | 1.28e+00 |
| *DLPM r=0.5* | 1.4e-02 | 3.32e-01 | 1.28e+00 |
| *DLPM r=0.7* | 2.4e-02 | 4.68e-01 | 1.68e+00 |
| *DLPM r=0.9* | 1.5e-01 | 1.02e+00 | 2.61e+00 |
| test vs true sample | 1.9e-04 | 1e-03 | 1.10e-02 |
