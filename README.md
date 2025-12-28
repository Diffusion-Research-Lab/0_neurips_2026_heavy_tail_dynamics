## Heavy-Tail Flow Matching

Official code release for **Heavy-Tail Flow Matching** (NeurIPS 2026).

This repository provides:
- reference implementations of heavy-tailed bridges / noise models,
- training code for flow matching baselines and heavy-tail variants,
- evaluation metrics for heavy-tailed generative modeling,
- scripts to reproduce the main figures and tables.

### Results reproduction

To reproduce the figures of the paper, run the command::

    cd benchmarks
    bash bench_0_run_all.sh


### Citation

If you use this code, please cite:
```bib
@inproceedings{cherkaoui2026heavytailedfm,
  title     = {Heavy-Tail Flow Matching},
  author    = {Cherkaoui, Hamza and Antonio Ocello and Hélèné Halconruy},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
