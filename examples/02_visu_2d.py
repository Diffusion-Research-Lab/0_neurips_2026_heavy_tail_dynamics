"""Simple 1D generation illustrative example."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from genkit import (DDPMV, GaussianFlowDDPM, GaussianFlowLinear, GaussianFlowOT, FlowMatchingOrigin,
                    ScoreSDEOrigin)
from genkit import AlphaStableFlowLinear, DLPMEps, DLPMEpsOrigin
from genkit.plotting import PRETTY_RCPARAMS, plot_scatter
from _utils import run_example


plt.rcParams.update(PRETTY_RCPARAMS)

figures_dir = "_figures"
figures_dir = Path(figures_dir)
figures_dir.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--blank", action="store_false", help="CI helper.")
parser.add_argument("--data", type=str, default='spiral', help="Data distribution.")
args = parser.parse_args()

light_tailed_data = ["balanced_bimodal_gaussian",
                     "unbalanced_bimodal_gaussian",
                     "gaussian",
                     "checker",
                     "spiral",
                     ]
light_tailed_models = [DDPMV,
                       GaussianFlowDDPM,
                       GaussianFlowLinear,
                       GaussianFlowOT,
                       FlowMatchingOrigin,
                       ScoreSDEOrigin,
                       ]

if args.data in light_tailed_data:
    models = light_tailed_models
else:
    models = [AlphaStableFlowLinear,
              DLPMEps,
              DLPMEpsOrigin,
              ]

exp_kwargs = dict(extra_data_kwargs=dict(),
                  n_steps=300,
                  batch_size=1024,
                  n_epochs=250,
                  lr=1e-3,
                  width=64,
                  depth=3,
                  extra_gen_kwargs=dict(),
                  device="cuda" if torch.cuda.is_available() else "cpu",
                  fdtype=torch.float32,
                  idtype=torch.int32,
                  )

results = run_example(models,
                      target_data_type=args.data,
                      n_samples=10_000 if args.blank else 100,
                      exp_kwargs=exp_kwargs,
                      verbose=True,
                      )

for i, (name, (x_gen, x_ref)) in enumerate(results.items()):
    plot_scatter(x=x_gen,
                 x_ref=x_ref,
                 plot_dir=figures_dir,
                 suffix=f"{args.data}_{name}",
                 fontsize=10,
                 alpha=0.6,
                 )
