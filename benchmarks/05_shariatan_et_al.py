"""Reproduce the Shariatian et al. synthetic stable-data benchmark."""

import argparse
import ast
import copy
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR]
for path_entry in [REPO_ROOT]:
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))

import gendynamics                                                                           # noqa

VENDOR_ROOT = Path(gendynamics.__file__).resolve().parent / "_vendor" / "DLPM"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from bem.datasets.Distributions import gen_sas                                                       # noqa
from dlpm.methods.GenerativeLevyProcess import GenerativeLevyProcess                                 # noqa
from dlpm.models.Model import MLPModel as VendorMLPModel                                             # noqa
from gendynamics.diffusion import DDPMV, DLPMEps                                                          # noqa
from gendynamics.thirdparty import DLPMEpsOrigin                                                          # noqa

####################################################################################################
# Globals
VENDOR_CONFIG = yaml.safe_load((VENDOR_ROOT / "dlpm" / "configs" / "2d_sas.yml").read_text())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FDTYPE = torch.float32
IDTYPE = torch.int32

ARTIFACT_ROOT = REPO_ROOT / "benchmarks" / "artifacts" / "shariatan_et_al"
TABLE_ROOT = REPO_ROOT / "benchmarks" / "tables"

N_SEEDS = 20
SEEDS = list(range(N_SEEDS))
MODEL_ALPHAS = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
METHODS = ["DDPMV", "DLPMEps", "DLPMEpsOrig"]
AUTHOR_METHOD = "DLPMAuthor"

TRAIN_SIZE = 32_000
EVAL_SIZE = 15_000
BATCH_SIZE = 1024
N_EPOCHS = 20
DIFFUSION_STEPS = 100
LEARNING_RATE = 5e-3
ALPHA_DATA = 1.7
DATA_SCALE = 0.1
XI = 0.95

METHOD_LABELS = {
    "DDPMV": "DDPM (our code)",
    "DLPMEps": "DLPM (our code)",
    "DLPMEpsOrig": "DLPM (author code wrapped)",
    AUTHOR_METHOD: "DLPM (author code)",
}
PAPER_VALUES = {
    1.5: (0.160, 0.128),
    1.6: (0.081, 0.078),
    1.7: (0.071, 0.028),
    1.8: (0.099, 0.044),
    1.9: (0.132, 0.101),
    2.0: (0.798, 0.601),
}
RESULT_COLUMNS = ["method", "model_alpha", "seed", "msle95", "loss_last", "n_steps"]


def sample_target_sas(n_samples: int, seed: int) -> torch.Tensor:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    x = DATA_SCALE * gen_sas(alpha=ALPHA_DATA, size=(n_samples, 2), device=DEVICE, isotropic=True)
    return x.to(device=DEVICE, dtype=FDTYPE)


def msle(x_real: torch.Tensor, x_gen: torch.Tensor, xi: float = XI, n_grid: int = 1000) -> float:
    x_real = x_real.detach().cpu().numpy()
    x_gen = x_gen.detach().cpu().numpy()
    probs = xi + (np.arange(1, n_grid + 1) - 0.5) / n_grid * (1 - xi)
    per_dim = []
    for dim in range(x_real.shape[1]):
        q_real = np.clip(np.quantile(np.abs(x_real[:, dim]), probs), 1e-12, None)
        q_gen = np.clip(np.quantile(np.abs(x_gen[:, dim]), probs), 1e-12, None)
        per_dim.append((1 - xi) / n_grid * np.sum((np.log(q_real) - np.log(q_gen)) ** 2))
    return float(np.mean(per_dim))


def build_vendor_params(model_alpha: float) -> dict:
    params = copy.deepcopy(VENDOR_CONFIG)
    params["device"] = str(DEVICE)
    params["method"] = "dlpm"
    params["data"].update(
        dataset="sas",
        data_alpha=float(ALPHA_DATA),
        std=float(DATA_SCALE),
        dim=2,
        nfeatures=2,
        nsamples=int(TRAIN_SIZE),
        isotropic=True,
        normalized=False,
    )
    params["dlpm"].update(
        alpha=float(model_alpha),
        reverse_steps=int(DIFFUSION_STEPS),
        isotropic=True,
        mean_predict="EPSILON",
        var_predict="FIXED",
        rescale_timesteps=True,
        scale="scale_preserving",
        input_scaling=False,
    )
    params["training"]["batch_size"] = int(BATCH_SIZE)
    params["training"]["num_workers"] = 0
    params["training"]["dlpm"].update(
        loss_monte_carlo="mean",
        loss_type="EPS_LOSS",
        lploss=2.0,
        monte_carlo_inner=1,
        monte_carlo_outer=1,
    )
    params["optim"].update(optimizer="adamw", lr=float(LEARNING_RATE))
    params["run"]["epochs"] = int(N_EPOCHS)
    return params


class Vendor2DAdapter(torch.nn.Module):
    def __init__(self, vendor_params: dict):
        super().__init__()
        self.core = VendorMLPModel(copy.deepcopy(vendor_params))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if isinstance(t, torch.Tensor) and t.ndim > 1:
            t = t.reshape(t.shape[0])
        y = self.core(x, t)
        if y.ndim == 3 and y.shape[1] == 1:
            y = y.squeeze(1)
        if y.ndim == 4 and y.shape[1:3] == (1, 1):
            y = y.squeeze(1).squeeze(1)
        return y


AUTHOR_SCRIPT = (
    "import sys, types\n"
    f"sys.path.insert(0, {str(VENDOR_ROOT)!r})\n"
    "neptune = types.ModuleType('neptune')\n"
    "class _DummySeries:\n"
    "    def append(self, *args, **kwargs):\n"
    "        pass\n"
    "    def extend(self, *args, **kwargs):\n"
    "        pass\n"
    "class _DummyRun(dict):\n"
    "    def __getitem__(self, key):\n"
    "        return self.setdefault(key, _DummySeries())\n"
    "    def __setitem__(self, key, value):\n"
    "        dict.__setitem__(self, key, value)\n"
    "    def exists(self, key):\n"
    "        return key in self\n"
    "    def stop(self):\n"
    "        pass\n"
    "def _init_run(*args, **kwargs):\n"
    "    return _DummyRun()\n"
    "neptune.init_run = _init_run\n"
    "neptune_utils = types.ModuleType('neptune.utils')\n"
    "neptune_utils.stringify_unsupported = lambda x: x\n"
    "sys.modules['neptune'] = neptune\n"
    "sys.modules['neptune.utils'] = neptune_utils\n"
    "private_code = types.ModuleType('dlpm.private_code')\n"
    "private_code.Neptune_Project = ''\n"
    "private_code.Neptune_API_key = ''\n"
    "sys.modules['dlpm.private_code'] = private_code\n"
    "import bem.datasets.Data as data_mod\n"
    "from bem.datasets.Distributions import gen_sas as _vendor_gen_sas\n"
    "def _compat_gen_sas(*args, **kwargs):\n"
    "    alpha = kwargs.pop('alpha', args[0] if args else None)\n"
    "    size = kwargs.pop('size', None)\n"
    "    n_samples = int(kwargs.pop('n_samples', kwargs.pop('n', 0) or 0))\n"
    "    device = kwargs.pop('device', None)\n"
    "    isotropic = kwargs.pop('isotropic', True)\n"
    "    if size is None:\n"
    "        size = (n_samples, 2)\n"
    "    return _vendor_gen_sas(alpha=alpha, size=size, device=device, isotropic=isotropic)\n"
    "data_mod.gen_sas = _compat_gen_sas\n"
    "from run import run_exp\n"
    f"run_exp({str(VENDOR_ROOT / 'dlpm' / 'configs')!r})\n"
)


####################################################################################################
# Main
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the standalone Shariatian et al. synthetic benchmark.")
    parser.add_argument("--output-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--table-root", type=Path, default=TABLE_ROOT)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--model-alphas", nargs="+", type=float, default=MODEL_ALPHAS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--skip-author", action="store_true", help="Skip the direct author-code baseline.")
    args = parser.parse_args()
    if args.n_seeds < 1:
        raise ValueError("--n-seeds must be >= 1.")
    output_root = args.output_root.expanduser()
    table_root = args.table_root.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.n_seeds))
    model_alphas = list(args.model_alphas)
    methods = list(args.methods)

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    print("[INFO] Reproducing experiments from Shariatian et al. (2025)")

    # base on our code
    rows = []
    total = len(methods) * len(model_alphas) * len(seeds)
    progress = tqdm(total=total, desc="[INFO] In-house models sweep")
    for method_name in methods:
        for model_alpha in model_alphas:
            for seed in seeds:
                x_train = sample_target_sas(TRAIN_SIZE, seed)
                x_eval = sample_target_sas(EVAL_SIZE, 100_000 + seed)
                params = build_vendor_params(model_alpha)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                ddpmv = DDPMV(net=Vendor2DAdapter(params), dim=2, n_steps=DIFFUSION_STEPS, fdtype=FDTYPE, idtype=IDTYPE, device=DEVICE)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                native = DLPMEps(
                    net=Vendor2DAdapter(params),
                    dim=2,
                    n_steps=DIFFUSION_STEPS,
                    alpha=model_alpha,
                    n_trial_A=1,
                    n_trial_G=1,
                    reduce_type="mean",
                    fdtype=FDTYPE,
                    idtype=IDTYPE,
                    device=DEVICE,
                )
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                wrapped = DLPMEpsOrigin(
                    net=Vendor2DAdapter(params),
                    dim=2,
                    n_steps=DIFFUSION_STEPS,
                    alpha=model_alpha,
                    loss_monte_carlo="mean",
                    monte_carlo_outer=1,
                    monte_carlo_inner=1,
                    lploss=2.0,
                    scale="scale_preserving",
                    fdtype=FDTYPE,
                    idtype=IDTYPE,
                    device=DEVICE,
                )
                model = {"DDPMV": ddpmv, "DLPMEps": native, "DLPMEpsOrig": wrapped}[method_name]
                dataset = TensorDataset(x_train.detach().cpu())
                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed)
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)
                optimizer = torch.optim.AdamW(model._net.parameters(), lr=LEARNING_RATE)

                loss_last = float("nan")
                model._net.train()
                for _ in range(N_EPOCHS):
                    for (xb,) in loader:
                        xb = xb.to(device=DEVICE, dtype=FDTYPE)
                        optimizer.zero_grad(set_to_none=True)
                        loss = model.loss(xb)
                        loss.backward()
                        optimizer.step()
                        loss_last = float(loss.detach().cpu())

                model._net.eval()
                with torch.no_grad():
                    x_gen = model.sample(int(x_eval.shape[0]))
                rows.append(
                    {
                        "method": method_name,
                        "model_alpha": float(model_alpha),
                        "seed": int(seed),
                        "msle95": msle(x_eval, x_gen),
                        "loss_last": loss_last,
                        "n_steps": N_EPOCHS * int(np.ceil(TRAIN_SIZE / BATCH_SIZE)),
                    }
                )
                progress.update(1)
    progress.close()
    our_df = pd.DataFrame(rows).sort_values(["method", "model_alpha", "seed"]).reset_index(drop=True)
    our_df.to_csv(output_root / "inhouse_results.csv", index=False)

    # direct author's code call
    rows = []
    if not args.skip_author:
        total = len(model_alphas) * len(seeds)
        progress = tqdm(total=total, desc="[INFO] Author models sweep")
        for model_alpha in model_alphas:
            for seed in seeds:
                cmd = [
                    sys.executable,
                    "-c",
                    AUTHOR_SCRIPT,
                    "--config",
                    "2d_sas",
                    "--name",
                    f"author_sas_alpha_{model_alpha:.1f}_seed_{seed}".replace(".", "p"),
                    "--method",
                    "dlpm",
                    "--set_seed",
                    str(seed),
                    "--alpha",
                    str(float(model_alpha)),
                    "--epochs",
                    str(N_EPOCHS),
                    "--train_reverse_steps",
                    str(DIFFUSION_STEPS),
                    "--reverse_steps",
                    str(DIFFUSION_STEPS),
                    "--data_std",
                    str(float(DATA_SCALE)),
                    "--nsamples",
                    str(TRAIN_SIZE),
                    "--dataset",
                    "sas",
                    "--lr",
                    str(float(LEARNING_RATE)),
                    "--no_ema_eval",
                ]

                with tempfile.TemporaryDirectory(prefix="dlpm_author_", dir="/tmp") as tmpdir:
                    tmpdir = Path(tmpdir)
                    mplconfig_dir = tmpdir / ".mplconfig"
                    mplconfig_dir.mkdir(parents=True, exist_ok=True)
                    (tmpdir / "transformers.py").write_text(
                        "import math\n"
                        "import torch\n"
                        "\n"
                        "def get_scheduler(name, optimizer, num_warmup_steps=0, num_training_steps=0):\n"
                        "    name = str(name).lower()\n"
                        "    if name == 'constant':\n"
                        "        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)\n"
                        "    if num_training_steps <= 0:\n"
                        "        raise ValueError('num_training_steps must be positive')\n"
                        "    def lr_lambda(step):\n"
                        "        if num_warmup_steps > 0 and step < num_warmup_steps:\n"
                        "            return float(step) / float(max(1, num_warmup_steps))\n"
                        "        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))\n"
                        "        progress = min(max(progress, 0.0), 1.0)\n"
                        "        if name == 'linear':\n"
                        "            return max(0.0, 1.0 - progress)\n"
                        "        if name == 'cosine':\n"
                        "            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))\n"
                        "        if name == 'cosine_with_restarts':\n"
                        "            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((progress * 2.0) % 1.0))))\n"
                        "        raise ValueError(f'Unsupported scheduler: {name}')\n"
                        "    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)\n"
                    )
                    (tmpdir / "prdc.py").write_text(
                        "def compute_prdc(*args, **kwargs):\n"
                        "    return {'precision': float('nan'), 'recall': float('nan'), 'density': float('nan'), 'coverage': float('nan')}\n"
                    )
                    (tmpdir / "pyemd.py").write_text(
                        "def emd(*args, **kwargs):\n"
                        "    return float('nan')\n"
                        "\n"
                        "def emd_samples(*args, **kwargs):\n"
                        "    return float('nan')\n"
                    )
                    env = os.environ.copy()
                    env["MPLCONFIGDIR"] = str(mplconfig_dir)
                    proc = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True, env=env)
                    if proc.returncode != 0:
                        raise RuntimeError(
                            "Author reference subprocess failed.\n"
                            f"Command: {cmd}\n"
                            f"stdout:\n{proc.stdout}\n"
                            f"stderr:\n{proc.stderr}"
                        )

                    model_path_rel = None
                    for line in reversed(proc.stdout.splitlines()):
                        line = line.strip()
                        if line.startswith("(") and line.endswith(")"):
                            maybe_paths = ast.literal_eval(line)
                            if isinstance(maybe_paths, tuple) and len(maybe_paths) == 3:
                                model_path_rel = Path(maybe_paths[0])
                                break
                    if model_path_rel is None:
                        raise RuntimeError("Could not parse saved-model paths from vendor stdout.")
                    checkpoint = torch.load(tmpdir / model_path_rel, map_location=DEVICE, weights_only=False)

                    author_model = VendorMLPModel(build_vendor_params(model_alpha)).to(DEVICE)
                    author_model.load_state_dict(checkpoint["model_parameters"])
                    author_model.eval()

                    sampler = GenerativeLevyProcess(
                        alpha=float(model_alpha),
                        device=DEVICE,
                        reverse_steps=DIFFUSION_STEPS,
                        rescale_timesteps=True,
                        isotropic=True,
                        scale="scale_preserving",
                    )
                    x_eval = sample_target_sas(EVAL_SIZE, 100_000 + seed)
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    with torch.no_grad():
                        x_gen = sampler.p_sample_loop(author_model, shape=(int(x_eval.shape[0]), 1, int(x_eval.shape[1])), progress=False)
                    if x_gen.ndim == 3 and x_gen.shape[1] == 1:
                        x_gen = x_gen.squeeze(1)

                loss_last = float("nan")
                for line in reversed(proc.stdout.splitlines()):
                    line = line.strip()
                    if line.startswith("epoch_loss "):
                        loss_last = float(line.split()[-1])
                        break

                rows.append(
                    {
                        "method": AUTHOR_METHOD,
                        "model_alpha": float(model_alpha),
                        "seed": int(seed),
                        "msle95": msle(x_eval, x_gen),
                        "loss_last": loss_last,
                        "n_steps": int(checkpoint["steps"]),
                    }
                )
                progress.update(1)
        progress.close()
    author_df = pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values(["method", "model_alpha", "seed"]).reset_index(drop=True)
    author_df.to_csv(output_root / "author_results.csv", index=False)

    combined_df = pd.concat([our_df, author_df], ignore_index=True)
    combined_df.to_csv(output_root / "results.csv", index=False)
    summary_df = (
        combined_df.groupby(["model_alpha", "method"], as_index=False)
        .agg(msle95_mean=("msle95", "mean"), msle95_std=("msle95", "std"))
        .sort_values(["model_alpha", "method"])
        .reset_index(drop=True)
    )

    row_labels = [METHOD_LABELS[method] for method in methods]
    if not args.skip_author:
        row_labels.append(METHOD_LABELS[AUTHOR_METHOD])
    row_labels.append("DLPM (paper results)")
    table_df = pd.DataFrame(
        "NA",
        index=row_labels,
        columns=[f"$\\alpha = {alpha:.1f}$" for alpha in model_alphas],
        dtype=object,
    )

    for _, row in summary_df.iterrows():
        label = METHOD_LABELS[row["method"]]
        col = f"$\\alpha = {float(row['model_alpha']):.1f}$"
        table_df.loc[label, col] = f"${row['msle95_mean']:.3f} \\pm {row['msle95_std']:.3f}$"

    for alpha, (mean, std) in PAPER_VALUES.items():
        if alpha not in model_alphas:
            continue
        table_df.loc["DLPM (paper results)", f"$\\alpha = {alpha:.1f}$"] = f"${mean:.3f} \\pm {std:.3f}$"

####################################################################################################
# Results table
    filepath = table_root / "shariatan_et_al_msle_table.tex"
    print(f"[INFO] Saving results under '{filepath}'")
    lines = [
        r"% Requires \usepackage{booktabs}",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Synthetic stable-data results from Shariatian et al. (2025). Lower $\mathrm{MSLE}_{0.95}$ is better.}",
        r"\label{tab:shariatan-et-al-msle}",
        rf"\begin{{tabular}}{{l{'c' * len(table_df.columns)}}}",
        r"\toprule",
        r"$\mathrm{MSLE}_{0.95}$ & " + " & ".join(table_df.columns) + r" \\",
        r"\midrule",
    ]
    for row_label, row in table_df.iterrows():
        values = [str(value).replace("NA", r"--") for value in row]
        lines.append(f"{row_label} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    filepath.write_text("\n".join(lines), encoding="utf-8")
