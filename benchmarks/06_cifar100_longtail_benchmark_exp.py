"""CIFAR100 long-tail benchmark simulation in frozen feature space."""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from cauda.flow import AlphaStableFlowLinear, GaussianFlowLinear
from cauda.model import FlowNet
from cauda.training import train
from labkit.config import load_config
from results_utils import create_run_dir, write_artifacts
from tqdm import tqdm
SCRIPT_DIR = Path(__file__).resolve().parent

try:
    from torchvision import datasets, models, transforms
except Exception as exc:  # pragma: no cover
    raise RuntimeError("torchvision is required for CIFAR100 benchmark.") from exc


def _build_transforms(use_imagenet_weights: bool):
    if use_imagenet_weights:
        weights = models.ResNet18_Weights.DEFAULT
        return weights.transforms(), weights
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return tfm, None


def _longtail_indices(
    targets: List[int],
    n_classes: int,
    head_fraction: float,
    ratio: int,
    max_per_head: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    g = torch.Generator().manual_seed(int(seed))
    cls_to_idx: Dict[int, List[int]] = {c: [] for c in range(n_classes)}
    for i, y in enumerate(targets):
        cls_to_idx[int(y)].append(i)

    head_cut = int(round(n_classes * head_fraction))
    tail_classes = list(range(head_cut, n_classes))

    chosen: List[int] = []
    for c in range(n_classes):
        idx = cls_to_idx[c]
        if len(idx) == 0:
            continue
        perm = torch.randperm(len(idx), generator=g).tolist()
        n_keep = min(max_per_head, len(idx))
        if c in tail_classes:
            n_keep = max(1, n_keep // int(ratio))
        chosen.extend([idx[p] for p in perm[:n_keep]])

    return chosen, tail_classes


@torch.no_grad()
def _extract_features(
    backbone: nn.Module,
    dataset,
    indices: List[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
    )

    feats, labels = [], []
    for x, y in tqdm(loader, desc="bench3/feature_batches", unit="batch"):
        x = x.to(device=device)
        h = backbone(x)
        feats.append(h.detach().cpu())
        labels.append(y.detach().cpu())

    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def _nearest_centroid_predict(samples: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(samples, centroids)
    return d.argmin(dim=1)


def _metrics(
    y_true: torch.Tensor,
    y_gen: torch.Tensor,
    minority: List[int],
    n_classes: int,
    k: int,
) -> Dict[str, float]:
    eps = 1e-12
    cnt_true = torch.bincount(y_true, minlength=n_classes).float()
    cnt_gen = torch.bincount(y_gen, minlength=n_classes).float()

    idx = torch.tensor(minority, dtype=torch.long)

    minority_cov_at_k = float((cnt_gen[idx] >= float(k)).float().mean().item())

    p_true = cnt_true / cnt_true.sum().clamp_min(eps)
    p_gen = cnt_gen / cnt_gen.sum().clamp_min(eps)

    minority_mass_ratio = float((p_gen[idx].sum() / p_true[idx].sum().clamp_min(eps)).item())
    minority_recall = float(
        (torch.minimum(cnt_gen[idx], cnt_true[idx]).sum() / cnt_true[idx].sum().clamp_min(eps)).item()
    )

    kl = float((p_true * (p_true.add(eps).log() - p_gen.add(eps).log())).sum().item())

    return {
        "minority_coverage_at_k": minority_cov_at_k,
        "minority_mass_ratio": minority_mass_ratio,
        "minority_recall": minority_recall,
        "class_hist_kl_true_to_gen": kl,
    }


if __name__ == "__main__":
    t0 = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config") / "bench_3_config.yaml")
    parser.add_argument("--out-root", type=Path, default=Path("_results"))
    args = parser.parse_args()

    cfg = load_config(args.config).set_up()
    run_dir = create_run_dir(args.out_root, "bench_3")
    print(f"[INFO] bench_3 config loaded: {args.config}")
    print(f"[INFO] bench_3 run directory: {run_dir}")

    data_root = Path(cfg.data_root)
    if not data_root.is_absolute():
        data_root = SCRIPT_DIR / data_root
    data_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] bench_3 data root: {data_root}")

    weights_root = SCRIPT_DIR / "_weights"
    weights_root.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(weights_root))
    print(f"[INFO] bench_3 weights root: {weights_root}")

    tfm, weights = _build_transforms(bool(cfg.use_imagenet_weights))
    print("[INFO] bench_3 loading CIFAR100")
    ds = datasets.CIFAR100(root=str(data_root), train=True, download=bool(cfg.download), transform=tfm)

    chosen_idx, minority_classes = _longtail_indices(
        targets=ds.targets,
        n_classes=100,
        head_fraction=float(cfg.head_fraction),
        ratio=int(cfg.longtail_ratio),
        max_per_head=int(cfg.max_per_head),
        seed=int(cfg.seed),
    )

    if getattr(cfg, "max_train_samples", None) is not None:
        chosen_idx = chosen_idx[: int(cfg.max_train_samples)]
    print(f"[INFO] bench_3 selected samples: {len(chosen_idx)}")

    device = torch.device(cfg.device)

    print("[INFO] bench_3 loading ResNet18 backbone")
    backbone = models.resnet18(weights=weights if weights is not None else None)
    backbone.fc = nn.Identity()
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    print("[INFO] bench_3 extracting frozen features")
    x_feat, y_label = _extract_features(
        backbone=backbone,
        dataset=ds,
        indices=chosen_idx,
        batch_size=int(cfg.feature_batch_size),
        num_workers=int(cfg.feature_num_workers),
        device=device,
    )

    x_feat = x_feat.to(dtype=cfg.dtype)
    feat_dim = x_feat.size(1)
    print(f"[INFO] bench_3 feature tensor: n={x_feat.size(0)} dim={feat_dim}")
    centroids = torch.stack([x_feat[y_label == c].mean(dim=0) for c in range(100)], dim=0)

    n_gen = int(getattr(cfg, "n_gen_samples", x_feat.size(0)))

    models_to_run = [
        ("GaussianFlowLinear", GaussianFlowLinear, {}),
        ("AlphaStableFlowLinear", AlphaStableFlowLinear, {"alpha": float(cfg.alpha)}),
    ]

    results = {}
    for name, cls, extra in tqdm(models_to_run, desc="bench3/models", unit="model"):
        t_model = time.perf_counter()
        print(f"[INFO] bench_3 training model: {name}")

        net = FlowNet(
            dim=feat_dim,
            width=int(cfg.net_width),
            depth=int(cfg.net_depth),
            tdim=int(cfg.net_tdim),
            dropout=float(cfg.net_dropout),
        ).to(device=cfg.device, dtype=cfg.dtype)

        generator = cls(
            net=net,
            dim=feat_dim,
            n_steps=int(cfg.n_steps),
            dtype=cfg.dtype,
            device=cfg.device,
            **extra,
        )

        train(
            generator,
            target_data=x_feat.clone(),
            batch_size=int(cfg.batch_size),
            n_epochs=int(cfg.n_epochs),
            lr=float(cfg.lr),
            use_adamw=bool(cfg.use_adamw),
            grad_clip_norm=getattr(cfg, "grad_clip_norm", None),
            num_workers=int(cfg.num_workers),
            device=cfg.device,
            lr_schedule=str(cfg.lr_schedule),
            warmup_steps=int(cfg.warmup_steps),
            weight_decay=float(cfg.weight_decay),
            freq_logging=int(cfg.freq_logging),
        )

        x_gen = generator.sample(n_samples=n_gen).to(dtype=x_feat.dtype, device=x_feat.device)
        y_gen = _nearest_centroid_predict(x_gen, centroids.to(x_gen))

        m = _metrics(
            y_true=y_label,
            y_gen=y_gen.cpu(),
            minority=minority_classes,
            n_classes=100,
            k=int(cfg.minority_k),
        )
        m["runtime_sec"] = float(time.perf_counter() - t_model)
        results[name] = m
        print(f"[INFO] bench_3 done model: {name} ({m['runtime_sec']:.2f}s)")

    payload = {
        "benchmark": "bench_3",
        "config_path": str(args.config),
        "n_train_samples": int(x_feat.size(0)),
        "feature_dim": int(feat_dim),
        "minority_classes": minority_classes,
        "results": results,
    }

    runtime = time.perf_counter() - t0
    run_txt = (
        f"benchmark: bench_3\n"
        f"config: {args.config}\n"
        f"run_dir: {run_dir}\n"
        f"n_train_samples: {x_feat.size(0)}\n"
        f"runtime_sec: {runtime:.2f}\n"
    )

    write_artifacts(run_dir, cfg.as_dict(), payload, run_txt)
    print(f"[INFO] Saved run artifacts in {run_dir}")
