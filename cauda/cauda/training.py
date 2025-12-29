""""Training utilities for diffusion models."""

# Authors: Hamza Cherkaoui

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset


def train(
    generative_model,
    target_data: torch.Tensor,
    source_data: Optional[torch.Tensor] = None,
    batch_size: int = 512,
    n_epochs: int = 250,
    lr: float = 1e-4,
    device: torch.device = "cpu",
    num_workers: int = 2,
    use_adamw: bool = True,
    weight_decay: float = 0.0,
    grad_clip_norm: Optional[float] = None,
    lr_schedule: str = "cosine",  # {"cosine","linear","constant","none"}
    warmup_steps: int = 0,        # steps (not epochs)
    freq_logging: int = 10,
    ckpt_dir: Optional[str] = None,
    ckpt_freq_epochs: int = 10,
    ckpt_keep_last: int = 3,
) -> Tuple[Any, Dict[str, List[float]]]:
    logger = logging.getLogger(__name__)

    if not hasattr(generative_model, "_net") or not isinstance(generative_model._net, torch.nn.Module):
        raise ValueError("generative_model must have a torch.nn.Module attribute `_net`.")
    if not hasattr(generative_model, "loss") or not callable(generative_model.loss):
        raise ValueError("generative_model must have a callable method `loss(x, z=...)`.")
    if not isinstance(target_data, torch.Tensor):
        raise TypeError(f"target_data must be a torch.Tensor, got {type(target_data)}")
    if source_data is not None and not isinstance(source_data, torch.Tensor):
        raise TypeError(f"source_data must be a torch.Tensor or None, got {type(source_data)}")

    device = torch.device(device)
    target = target_data.detach().contiguous().cpu()
    dtype = target.dtype
    dim = target.size(-1)

    net = generative_model._net.to(device=device, dtype=dtype)
    net.train()

    source = None
    if source_data is not None:
        source = source_data.detach().contiguous().cpu()
        if source.size(-1) != dim:
            raise ValueError(f"source_data dim {source.size(-1)} != target_data dim {dim}")

    pin = device.type == "cuda"

    loader = DataLoader(
        TensorDataset(target),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(num_workers),
        pin_memory=pin,
        persistent_workers=bool(int(num_workers) > 0),
    )

    if (not use_adamw) and float(weight_decay) != 0.0:
        raise ValueError("weight_decay is only supported with AdamW in this trainer.")

    opt_cls = torch.optim.AdamW if use_adamw else torch.optim.Adam
    opt = opt_cls(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    steps_per_epoch = max(len(loader), 1)
    total_steps = int(n_epochs) * steps_per_epoch

    def _lr_mult(step: int) -> float:
        if lr_schedule in (None, "none"):
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        denom = max(1, total_steps - max(warmup_steps, 0))
        p = float(step - max(warmup_steps, 0)) / float(denom)
        p = min(max(p, 0.0), 1.0)
        if lr_schedule == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * p))
        if lr_schedule == "linear":
            return 1.0 - p
        if lr_schedule == "constant":
            return 1.0
        raise ValueError(f"Unknown lr_schedule='{lr_schedule}'")

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_mult)

    training_loss: List[float] = []
    lr_hist: List[float] = []

    ckpt_path = None
    if ckpt_dir is not None:
        ckpt_path = Path(ckpt_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)

    def _save_ckpt(epoch_idx: int, global_step: int, last_loss: float) -> None:
        if ckpt_path is None:
            return
        payload = {
            "epoch": int(epoch_idx),
            "global_step": int(global_step),
            "loss": float(last_loss),
            "model_state": net.state_dict(),
            "opt_state": opt.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "meta": {
                "batch_size": int(batch_size),
                "n_epochs": int(n_epochs),
                "lr": float(lr),
                "lr_schedule": str(lr_schedule),
                "warmup_steps": int(warmup_steps),
                "weight_decay": float(weight_decay),
                "use_adamw": bool(use_adamw),
                "grad_clip_norm": None if grad_clip_norm is None else float(grad_clip_norm),
                "dtype": str(dtype),
                "device": str(device),
            },
        }
        fname = ckpt_path / f"ckpt_epoch_{epoch_idx:04d}.pt"
        torch.save(payload, fname)

        last = ckpt_path / "ckpt_last.pt"
        try:
            if last.exists() or last.is_symlink():
                last.unlink()
            last.symlink_to(fname.name)
        except Exception:
            torch.save(payload, last)

        if ckpt_keep_last > 0:
            ckpts = sorted(ckpt_path.glob("ckpt_epoch_*.pt"))
            excess = len(ckpts) - int(ckpt_keep_last)
            for p in ckpts[: max(excess, 0)]:
                try:
                    p.unlink()
                except Exception:
                    pass

    logger.info(
        f"train | epochs={n_epochs} bs={batch_size} lr={float(lr):g} "
        f"schedule={lr_schedule} warmup_steps={warmup_steps} "
        f"opt={'AdamW' if use_adamw else 'Adam'} wd={float(weight_decay):g} "
        f"device={device} dtype={dtype} ckpt_dir={ckpt_dir}"
    )

    global_step = 0
    for epoch in range(int(n_epochs)):
        losses_epoch: List[float] = []

        for (x_cpu,) in loader:
            x = x_cpu.to(device=device, dtype=dtype, non_blocking=pin)

            z = None
            if source is not None:
                idx = torch.randint(0, source.size(0), (x.size(0),), device="cpu")
                z = source.index_select(0, idx).to(device=device, dtype=dtype, non_blocking=pin)

            opt.zero_grad(set_to_none=True)

            loss = generative_model.loss(x, z=z)
            if loss.ndim != 0:
                raise ValueError(f"generative_model.loss must return a scalar, got shape {tuple(loss.shape)}")

            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(grad_clip_norm))
            opt.step()
            scheduler.step()

            losses_epoch.append(float(loss.detach().item()))
            lr_hist.append(float(opt.param_groups[0]["lr"]))
            global_step += 1

        m = sum(losses_epoch) / max(len(losses_epoch), 1)
        training_loss.append(m)

        if (epoch + 1) % int(freq_logging) == 0:
            logger.info(f"epoch {epoch + 1:3d}/{n_epochs:3d} | loss {m:.6f} | lr {opt.param_groups[0]['lr']:.3e}")

        if ckpt_path is not None and (epoch + 1) % int(ckpt_freq_epochs) == 0:
            _save_ckpt(epoch_idx=epoch + 1, global_step=global_step, last_loss=m)

    if ckpt_path is not None:
        last_loss = training_loss[-1] if training_loss else float("nan")
        _save_ckpt(epoch_idx=n_epochs, global_step=global_step, last_loss=last_loss)

    logger.info("train | done")
    return generative_model, {"training_loss": training_loss, "lr": lr_hist}
