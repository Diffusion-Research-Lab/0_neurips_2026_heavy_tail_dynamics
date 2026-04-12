"""Training utilities for diffusion models."""

import inspect
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset
from .visitor import CoreMetricsVisitor, TrainVisitor


def _call_visitor_hook(visitor: TrainVisitor, hook_name: str, **kwargs) -> None:
    """Call a visitor hook while supporting legacy hook signatures."""
    hook = getattr(visitor, hook_name)
    params = inspect.signature(hook).parameters
    supported_kwargs = {name: value for name, value in kwargs.items() if name in params}
    hook(**supported_kwargs)


def _validate_train_inputs(
    generative_model: Any,
    target_data: torch.Tensor,
    source_data: Optional[torch.Tensor],
    use_adamw: bool,
    weight_decay: float,
) -> None:
    """Check that the model and tensors are compatible with the training loop."""
    if not hasattr(generative_model, "_net") or not isinstance(generative_model._net, torch.nn.Module):
        raise ValueError("generative_model must have a torch.nn.Module attribute `_net`.")
    if not hasattr(generative_model, "loss") or not callable(generative_model.loss):
        raise ValueError("generative_model must have a callable method `loss(x, z=...)`.")
    if not isinstance(target_data, torch.Tensor):
        raise TypeError(f"target_data must be a torch.Tensor, got {type(target_data)}")
    if source_data is not None and not isinstance(source_data, torch.Tensor):
        raise TypeError(f"source_data must be a torch.Tensor or None, got {type(source_data)}")
    if (not use_adamw) and float(weight_decay) != 0.0:
        raise ValueError("weight_decay is only supported with AdamW in this trainer.")


def _prepare_target_source(
    target_data: torch.Tensor,
    source_data: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.dtype]:
    """Detach, validate, and move the training tensors to CPU staging memory."""
    target = target_data.detach().contiguous().cpu()
    if target.size(0) == 0:
        raise ValueError("target_data is empty; at least one sample is required for training.")
    dtype = target.dtype
    dim = target.size(-1)

    source = None
    if source_data is not None:
        source = source_data.detach().contiguous().cpu()
        if source.size(0) == 0:
            raise ValueError("source_data is empty; at least one sample is required when provided.")
        if source.size(-1) != dim:
            raise ValueError(f"source_data dim {source.size(-1)} != target_data dim {dim}")
    return target, source, dtype


def _build_loader(
    target: torch.Tensor,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Build the shuffled dataloader used by the training loop."""
    return DataLoader(
        TensorDataset(target),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        persistent_workers=bool(int(num_workers) > 0),
    )


def _lr_mult(step: int, lr_schedule: str, warmup_steps: int, total_steps: int) -> float:
    """Return the scalar learning-rate multiplier for a given global step."""
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


def _build_train_config(
    batch_size: int,
    n_epochs: int,
    lr: float,
    lr_schedule: str,
    warmup_steps: int,
    weight_decay: float,
    use_adamw: bool,
    grad_clip_norm: Optional[float],
    dtype: torch.dtype,
    device: torch.device,
    num_workers: int,
) -> Dict[str, Any]:
    """Collect the effective training hyperparameters into a serializable dictionary."""
    return {
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
        "num_workers": int(num_workers),
    }


def _compute_grad_stats(net: torch.nn.Module) -> Tuple[Optional[float], Optional[float]]:
    """Compute simple variance and norm summaries for current parameter gradients."""
    grad_count = 0
    grad_sum = 0.0
    grad_sq_sum = 0.0

    for p in net.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        grad_count += int(g.numel())
        grad_sum += float(g.sum().item())
        grad_sq_sum += float((g * g).sum().item())

    if grad_count == 0:
        return None, None

    mean_g = grad_sum / float(grad_count)
    grad_var = max(0.0, (grad_sq_sum / float(grad_count)) - (mean_g * mean_g))
    grad_norm = math.sqrt(grad_sq_sum)
    return grad_var, grad_norm


def _save_ckpt(
    ckpt_path: Optional[Path],
    ckpt_keep_last: int,
    epoch_idx: int,
    global_step: int,
    last_loss: float,
    net: torch.nn.Module,
    opt,
    scheduler,
    train_config: Dict[str, Any],
) -> None:
    """Persist a training checkpoint and rotate older checkpoint files."""
    if ckpt_path is None:
        return

    payload = {
        "epoch": int(epoch_idx),
        "global_step": int(global_step),
        "loss": float(last_loss),
        "model_state": net.state_dict(),
        "opt_state": opt.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "train_config": train_config,
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


def train(
    generative_model,
    target_data: torch.Tensor,
    source_data: Optional[torch.Tensor] = None,
    batch_size: int = 512,
    n_epochs: int = 250,
    lr: float = 1e-4,
    device: torch.device = "cpu",
    num_workers: int = 0,
    use_adamw: bool = True,
    weight_decay: float = 0.0,
    grad_clip_norm: Optional[float] = None,
    lr_schedule: str = "cosine",  # {"cosine","linear","constant","none"}
    warmup_steps: int = 0,        # steps (not epochs)
    freq_logging: int = 10,
    ckpt_dir: Optional[str] = None,
    ckpt_freq_epochs: int = 10,
    ckpt_keep_last: int = 3,
    visitors: Optional[Sequence[TrainVisitor]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Train a native genkit generative model and return diagnostics."""
    logger = logging.getLogger(__name__)
    _validate_train_inputs(
        generative_model=generative_model,
        target_data=target_data,
        source_data=source_data,
        use_adamw=use_adamw,
        weight_decay=weight_decay,
    )
    device = torch.device(device)
    target, source, dtype = _prepare_target_source(target_data=target_data, source_data=source_data)

    net = generative_model._net.to(device=device, dtype=dtype)
    net.train()

    pin = device.type == "cuda"
    loader = _build_loader(target=target, batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
    opt_cls = torch.optim.AdamW if use_adamw else torch.optim.Adam
    opt = opt_cls(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    steps_per_epoch = max(len(loader), 1)
    total_steps = int(n_epochs) * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt,
        lr_lambda=lambda step: _lr_mult(
            step=step,
            lr_schedule=lr_schedule,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        ),
    )

    ckpt_path = None
    if ckpt_dir is not None:
        ckpt_path = Path(ckpt_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)

    if visitors is None:
        visitors = [CoreMetricsVisitor()]
    visitors = list(visitors)
    train_config = _build_train_config(
        batch_size=batch_size,
        n_epochs=n_epochs,
        lr=lr,
        lr_schedule=lr_schedule,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        use_adamw=use_adamw,
        grad_clip_norm=grad_clip_norm,
        dtype=dtype,
        device=device,
        num_workers=num_workers,
    )

    logger.info(
        f"train | epochs={n_epochs} bs={batch_size} lr={float(lr):g} "
        f"schedule={lr_schedule} warmup_steps={warmup_steps} "
        f"opt={'AdamW' if use_adamw else 'Adam'} wd={float(weight_decay):g} "
        f"device={device} dtype={dtype} ckpt_dir={ckpt_dir}"
    )

    for v in visitors:
        _call_visitor_hook(
            v,
            "on_train_start",
            target=target,
            source=source,
            config=train_config,
            generative_model=generative_model,
            net=net,
        )

    global_step = 0
    last_epoch_loss = float("nan")

    for epoch in range(int(n_epochs)):
        for v in visitors:
            _call_visitor_hook(v, "on_epoch_start", generative_model=generative_model, net=net)
        epoch_losses: List[float] = []
        for (x_cpu,) in loader:
            x = x_cpu.to(device=device, dtype=dtype, non_blocking=pin)
            if source is None:
                z = None
            else:
                idx = torch.randint(0, source.size(0), (x.size(0),), device="cpu")
                z = source.index_select(0, idx).to(device=device, dtype=dtype, non_blocking=pin)

            opt.zero_grad(set_to_none=True)
            loss = generative_model.loss(x, z=z)
            if loss.ndim != 0:
                raise ValueError(f"generative_model.loss must return a scalar, got shape {tuple(loss.shape)}")
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(grad_clip_norm))
            grad_var, grad_norm_val = _compute_grad_stats(net=net)
            opt.step()
            scheduler.step()

            loss_f = float(loss.detach().item())
            epoch_losses.append(loss_f)
            global_step += 1

            for v in visitors:
                _call_visitor_hook(
                    v,
                    "on_batch_end",
                    loss=loss_f,
                    grad_var=grad_var,
                    grad_norm=grad_norm_val,
                    generative_model=generative_model,
                    net=net,
                )

        last_epoch_loss = float(sum(epoch_losses) / float(len(epoch_losses))) if epoch_losses else float("nan")
        for v in visitors:
            _call_visitor_hook(v, "on_epoch_end", generative_model=generative_model, net=net)

        if (epoch + 1) % int(freq_logging) == 0:
            details = " | ".join([s for s in (v.format_epoch_log() for v in visitors) if s])
            if details:
                logger.info(f"epoch {epoch + 1:3d}/{int(n_epochs):3d} | {details} | lr {opt.param_groups[0]['lr']:.3e}")
            else:
                logger.info(f"epoch {epoch + 1:3d}/{int(n_epochs):3d} | lr {opt.param_groups[0]['lr']:.3e}")

        if ckpt_path is not None and (epoch + 1) % int(ckpt_freq_epochs) == 0:
            _save_ckpt(
                ckpt_path=ckpt_path,
                ckpt_keep_last=ckpt_keep_last,
                epoch_idx=epoch + 1,
                global_step=global_step,
                last_loss=last_epoch_loss,
                net=net,
                opt=opt,
                scheduler=scheduler,
                train_config=train_config,
            )

    if ckpt_path is not None:
        _save_ckpt(
            ckpt_path=ckpt_path,
            ckpt_keep_last=ckpt_keep_last,
            epoch_idx=n_epochs,
            global_step=global_step,
            last_loss=last_epoch_loss,
            net=net,
            opt=opt,
            scheduler=scheduler,
            train_config=train_config,
        )

    for v in visitors:
        _call_visitor_hook(v, "on_train_end", generative_model=generative_model, net=net)

    logger.info("train | done")
    diagnostics = {
        "train_config": train_config,
        "visitors": {v.name: v.get_records() for v in visitors},
    }
    return generative_model, diagnostics
