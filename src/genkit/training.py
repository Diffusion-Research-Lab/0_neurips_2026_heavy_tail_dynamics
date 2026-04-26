"""Training utilities for generative models."""

import inspect
import logging
import math
import warnings
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset
from .visitor import CoreMetricsVisitor, TrainVisitor


def _build_hook_param_cache(
    visitors: list[TrainVisitor], hook_names: list[str]
) -> dict[tuple[int, str], frozenset]:
    """Pre-compute accepted parameter sets for every (visitor, hook) pair."""
    return {
        (id(v), name): frozenset(inspect.signature(getattr(v, name)).parameters)
        for v in visitors
        for name in hook_names
    }


def _call_visitor_hook(
    visitor: TrainVisitor,
    hook_name: str,
    param_cache: dict[tuple[int, str], frozenset],
    **kwargs,
) -> None:
    """Call a visitor hook, filtering kwargs through the pre-built param cache."""
    params = param_cache[(id(visitor), hook_name)]
    getattr(visitor, hook_name)(**{k: v for k, v in kwargs.items() if k in params})


def _validate_train_inputs(generative_model, target_data, source_data, use_adamw, weight_decay) -> None:
    if not hasattr(generative_model, "_net") or not isinstance(generative_model._net, torch.nn.Module):
        raise ValueError("generative_model must have a torch.nn.Module attribute `_net`.")
    if not hasattr(generative_model, "loss") or not callable(generative_model.loss):
        raise ValueError("generative_model must have a callable method `loss(x, z=...)`.")
    if not isinstance(target_data, torch.Tensor):
        raise TypeError(f"target_data must be a torch.Tensor, got {type(target_data)}")
    if source_data is not None and not isinstance(source_data, torch.Tensor):
        raise TypeError(f"source_data must be a torch.Tensor or None, got {type(source_data)}")
    if not use_adamw and float(weight_decay) != 0.0:
        raise ValueError("weight_decay is only supported with AdamW in this trainer.")


def _prepare_target_source(
    target_data: torch.Tensor,
    source_data: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.dtype]:
    target = target_data.detach().contiguous().cpu()
    if target.size(0) == 0:
        raise ValueError("target_data is empty; at least one sample is required for training.")
    if source_data is not None:
        source = source_data.detach().contiguous().cpu()
        if source.size(0) == 0:
            raise ValueError("source_data is empty; at least one sample is required when provided.")
        if source.size(-1) != target.size(-1):
            raise ValueError(f"source_data dim {source.size(-1)} != target_data dim {target.size(-1)}")
    else:
        source = None
    return target, source, target.dtype


class _TensorBatchLoader:
    """Fast in-memory loader: one randperm per epoch, one slice per batch."""

    def __init__(self, target: torch.Tensor, batch_size: int) -> None:
        self._target = target
        self._batch_size = int(batch_size)

    def __len__(self) -> int:
        return (len(self._target) + self._batch_size - 1) // self._batch_size

    def __iter__(self):
        perm = torch.randperm(len(self._target))
        for start in range(0, len(self._target), self._batch_size):
            yield self._target[perm[start: start + self._batch_size]]


def _build_loader(
    target: torch.Tensor | Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool | None,
    prefetch_factor: int | None,
) -> _TensorBatchLoader | DataLoader:
    """Return a fast in-memory loader for tensors, or a DataLoader for Dataset."""
    if isinstance(target, torch.Tensor):
        if num_workers > 0:
            warnings.warn(
                "num_workers > 0 is ignored for in-memory tensor targets; "
                "set num_workers=0 to suppress this warning.",
                stacklevel=3,
            )
        return _TensorBatchLoader(target, batch_size)

    n_workers = int(num_workers)
    return DataLoader(
        target,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=n_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if persistent_workers is not None else n_workers > 0,
        prefetch_factor=prefetch_factor if n_workers > 0 else None,
    )


def _build_scheduler(
    opt: torch.optim.Optimizer,
    lr_schedule: str,
    warmup_steps: int,
    total_steps: int,
    cosine_eta_min_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build a native PyTorch LR scheduler for the requested schedule."""
    main_steps = max(1, total_steps - warmup_steps)
    if lr_schedule in (None, "none", "constant"):
        main = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=main_steps)
    elif lr_schedule == "cosine":
        if not 0.0 <= cosine_eta_min_ratio < 1.0:
            raise ValueError("cosine_eta_min_ratio must satisfy 0.0 <= cosine_eta_min_ratio < 1.0.")
        eta_min = float(opt.param_groups[0]["lr"]) * float(cosine_eta_min_ratio)
        main = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=main_steps, eta_min=eta_min)
    elif lr_schedule == "linear":
        main = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=main_steps)
    else:
        raise ValueError(f"Unknown lr_schedule='{lr_schedule}'")
    if warmup_steps <= 0:
        return main
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0 / warmup_steps, end_factor=1.0, total_iters=warmup_steps,
    )
    return torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, main], milestones=[warmup_steps])


def _compute_grad_stats(net: torch.nn.Module) -> tuple[float | None, float | None]:
    """Compute gradient variance and norm across all parameters."""
    count, g_sum, g_sq_sum = 0, 0.0, 0.0
    for p in net.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        count += g.numel()
        g_sum += float(g.sum())
        g_sq_sum += float(g.pow(2).sum())
    if count == 0:
        return None, None
    mean = g_sum / count
    return max(0.0, g_sq_sum / count - mean ** 2), math.sqrt(g_sq_sum)


def _loss_stats_or_nan(losses: list[torch.Tensor]) -> tuple[float, float]:
    """Return epoch loss mean and std from accumulated detached scalars."""
    if not losses:
        return float("nan"), float("nan")
    t = torch.stack(losses)
    return float(t.mean()), float(t.std(unbiased=False))


def _save_ckpt(ckpt_path, ckpt_keep_last, epoch_idx, global_step, last_loss, net, opt, scheduler, train_config) -> None:
    """Persist a checkpoint and rotate old ones."""
    if ckpt_path is None:
        return
    fname = ckpt_path / f"ckpt_epoch_{epoch_idx:04d}.pt"
    torch.save({
        "epoch": epoch_idx, "global_step": global_step, "loss": float(last_loss),
        "model_state": net.state_dict(), "opt_state": opt.state_dict(),
        "scheduler_state": scheduler.state_dict(), "train_config": train_config,
    }, fname)
    last = ckpt_path / "ckpt_last.pt"
    try:
        if last.exists() or last.is_symlink():
            last.unlink()
        last.symlink_to(fname.name)
    except Exception:
        torch.save(torch.load(fname), last)
    if ckpt_keep_last > 0:
        ckpts = sorted(ckpt_path.glob("ckpt_epoch_*.pt"))
        for p in ckpts[: max(len(ckpts) - ckpt_keep_last, 0)]:
            try:
                p.unlink()
            except Exception:
                pass


def train(
    generative_model,
    target_data: torch.Tensor,
    source_data: torch.Tensor | None = None,
    batch_size: int = 512,
    n_epochs: int = 250,
    lr: float = 1e-4,
    device: torch.device = "cpu",
    num_workers: int = 0,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = 2,
    use_adamw: bool = True,
    weight_decay: float = 0.0,
    grad_clip_norm: float | None = None,
    lr_schedule: str = "cosine",
    warmup_steps: int = 0,
    cosine_eta_min_ratio: float = 0.0,
    freq_logging: int = 10,
    ckpt_dir: str | None = None,
    ckpt_freq_epochs: int = 10,
    ckpt_keep_last: int = 3,
    visitors: list[TrainVisitor] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Train a native genkit generative model and return diagnostics."""
    logger = logging.getLogger(__name__)
    batch_size = int(batch_size)
    n_epochs = int(n_epochs)
    lr = float(lr)
    num_workers = int(num_workers)
    weight_decay = float(weight_decay)
    grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
    warmup_steps = int(warmup_steps)
    cosine_eta_min_ratio = float(cosine_eta_min_ratio)
    freq_logging = int(freq_logging)
    ckpt_freq_epochs = int(ckpt_freq_epochs)
    ckpt_keep_last = int(ckpt_keep_last)
    _validate_train_inputs(generative_model, target_data, source_data, use_adamw, weight_decay)

    device = torch.device(device)
    target, source, dtype = _prepare_target_source(target_data, source_data)
    net = generative_model._net.to(device=device, dtype=dtype)
    net.train()

    pin = device.type == "cuda"
    loader = _build_loader(target, batch_size, num_workers, pin, persistent_workers, prefetch_factor)
    opt_cls = torch.optim.AdamW if use_adamw else torch.optim.Adam
    opt = opt_cls(net.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(len(loader), 1)
    scheduler = _build_scheduler(
        opt,
        lr_schedule,
        warmup_steps,
        n_epochs * steps_per_epoch,
        cosine_eta_min_ratio=cosine_eta_min_ratio,
    )

    ckpt_path = Path(ckpt_dir) if ckpt_dir is not None else None
    if ckpt_path is not None:
        ckpt_path.mkdir(parents=True, exist_ok=True)

    visitors = list(visitors or [CoreMetricsVisitor()])
    param_cache = _build_hook_param_cache(visitors, ["on_train_start", "on_epoch_end"])
    train_config = {
        "batch_size": batch_size, "n_epochs": n_epochs, "lr": lr,
        "lr_schedule": lr_schedule, "warmup_steps": warmup_steps,
        "cosine_eta_min_ratio": cosine_eta_min_ratio,
        "weight_decay": weight_decay, "use_adamw": use_adamw,
        "grad_clip_norm": grad_clip_norm, "dtype": str(dtype), "device": str(device),
        "num_workers": num_workers, "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
    }

    def _fire(hook_name, **kw):
        for v in visitors:
            _call_visitor_hook(v, hook_name, param_cache, **kw)

    logger.info(
        f"train | epochs={n_epochs} bs={batch_size} lr={lr:g} schedule={lr_schedule} "
        f"warmup_steps={warmup_steps} cosine_eta_min_ratio={cosine_eta_min_ratio:g} "
        f"opt={'AdamW' if use_adamw else 'Adam'} "
        f"wd={weight_decay:g} device={device} dtype={dtype} ckpt_dir={ckpt_dir}"
    )
    _fire("on_train_start", target=target, source=source, config=train_config,
          generative_model=generative_model, net=net)

    last_epoch_loss = float("nan")
    for epoch in range(n_epochs):
        epoch_losses: list[torch.Tensor] = []
        for x in loader:
            x = x.to(device=device, dtype=dtype, non_blocking=pin)
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
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_norm)
            opt.step()
            scheduler.step()
            epoch_losses.append(loss.detach())

        last_epoch_loss, last_epoch_loss_std = _loss_stats_or_nan(epoch_losses)
        grad_var, grad_norm = _compute_grad_stats(net)
        _fire("on_epoch_end", loss=last_epoch_loss, loss_std=last_epoch_loss_std,
              grad_var=grad_var, grad_norm=grad_norm, generative_model=generative_model, net=net)

        if (epoch + 1) % freq_logging == 0:
            details = " | ".join(s for s in (v.format_epoch_log() for v in visitors) if s)
            lr_now = opt.param_groups[0]["lr"]
            if details:
                logger.info(f"epoch {epoch + 1:3d}/{n_epochs:3d} | {details} | lr {lr_now:.3e}")
            else:
                logger.info(f"epoch {epoch + 1:3d}/{n_epochs:3d} | lr {lr_now:.3e}")

        if ckpt_path is not None and (epoch + 1) % ckpt_freq_epochs == 0:
            _save_ckpt(ckpt_path, ckpt_keep_last, epoch + 1, (epoch + 1) * steps_per_epoch,
                       last_epoch_loss, net, opt, scheduler, train_config)

    if ckpt_path is not None:
        _save_ckpt(ckpt_path, ckpt_keep_last, n_epochs, n_epochs * steps_per_epoch,
                   last_epoch_loss, net, opt, scheduler, train_config)

    logger.info("train | done")
    return generative_model, {"train_config": train_config, "visitors": {v.name: v.get_records() for v in visitors}}
