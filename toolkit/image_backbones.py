#!/usr/bin/env python
"""Compare image backbone cost for the UNet and unconditional transformer.

This benchmark uses synthetic image-like tensors and does not use labels, text
conditions, cross-attention, classifier-free guidance, or dataset identifiers.
Runtime and memory are hardware- and configuration-dependent; these numbers do
not establish generative quality or state-of-the-art performance.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from gendynamics.nn import TransformerModel, UNetModel  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    channels: int
    height: int
    width: int


def parse_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype={name!r}.")


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024**2


def autocast_context(device: torch.device, dtype: torch.dtype):
    enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def make_model(name: str, case: Case, n_steps: int, patch_size: int) -> torch.nn.Module:
    if name == "Transformer":
        return TransformerModel(
            sample_size=(case.height, case.width),
            n_steps=n_steps,
            in_channels=case.channels,
            out_channels=case.channels,
            patch_size=patch_size,
            num_layers=4,
            num_attention_heads=4,
            attention_head_dim=64,
            dropout=0.0,
            timestep_mode="continuous",
            cross_attention_dim=None,
            caption_channels=None,
            use_additional_conditions=False,
        )
    if name == "UNet":
        return UNetModel(
            sample_size=(case.height, case.width),
            n_steps=n_steps,
            in_channels=case.channels,
            out_channels=case.channels,
            model_channels=64,
            num_res_blocks=2,
            channel_mult=(1, 2, 4),
            attention_resolutions=(4,),
            dropout=0.0,
            num_heads=4,
            norm_num_groups=8,
            use_scale_shift_norm=True,
        )
    raise ValueError(f"Unknown model name {name!r}.")


def finite_grads(model: torch.nn.Module) -> bool:
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    return bool(grads) and all(torch.isfinite(grad).all().item() for grad in grads)


def measure_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    target: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    model.train()
    total_params, trainable_params = count_params(model)

    with torch.no_grad(), autocast_context(device, dtype):
        y = model(x, t)
    output_shape = tuple(y.shape)
    output_finite = torch.isfinite(y).all().item()

    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        with autocast_context(device, dtype):
            y = model(x, t)
            loss = torch.nn.functional.mse_loss(y, target)
        loss.backward()
    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            with autocast_context(device, dtype):
                y = model(x, t)
    synchronize(device)
    forward_ms = (time.perf_counter() - start) * 1000.0 / iterations

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    last_loss = None
    for _ in range(iterations):
        model.zero_grad(set_to_none=True)
        with autocast_context(device, dtype):
            y = model(x, t)
            last_loss = torch.nn.functional.mse_loss(y, target)
        last_loss.backward()
    synchronize(device)
    forward_backward_ms = (time.perf_counter() - start) * 1000.0 / iterations

    batch_size = int(x.shape[0])
    return {
        "params": total_params,
        "trainable_params": trainable_params,
        "output_shape": output_shape,
        "forward_ms": forward_ms,
        "forward_backward_ms": forward_backward_ms,
        "peak_cuda_mb": peak_memory_mb(device),
        "samples_per_second": batch_size * 1000.0 / forward_ms,
        "output_finite": bool(output_finite),
        "grad_finite": finite_grads(model),
        "loss_finite": bool(torch.isfinite(last_loss).item()) if last_loss is not None else False,
    }


def format_memory(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--hrrr-channels", type=int, default=5)
    parser.add_argument("--hrrr-size", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    if device.type != "cuda" and dtype != torch.float32:
        raise ValueError("CPU benchmark supports only float32.")

    cases = [
        Case("CIFAR-like", 3, 32, 32),
        Case("ImageNet-like", 3, 64, 64),
        Case("HRRR-like", args.hrrr_channels, args.hrrr_size, args.hrrr_size),
    ]

    print("Runtime and memory are hardware- and configuration-dependent.")
    print("This benchmark does not establish generative quality or state-of-the-art performance.")
    print(f"device={device} dtype={dtype} batch_size={args.batch_size} warmup={args.warmup} iterations={args.iterations}")
    print(
        "case,model,params,trainable_params,output_shape,forward_ms,"
        "forward_backward_ms,peak_cuda_mb,samples_per_second,output_finite,grad_finite,loss_finite"
    )

    for case in cases:
        x = torch.randn(args.batch_size, case.channels, case.height, case.width, device=device, dtype=dtype)
        target = torch.randn_like(x)
        t = torch.rand(args.batch_size, device=device, dtype=torch.float32)
        for model_name in ("UNet", "Transformer"):
            model = make_model(model_name, case, n_steps=args.n_steps, patch_size=args.patch_size).to(device=device, dtype=dtype)
            stats = measure_model(
                model,
                x,
                t,
                target,
                device=device,
                dtype=dtype,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            print(
                f"{case.name},{model_name},{stats['params']},{stats['trainable_params']},"
                f"{stats['output_shape']},{stats['forward_ms']:.3f},{stats['forward_backward_ms']:.3f},"
                f"{format_memory(stats['peak_cuda_mb'])},{stats['samples_per_second']:.2f},"
                f"{stats['output_finite']},{stats['grad_finite']},{stats['loss_finite']}"
            )


if __name__ == "__main__":
    main()
