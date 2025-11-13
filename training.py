"""Training utilities for diffusion models."""

from typing import Callable, Tuple, List
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


StepFn = Callable[
    [nn.Module, torch.Tensor, int, torch.dtype, torch.device],
    torch.Tensor,
]


def _train(
    model: nn.Module,
    data: torch.Tensor,
    batch_size: int,
    n_epochs: int,
    lr: float,
    device: torch.device,
    step_fn: StepFn,
) -> Tuple[nn.Module, dict]:
    """Train model."""
    train_loss_hist: List[float] = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True, drop_last=True)
    model.train()

    dim = data.shape[-1]
    dtype = data.dtype

    for epoch in range(n_epochs):
        epoch_losses: List[float] = []

        for (x_1,) in data_loader:
            x_1 = x_1.to(device=device, dtype=dtype)
            loss = step_fn(model, x_1, dim, dtype, device)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = float(torch.tensor(epoch_losses, device=device).mean().item())
        train_loss_hist.append(mean_loss)
        logging.info(f"epoch {epoch+1:3d} / {n_epochs:3d} | loss {mean_loss:.6f}")

    logging.info("Training finished.")
    return model, {"training_loss": train_loss_hist}
