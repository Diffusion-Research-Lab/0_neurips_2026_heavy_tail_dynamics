"""Training visitors for collecting diagnostics."""

from typing import Any, Dict, List, Optional
import torch


def _mean_or_nan(values: List[float]) -> float:
    """Return the mean of a list or NaN when the list is empty."""
    if not values:
        return float("nan")
    return float(sum(values) / float(len(values)))


def _summary_from_values(values: List[float]) -> Dict[str, float]:
    """Summarize a list of scalars with standard descriptive statistics."""
    if len(values) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "q90": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    t = torch.tensor(values, dtype=torch.float64)
    q = torch.quantile(t, torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=torch.float64))
    return {
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "median": float(q[0].item()),
        "q90": float(q[1].item()),
        "q95": float(q[2].item()),
        "q99": float(q[3].item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
    }


class TrainVisitor:
    """Base class for training visitors (callback-style hooks)."""

    name: str = "base"

    def on_train_start(
        self,
        target,
        source,
        config: Dict[str, Any],
        generative_model=None,
        net: Optional[torch.nn.Module] = None,
    ) -> None:
        """Hook called once before the training loop starts."""
        pass

    def on_epoch_start(self, generative_model=None, net: Optional[torch.nn.Module] = None) -> None:
        """Hook called at the beginning of each epoch."""
        pass

    def on_batch_end(
        self,
        loss: float,
        grad_var: Optional[float],
        grad_norm: Optional[float],
        generative_model=None,
        net: Optional[torch.nn.Module] = None,
    ) -> None:
        """Hook called after each optimization step."""
        pass

    def on_epoch_end(self, generative_model=None, net: Optional[torch.nn.Module] = None) -> None:
        """Hook called after the last batch of an epoch."""
        pass

    def on_train_end(self, generative_model=None, net: Optional[torch.nn.Module] = None) -> None:
        """Hook called once after the training loop finishes."""
        pass

    def format_epoch_log(self) -> str:
        """Return a short human-readable summary for the current epoch."""
        return ""

    def get_records(self) -> Dict[str, Any]:
        """Return structured visitor outputs collected during training."""
        return {}


class CoreMetricsVisitor(TrainVisitor):
    """Collect core loss and gradient statistics during training."""

    name = "core"

    def __init__(self) -> None:
        """Initialize the public records and per-epoch accumulators."""
        # Public records
        self.training_loss: List[float] = []
        self.training_loss_std: List[float] = []
        self.grad_variance_epoch: List[float] = []
        self.grad_norm_epoch: List[float] = []

        # Per-epoch accumulators
        self._losses: List[float] = []
        self._grad_vars: List[float] = []
        self._grad_norms: List[float] = []

    def on_epoch_start(self, generative_model=None, net: Optional[torch.nn.Module] = None) -> None:
        """Reset per-epoch accumulators before processing a new epoch."""
        self._losses = []
        self._grad_vars = []
        self._grad_norms = []

    def on_batch_end(
        self,
        loss: float,
        grad_var: Optional[float],
        grad_norm: Optional[float],
        generative_model=None,
        net: Optional[torch.nn.Module] = None,
    ) -> None:
        """Accumulate batch-level loss and gradient statistics."""
        self._losses.append(float(loss))
        if grad_var is not None:
            self._grad_vars.append(float(grad_var))
        if grad_norm is not None:
            self._grad_norms.append(float(grad_norm))

    def on_epoch_end(self, generative_model=None, net: Optional[torch.nn.Module] = None) -> None:
        """Aggregate the current epoch statistics into the public records."""
        loss_stats = _summary_from_values(self._losses)
        self.training_loss.append(loss_stats["mean"])
        self.training_loss_std.append(loss_stats["std"])
        self.grad_variance_epoch.append(_mean_or_nan(self._grad_vars))
        self.grad_norm_epoch.append(_mean_or_nan(self._grad_norms))

    def format_epoch_log(self) -> str:
        """Format the latest epoch statistics for console logging."""
        if not self.training_loss:
            return ""
        return (
            f"loss {self.training_loss[-1]:.6f} ± {self.training_loss_std[-1]:.6f} | "
            f"grad_var {self.grad_variance_epoch[-1]:.3e} | "
            f"grad_norm {self.grad_norm_epoch[-1]:.3e}"
        )

    def get_records(self) -> Dict[str, Any]:
        """Return all collected loss and gradient histories."""
        return {
            "training_loss": self.training_loss,
            "training_loss_std": self.training_loss_std,
            "grad_variance_epoch": self.grad_variance_epoch,
            "grad_norm_epoch": self.grad_norm_epoch,
        }
