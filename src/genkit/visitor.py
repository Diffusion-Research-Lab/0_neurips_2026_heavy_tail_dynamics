"""Training visitors for collecting diagnostics."""

from typing import Any, Dict, List, Optional
import torch


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

    def on_epoch_end(
        self,
        loss: float = float("nan"),
        loss_std: float = float("nan"),
        grad_var: Optional[float] = None,
        grad_norm: Optional[float] = None,
        generative_model=None,
        net: Optional[torch.nn.Module] = None,
    ) -> None:
        """Hook called after the last batch of an epoch with epoch summaries."""
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
        """Initialize the public records."""
        self.training_loss: List[float] = []
        self.training_loss_std: List[float] = []
        self.grad_variance_epoch: List[float] = []
        self.grad_norm_epoch: List[float] = []

    def on_epoch_start(self, generative_model=None, net: Optional[torch.nn.Module] = None) -> None:
        """No-op hook kept for symmetry with the visitor interface."""
        return None

    def on_epoch_end(
        self,
        loss: float = float("nan"),
        loss_std: float = float("nan"),
        grad_var: Optional[float] = None,
        grad_norm: Optional[float] = None,
        generative_model=None,
        net: Optional[torch.nn.Module] = None,
    ) -> None:
        """Store one epoch summary produced by the trainer."""
        self.training_loss.append(float(loss))
        self.training_loss_std.append(float(loss_std))
        self.grad_variance_epoch.append(float("nan") if grad_var is None else float(grad_var))
        self.grad_norm_epoch.append(float("nan") if grad_norm is None else float(grad_norm))

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
