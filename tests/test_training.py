"""Training module unittests."""

import pytest
import torch
from genkit.training import train
from genkit.visitor import CoreMetricsVisitor, TrainVisitor


class _DummyGenerativeModel:
    def __init__(self, dim: int):
        self._net = torch.nn.Sequential(
            torch.nn.Linear(dim, 16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, dim),
        )

    def loss(self, x, z=None):
        y = self._net(x)
        return ((y - x) ** 2).mean()


def _run_train(x, *, visitors=None, **kwargs):
    defaults = {
        "batch_size": 24,
        "n_epochs": 2,
        "lr": 1e-3,
        "device": "cpu",
        "num_workers": 0,
        "lr_schedule": "constant",
        "freq_logging": 10,
    }
    defaults.update(kwargs)
    return train(_DummyGenerativeModel(dim=x.shape[1]), target_data=x, visitors=visitors, **defaults)


class _CounterVisitor(TrainVisitor):
    name = "counter"

    def __init__(self):
        self.train_start = 0
        self.epoch_end = 0
        self.losses = []
        self.grad_vars = []
        self.grad_norms = []

    def on_train_start(self, target, source, config):
        self.train_start += 1

    def on_epoch_end(self, loss: float, grad_var, grad_norm):
        self.epoch_end += 1
        self.losses.append(loss)
        self.grad_vars.append(grad_var)
        self.grad_norms.append(grad_norm)

    def get_records(self):
        return {
            "train_start": self.train_start,
            "epoch_end": self.epoch_end,
            "losses": self.losses,
            "grad_vars": self.grad_vars,
            "grad_norms": self.grad_norms,
        }


def test_train_diagnostics_with_core_visitor_enabled():
    x = torch.randn(96, 3, dtype=torch.float32)
    _, diagnostics = _run_train(x, n_epochs=4, visitors=[CoreMetricsVisitor()])

    assert "train_config" in diagnostics
    assert "visitors" in diagnostics

    core = diagnostics["visitors"]["core"]

    assert len(core["training_loss"]) == 4
    assert len(core["training_loss_std"]) == 4
    assert len(core["grad_variance_epoch"]) == 4
    assert len(core["grad_norm_epoch"]) == 4

    assert all(v >= 0.0 for v in core["training_loss_std"])
    assert all(v >= 0.0 for v in core["grad_variance_epoch"])
    assert all(v >= 0.0 for v in core["grad_norm_epoch"])

    assert torch.isfinite(torch.tensor(core["training_loss_std"])).all().item()
    assert torch.isfinite(torch.tensor(core["grad_variance_epoch"])).all().item()
    assert torch.isfinite(torch.tensor(core["grad_norm_epoch"])).all().item()


def test_train_default_uses_core_visitor_only():
    x = torch.randn(96, 3, dtype=torch.float32)
    _, diagnostics = _run_train(x)

    assert set(diagnostics["visitors"].keys()) == {"core"}
    core = diagnostics["visitors"]["core"]
    assert len(core["training_loss"]) == 2
    assert len(core["training_loss_std"]) == 2
    assert len(core["grad_variance_epoch"]) == 2
    assert len(core["grad_norm_epoch"]) == 2


def test_train_accepts_custom_visitors():
    x = torch.randn(96, 3, dtype=torch.float32)
    counter = _CounterVisitor()
    core = CoreMetricsVisitor()
    _, diagnostics = _run_train(x, n_epochs=3, visitors=[core, counter])

    assert counter.train_start == 1
    assert counter.epoch_end == 3
    assert len(counter.losses) == 3
    assert len(counter.grad_vars) == 3
    assert len(counter.grad_norms) == 3

    assert "core" in diagnostics["visitors"]
    assert "counter" in diagnostics["visitors"]


def test_train_tiny_dataset_with_large_batch_still_trains():
    x = torch.randn(3, 3, dtype=torch.float32)
    _, diagnostics = _run_train(x, batch_size=16, visitors=[CoreMetricsVisitor()])

    core = diagnostics["visitors"]["core"]
    assert len(core["training_loss"]) == 2
    assert torch.isfinite(torch.tensor(core["training_loss"])).all().item()


def test_train_empty_dataset_raises_clear_error():
    x = torch.empty(0, 3, dtype=torch.float32)

    with pytest.raises(ValueError, match="target_data is empty"):
        _run_train(x, batch_size=8, n_epochs=1)


def test_train_rejects_weight_decay_without_adamw():
    gm = _DummyGenerativeModel(dim=2)
    x = torch.randn(8, 2, dtype=torch.float32)

    with pytest.raises(ValueError, match="weight_decay"):
        train(
            gm,
            target_data=x,
            batch_size=4,
            n_epochs=1,
            lr=1e-3,
            device="cpu",
            num_workers=0,
            use_adamw=False,
            weight_decay=1e-4,
        )


def test_train_rejects_non_tensor_target_data():
    gm = _DummyGenerativeModel(dim=2)

    with pytest.raises(TypeError, match="target_data must be a torch.Tensor"):
        train(
            gm,
            target_data=[[1.0, 2.0]],
            batch_size=1,
            n_epochs=1,
            device="cpu",
        )


def test_train_rejects_source_with_wrong_dimension():
    gm = _DummyGenerativeModel(dim=2)
    x = torch.randn(8, 2, dtype=torch.float32)
    z = torch.randn(8, 3, dtype=torch.float32)

    with pytest.raises(ValueError, match="source_data dim"):
        train(
            gm,
            target_data=x,
            source_data=z,
            batch_size=4,
            n_epochs=1,
            device="cpu",
        )
