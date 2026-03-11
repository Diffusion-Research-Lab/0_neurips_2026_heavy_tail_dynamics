"""Training module unittests."""

import torch
from cauda.training import train
from cauda.visitor import CoreMetricsVisitor, TrainVisitor


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


class _CounterVisitor(TrainVisitor):
    name = "counter"

    def __init__(self):
        self.train_start = 0
        self.epoch_start = 0
        self.batch_end = 0
        self.epoch_end = 0
        self.train_end = 0

    def on_train_start(self, target, source, config):
        self.train_start += 1

    def on_epoch_start(self):
        self.epoch_start += 1

    def on_batch_end(self, loss: float, grad_var, grad_norm):
        self.batch_end += 1

    def on_epoch_end(self):
        self.epoch_end += 1

    def on_train_end(self):
        self.train_end += 1

    def get_records(self):
        return {
            "train_start": self.train_start,
            "epoch_start": self.epoch_start,
            "batch_end": self.batch_end,
            "epoch_end": self.epoch_end,
            "train_end": self.train_end,
        }


def test_train_diagnostics_with_core_visitor_enabled():
    dim = 3
    gm = _DummyGenerativeModel(dim=dim)
    x = torch.randn(96, dim, dtype=torch.float32)

    _, diagnostics = train(
        gm,
        target_data=x,
        batch_size=24,
        n_epochs=4,
        lr=1e-3,
        device="cpu",
        num_workers=0,
        lr_schedule="constant",
        freq_logging=10,
        visitors=[CoreMetricsVisitor()],
    )

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
    dim = 3
    gm = _DummyGenerativeModel(dim=dim)
    x = torch.randn(96, dim, dtype=torch.float32)

    _, diagnostics = train(
        gm,
        target_data=x,
        batch_size=24,
        n_epochs=2,
        lr=1e-3,
        device="cpu",
        num_workers=0,
        lr_schedule="constant",
        freq_logging=10,
    )

    assert set(diagnostics["visitors"].keys()) == {"core"}
    core = diagnostics["visitors"]["core"]
    assert len(core["training_loss"]) == 2
    assert len(core["training_loss_std"]) == 2
    assert len(core["grad_variance_epoch"]) == 2
    assert len(core["grad_norm_epoch"]) == 2


def test_train_accepts_custom_visitors():
    dim = 3
    gm = _DummyGenerativeModel(dim=dim)
    x = torch.randn(96, dim, dtype=torch.float32)

    counter = _CounterVisitor()
    core = CoreMetricsVisitor()

    _, diagnostics = train(
        gm,
        target_data=x,
        batch_size=24,
        n_epochs=3,
        lr=1e-3,
        device="cpu",
        num_workers=0,
        lr_schedule="constant",
        freq_logging=10,
        visitors=[core, counter],
    )

    assert counter.train_start == 1
    assert counter.epoch_start == 3
    assert counter.epoch_end == 3
    assert counter.train_end == 1
    assert counter.batch_end == 12  # 96 / 24 * 3 epochs

    assert "core" in diagnostics["visitors"]
    assert "counter" in diagnostics["visitors"]
