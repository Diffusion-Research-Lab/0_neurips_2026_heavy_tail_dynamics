"""Training module unittests."""

import pytest
import torch
from genkit.training import (
    train, _build_scheduler, _save_ckpt, _build_loader,
    _call_visitor_hook, _build_hook_param_cache,
)
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


def _make_opt():
    return torch.optim.SGD([torch.nn.Parameter(torch.tensor(0.0))], lr=1.0)


def test_build_scheduler_cosine_lr_decreases_over_steps():
    opt = _make_opt()
    sched = _build_scheduler(opt, "cosine", warmup_steps=0, total_steps=20)
    lrs = []
    for _ in range(20):
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[0] > lrs[-1]


def test_build_scheduler_cosine_lr_keeps_floor():
    opt = _make_opt()
    sched = _build_scheduler(opt, "cosine", warmup_steps=0, total_steps=20, cosine_eta_min_ratio=0.1)
    lrs = []
    for _ in range(20):
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[-1] == pytest.approx(0.1, rel=1e-6, abs=1e-6)


def test_build_scheduler_linear_lr_approaches_zero():
    opt = _make_opt()
    sched = _build_scheduler(opt, "linear", warmup_steps=0, total_steps=20)
    for _ in range(20):
        sched.step()
    assert opt.param_groups[0]["lr"] < 0.1


def test_build_scheduler_warmup_increases_lr_initially():
    opt = _make_opt()
    sched = _build_scheduler(opt, "cosine", warmup_steps=5, total_steps=20)
    lrs = []
    for _ in range(5):
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[-1] > lrs[0]


def test_build_scheduler_constant_does_not_change_lr():
    opt = _make_opt()
    initial_lr = opt.param_groups[0]["lr"]
    sched = _build_scheduler(opt, "constant", warmup_steps=0, total_steps=10)
    for _ in range(10):
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - initial_lr) < 1e-6


def test_build_scheduler_unknown_raises():
    opt = _make_opt()
    with pytest.raises(ValueError, match="Unknown lr_schedule"):
        _build_scheduler(opt, "bad_schedule", warmup_steps=0, total_steps=10)


def test_save_ckpt_writes_file_and_symlink(tmp_path):
    net = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(net.parameters())
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=1)
    _save_ckpt(tmp_path, 3, 1, 10, 0.5, net, opt, sched, {"lr": 1e-3})
    assert (tmp_path / "ckpt_epoch_0001.pt").exists()
    assert (tmp_path / "ckpt_last.pt").exists()
    ckpt = torch.load(tmp_path / "ckpt_epoch_0001.pt", map_location="cpu")
    assert ckpt["epoch"] == 1
    assert ckpt["loss"] == pytest.approx(0.5)


def test_save_ckpt_rotates_old_checkpoints_beyond_keep_last(tmp_path):
    net = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(net.parameters())
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=1)
    for epoch in range(1, 5):
        _save_ckpt(tmp_path, 2, epoch, epoch * 10, 0.5, net, opt, sched, {})
    kept = sorted(tmp_path.glob("ckpt_epoch_*.pt"))
    assert len(kept) == 2
    assert kept[0].name == "ckpt_epoch_0003.pt"
    assert kept[1].name == "ckpt_epoch_0004.pt"


def test_build_loader_warns_for_num_workers_on_tensor():
    target = torch.randn(10, 2)
    with pytest.warns(UserWarning, match="num_workers > 0 is ignored"):
        _build_loader(target, batch_size=5, num_workers=2, pin_memory=False,
                      persistent_workers=None, prefetch_factor=None)


def test_call_visitor_hook_filters_extra_kwargs_not_in_signature():
    received = {}

    class _MinimalVisitor(TrainVisitor):
        def on_epoch_end(self, loss=0.0):
            received["loss"] = loss

    v = _MinimalVisitor()
    cache = _build_hook_param_cache([v], ["on_epoch_end"])
    # grad_var and grad_norm are not in the signature — must be silently dropped
    _call_visitor_hook(v, "on_epoch_end", cache, loss=0.42, grad_var=0.1, grad_norm=0.5)
    assert received["loss"] == pytest.approx(0.42)


def test_train_with_cosine_schedule_completes():
    x = torch.randn(50, 2, dtype=torch.float32)
    _, diagnostics = _run_train(x, n_epochs=3, lr_schedule="cosine", visitors=[CoreMetricsVisitor()])
    assert len(diagnostics["visitors"]["core"]["training_loss"]) == 3


def test_train_with_grad_clip_norm_completes():
    x = torch.randn(50, 2, dtype=torch.float32)
    _, diagnostics = _run_train(x, n_epochs=2, grad_clip_norm=1.0, visitors=[CoreMetricsVisitor()])
    assert len(diagnostics["visitors"]["core"]["training_loss"]) == 2
