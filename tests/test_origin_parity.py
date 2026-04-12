"""Tight parity tests between native models and origin adapters."""

from pathlib import Path
import numpy as np
import pytest
import torch
from genkit.diffusion import DDPMV, DDPMX0, DLPMEps
from genkit.flow import GaussianFlowLinear
from genkit.thirdparty import DLPMEpsOrigin, FlowMatchingOrigin, ScoreSDEOrigin


class _AffineTimeNet(torch.nn.Module):
    """Simple deterministic network with trainable affine dependence on x and t."""

    def __init__(self):
        """Initialize a tiny network with fixed scalar parameters."""
        super().__init__()
        self.wx = torch.nn.Parameter(torch.tensor(0.35))
        self.wt = torch.nn.Parameter(torch.tensor(-0.2))
        self.b = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, x, t):
        """Return an affine map that broadcasts over x and t."""
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return self.wx * x + self.wt * t + self.b


def _write(path: Path, content: str):
    """Write one helper file inside a temporary fake-vendor tree."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _collect_grads(loss: torch.Tensor, model: torch.nn.Module) -> list[torch.Tensor]:
    """Return detached parameter gradients for one scalar loss."""
    grads = torch.autograd.grad(loss, tuple(model.parameters()))
    return [grad.detach().clone() for grad in grads]


def _assert_tensors_close(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-6, rtol: float = 1e-6):
    """Assert two tensors are numerically indistinguishable at tight tolerance."""
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), (actual, expected)


def _assert_grad_lists_close(actual: list[torch.Tensor], expected: list[torch.Tensor], atol: float = 1e-6, rtol: float = 1e-6):
    """Assert two parameter-gradient lists match elementwise."""
    assert len(actual) == len(expected)
    for actual_grad, expected_grad in zip(actual, expected):
        _assert_tensors_close(actual_grad, expected_grad, atol=atol, rtol=rtol)


def _make_flow_matching_vendor(tmp_path: Path) -> Path:
    """Build a tiny flow-matching vendor that mirrors GaussianFlowLinear exactly."""
    root = tmp_path / "flow_matching_vendor"
    _write(root / "flow_matching" / "__init__.py", "")
    _write(
        root / "flow_matching" / "path" / "__init__.py",
        """
import torch
from types import SimpleNamespace


class AffineProbPath:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def sample(self, x_0, x_1, t):
        x_t = (1.0 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * x_1
        dx_t = x_1 - x_0
        return SimpleNamespace(x_t=x_t, dx_t=dx_t, t=t, x_0=x_0, x_1=x_1)
""",
    )
    _write(
        root / "flow_matching" / "path" / "scheduler.py",
        """
class CondOTScheduler:
    pass
""",
    )
    _write(
        root / "flow_matching" / "solver" / "__init__.py",
        """
class ODESolver:
    def __init__(self, velocity_model):
        self.velocity_model = velocity_model

    def sample(self, x_init, step_size, method, time_grid, return_intermediates=False, **kwargs):
        x = x_init
        t = time_grid[:1].expand(x.size(0))
        for _ in range(max(int(round((time_grid[-1] - time_grid[0]).item() / step_size)), 0)):
            v0 = self.velocity_model(x=x, t=t)
            t_next = (t + step_size).clamp_max(time_grid[-1])
            x_euler = x + step_size * v0
            v1 = self.velocity_model(x=x_euler, t=t_next)
            x = x + 0.5 * step_size * (v0 + v1)
            t = t_next
        return x
""",
    )
    _write(
        root / "flow_matching" / "utils" / "__init__.py",
        """
import torch.nn as nn


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
""",
    )
    return root


def _make_score_sde_vendor(tmp_path: Path, mode: str) -> Path:
    """Build a fake score-SDE vendor that mirrors one DDPM parameterization exactly."""
    root = tmp_path / f"score_sde_vendor_{mode}"
    _write(
        root / "sde_lib.py",
        f"""
MODE = {mode!r}


class VESDE:
    def __init__(self, sigma_min=0.01, sigma_max=50.0, N=100):
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.N = int(N)
        self.mode = MODE
""",
    )
    _write(
        root / "losses.py",
        """
import torch
from genkit.utils import cosine_schedule


def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=False, eps=1e-3):
    def loss_fn(model, batch):
        x_1 = batch.view(batch.size(0), -1)
        t = torch.randint(1, sde.N + 1, (x_1.size(0),), device=x_1.device, dtype=torch.int64)
        alpha_bar, _, _, _ = cosine_schedule(sde.N, x_1.device, x_1.dtype, torch.int64)
        a_bar_t = alpha_bar.index_select(0, t - 1).unsqueeze(-1)
        noise = torch.randn_like(x_1)
        x_t = torch.sqrt(a_bar_t) * x_1 + torch.sqrt(1.0 - a_bar_t) * noise
        t_norm = (t.to(dtype=x_1.dtype) / sde.N).unsqueeze(-1)
        pred = model.model(x_t, t_norm)
        if sde.mode == "v":
            target = torch.sqrt(a_bar_t) * noise - torch.sqrt(1.0 - a_bar_t) * x_1
            return torch.nn.functional.mse_loss(pred, target, reduction="none").mean()
        weight = (a_bar_t / (1.0 - a_bar_t)).clamp(min=1e-3, max=1e3)
        return (weight * (pred - x_1).square()).mean()
    return loss_fn
""",
    )
    _write(
        root / "sampling.py",
        """
import torch
from genkit.utils import cosine_schedule


class ReverseDiffusionPredictor:
    pass


class NoneCorrector:
    pass


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cpu'):
    def sampler(model):
        n_samples, dim = int(shape[0]), int(shape[1])
        x = torch.randn((n_samples, dim), device=device)
        alpha_bar, alphas, betas, sqrt_post_var = cosine_schedule(sde.N, x.device, x.dtype, torch.int64)
        for t in range(sde.N, 0, -1):
            t_idx = t - 1
            t_norm = torch.full((n_samples, 1), t / sde.N, device=x.device, dtype=x.dtype)
            raw = model.model(x, t_norm)
            if sde.mode == "v":
                eps_hat = torch.sqrt(1.0 - alpha_bar[t_idx]) * x + torch.sqrt(alpha_bar[t_idx]) * raw
            else:
                eps_hat = (x - torch.sqrt(alpha_bar[t_idx]) * raw) / torch.sqrt(1.0 - alpha_bar[t_idx])
            x = (x - betas[t_idx] / torch.sqrt(1.0 - alpha_bar[t_idx]) * eps_hat) / torch.sqrt(alphas[t_idx])
            if t > 1:
                x = x + sqrt_post_var[t_idx] * torch.randn_like(x)
        return inverse_scaler(x.view(n_samples, dim, 1, 1)), sde.N
    return sampler
""",
    )
    return root


def test_dlpmeps_matches_origin_loss_and_gradients():
    """DLPM native and origin implementations should match on the training path."""
    x = torch.randn(6, 2, dtype=torch.float32)
    for seed in (0, 1, 2):
        native = DLPMEps(
            net=_AffineTimeNet(),
            dim=2,
            n_steps=8,
            alpha=1.9,
            n_trial_A=1,
            n_trial_G=1,
            reduce_type="median",
            fdtype=torch.float32,
            device="cpu",
        )
        origin = DLPMEpsOrigin(
            net=_AffineTimeNet(),
            dim=2,
            n_steps=8,
            alpha=1.9,
            monte_carlo_outer=1,
            monte_carlo_inner=1,
            loss_monte_carlo="median",
            fdtype=torch.float32,
            device="cpu",
        )

        np.random.seed(seed)
        torch.manual_seed(seed)
        native_loss = native.loss(x)
        native_grads = _collect_grads(native_loss, native._net)

        np.random.seed(seed)
        torch.manual_seed(seed)
        origin_loss = origin.loss(x)
        origin_grads = _collect_grads(origin_loss, origin._net)

        _assert_tensors_close(native_loss.detach(), origin_loss.detach(), atol=1e-7, rtol=1e-7)
        _assert_grad_lists_close(native_grads, origin_grads, atol=2e-7, rtol=1e-6)


def test_dlpmeps_matches_origin_sample_and_one_step_update():
    """Native DLPM should match the origin adapter for sampling and one SGD step."""
    x = torch.randn(6, 2, dtype=torch.float32)
    native = DLPMEps(
        net=_AffineTimeNet(),
        dim=2,
        n_steps=8,
        alpha=1.9,
        n_trial_A=1,
        n_trial_G=1,
        reduce_type="median",
        fdtype=torch.float32,
        device="cpu",
    )
    origin = DLPMEpsOrigin(
        net=_AffineTimeNet(),
        dim=2,
        n_steps=8,
        alpha=1.9,
        monte_carlo_outer=1,
        monte_carlo_inner=1,
        loss_monte_carlo="median",
        fdtype=torch.float32,
        device="cpu",
    )

    torch.manual_seed(3)
    np.random.seed(3)
    native_sample = native.sample(5)
    torch.manual_seed(3)
    np.random.seed(3)
    origin_sample = origin.sample(5)
    _assert_tensors_close(native_sample, origin_sample, atol=2e-5, rtol=1e-5)

    native_optim = torch.optim.SGD(native._net.parameters(), lr=0.05)
    origin_optim = torch.optim.SGD(origin._net.parameters(), lr=0.05)

    torch.manual_seed(5)
    np.random.seed(5)
    native_loss = native.loss(x)
    native_optim.zero_grad()
    native_loss.backward()
    native_optim.step()

    torch.manual_seed(5)
    np.random.seed(5)
    origin_loss = origin.loss(x)
    origin_optim.zero_grad()
    origin_loss.backward()
    origin_optim.step()

    for native_param, origin_param in zip(native._net.parameters(), origin._net.parameters()):
        _assert_tensors_close(native_param.detach(), origin_param.detach(), atol=2e-7, rtol=1e-6)


def test_gaussian_flow_linear_matches_flow_matching_origin(tmp_path, monkeypatch):
    """GaussianFlowLinear should match the origin flow-matching adapter exactly."""
    vendor_root = _make_flow_matching_vendor(tmp_path)
    x = torch.tensor([[1.0, -0.5], [0.25, 2.0], [-1.5, 0.75]], dtype=torch.float32)
    z = torch.tensor([[0.2, -1.0], [1.5, 0.3], [0.7, -0.4]], dtype=torch.float32)
    t = torch.tensor([[0.15], [0.5], [0.85]], dtype=torch.float32)
    base = torch.tensor([0.4, -0.3], dtype=torch.float32)

    native = GaussianFlowLinear(net=_AffineTimeNet(), dim=2, n_steps=6, t_min=0.0, t_max=1.0, base_or_sample=base, fdtype=torch.float32)
    origin = FlowMatchingOrigin(net=_AffineTimeNet(), dim=2, n_steps=6, package_root=str(vendor_root), base_or_sample=base, fdtype=torch.float32)

    monkeypatch.setattr(torch, "rand", lambda size, device=None, dtype=None: t.squeeze(-1).to(device=device, dtype=dtype))
    origin_loss = origin.loss(x, z=z)
    monkeypatch.undo()
    native_loss = native.loss(x, z=z, t=t)

    native_grads = _collect_grads(native_loss, native._net)
    origin_grads = _collect_grads(origin_loss, origin._net)

    _assert_tensors_close(native_loss.detach(), origin_loss.detach())
    _assert_grad_lists_close(native_grads, origin_grads)

    native_sample = native.sample(4)
    origin_sample = origin.sample(4)
    _assert_tensors_close(native_sample, origin_sample)


def test_gaussian_flow_linear_matches_flow_matching_origin_after_one_step(tmp_path, monkeypatch):
    """Native flow matching and origin adapter should remain aligned after one optimizer step."""
    vendor_root = _make_flow_matching_vendor(tmp_path)
    x = torch.tensor([[1.0, -0.5], [0.25, 2.0], [-1.5, 0.75]], dtype=torch.float32)
    z = torch.tensor([[0.2, -1.0], [1.5, 0.3], [0.7, -0.4]], dtype=torch.float32)
    t = torch.tensor([[0.15], [0.5], [0.85]], dtype=torch.float32)
    base = torch.tensor([0.4, -0.3], dtype=torch.float32)

    native = GaussianFlowLinear(net=_AffineTimeNet(), dim=2, n_steps=6, t_min=0.0, t_max=1.0, base_or_sample=base, fdtype=torch.float32)
    origin = FlowMatchingOrigin(net=_AffineTimeNet(), dim=2, n_steps=6, package_root=str(vendor_root), base_or_sample=base, fdtype=torch.float32)

    native_optim = torch.optim.SGD(native._net.parameters(), lr=0.1)
    origin_optim = torch.optim.SGD(origin._net.parameters(), lr=0.1)

    monkeypatch.setattr(torch, "rand", lambda size, device=None, dtype=None: t.squeeze(-1).to(device=device, dtype=dtype))
    native_loss = native.loss(x, z=z, t=t)
    origin_loss = origin.loss(x, z=z)

    native_optim.zero_grad()
    native_loss.backward()
    native_optim.step()

    origin_optim.zero_grad()
    origin_loss.backward()
    origin_optim.step()
    monkeypatch.undo()

    for native_param, origin_param in zip(native._net.parameters(), origin._net.parameters()):
        _assert_tensors_close(native_param.detach(), origin_param.detach(), atol=2e-7, rtol=1e-6)


@pytest.mark.parametrize(("native_cls", "mode"), [(DDPMV, "v"), (DDPMX0, "x0")])
def test_ddpm_variants_match_score_sde_origin_with_exact_vendor(tmp_path, native_cls, mode):
    """One fake score-SDE backend per parameterization should reproduce native DDPM exactly."""
    vendor_root = _make_score_sde_vendor(tmp_path, mode=mode)
    x = torch.tensor([[1.0, -0.5], [0.25, 2.0], [-1.5, 0.75]], dtype=torch.float32)

    native = native_cls(net=_AffineTimeNet(), dim=2, n_steps=6, fdtype=torch.float32, device="cpu")
    origin = ScoreSDEOrigin(net=_AffineTimeNet(), dim=2, n_steps=6, package_root=str(vendor_root), fdtype=torch.float32, device="cpu")

    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        native_loss = native.loss(x)
        native_grads = _collect_grads(native_loss, native._net)

        torch.manual_seed(seed)
        origin_loss = origin.loss(x)
        origin_grads = _collect_grads(origin_loss, origin._net)

        _assert_tensors_close(native_loss.detach(), origin_loss.detach(), atol=1e-7, rtol=1e-7)
        _assert_grad_lists_close(native_grads, origin_grads, atol=2e-7, rtol=1e-6)

        torch.manual_seed(seed)
        native_sample = native.sample(5)
        torch.manual_seed(seed)
        origin_sample = origin.sample(5)
        _assert_tensors_close(native_sample, origin_sample, atol=1e-6, rtol=1e-6)
