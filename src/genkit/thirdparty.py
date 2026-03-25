"""Optional adapters around vendored third-party generative backends."""

import sys
import warnings
from pathlib import Path
from typing import Optional
import torch
from ._abs import Base
from ._sampling import sample_gaussian


def _resolve_fdtype(fdtype: torch.dtype, dtype: Optional[torch.dtype]) -> torch.dtype:
    """Resolve the preferred floating dtype while preserving a legacy alias."""
    if dtype is None:
        return fdtype
    if fdtype != torch.float32 and fdtype != dtype:
        raise ValueError(f"Conflicting fdtype={fdtype} and legacy dtype={dtype}.")
    warnings.warn("`dtype` is deprecated in thirdparty adapters; use `fdtype` instead.")
    return dtype


class _NetAdapter(torch.nn.Module):
    """Wrap a native genkit network behind the vendor calling convention."""

    def __init__(self, net: torch.nn.Module):
        """Store the wrapped network used by a third-party backend."""
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Move inputs to the wrapped net device/dtype and delegate the forward pass."""
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        p = next(self.net.parameters(), None)
        if p is not None:
            if x.device != p.device:
                x = x.to(p.device)
            if t.device != p.device:
                t = t.to(p.device)
            if x.dtype != p.dtype:
                x = x.to(p.dtype)
            if t.dtype != p.dtype:
                t = t.to(p.dtype)

        return self.net(x, t)


class DLPMEpsOrigin(Base):
    """Adapter around the vendored authors' GenerativeLevyProcess(DLPM)."""
    _family = "vendor"

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 100,
        alpha: float = 1.8,
        authors_root: Optional[str] = None,
        time_spacing: str = "linear",
        rescale_timesteps: bool = True,
        isotropic: bool = True,
        loss_monte_carlo: str = "mean",
        monte_carlo_outer: int = 5,
        monte_carlo_inner: int = 1,
        lploss: float = 2.0,
        clamp_a: Optional[float] = None,
        clamp_eps: Optional[float] = None,
        scale: str = "scale_preserving",
        base_or_sample: torch.Tensor = None,
        fdtype: torch.dtype = torch.float32,
        idtype: torch.dtype = torch.int32,
        dtype: Optional[torch.dtype] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """Configure the vendored DLPM adapter and its training/sampling options."""
        fdtype = _resolve_fdtype(fdtype, dtype)
        super().__init__(net=net, dim=dim, n_steps=n_steps, base_or_sample=base_or_sample,
                         fdtype=fdtype, idtype=idtype, device=device)

        root = Path(authors_root).expanduser().resolve() if authors_root is not None else (
            Path(__file__).resolve().parent / "_vendor" / "DLPM"
        )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from dlpm.methods.GenerativeLevyProcess import GenerativeLevyProcess  # noqa: E402

        self._net_auth = _NetAdapter(self._net)
        self._loss_monte_carlo = loss_monte_carlo
        self._monte_carlo_outer = monte_carlo_outer
        self._monte_carlo_inner = monte_carlo_inner
        self._lploss = lploss
        self._clamp_a = clamp_a
        self._clamp_eps = clamp_eps

        self._glp = GenerativeLevyProcess(
            alpha=float(alpha),
            device=self._device,
            reverse_steps=int(n_steps),
            time_spacing=str(time_spacing),
            rescale_timesteps=bool(rescale_timesteps),
            isotropic=bool(isotropic),
            scale=str(scale),
        )

    def _sample_source_default(self, n_samples: int) -> torch.Tensor:
        """Draw Gaussian base samples for the DLPM adapter."""
        return sample_gaussian(n_samples, self._dim, device=self._device, dtype=self._fdtype)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Delegate loss computation to the vendored DLPM implementation."""
        if z is not None:
            warnings.warn("In 'DLPMEpsOrigin.loss', input 'z' is ignored (z = A G are sampled internally).")

        x = x.to(device=self._device, dtype=self._fdtype)
        out = self._glp.training_losses(
            models={"default": self._net_auth},
            x_start=x,
            loss_type="EPS_LOSS",
            lploss=self._lploss,
            loss_monte_carlo=self._loss_monte_carlo,
            monte_carlo_outer=self._monte_carlo_outer,
            monte_carlo_inner=self._monte_carlo_inner,
            clamp_a=self._clamp_a,
            clamp_eps=self._clamp_eps,
        )
        return out["loss"].to(device=self._device, dtype=self._fdtype).mean()

    @torch.no_grad()
    def sample(self, n_samples: int, **kwargs) -> torch.Tensor:
        """Generate samples with the vendored DLPM reverse sampler."""
        self._net.eval()

        if self._clamp_a is not None:
            self._glp.dlpm.gen_a.setParams(clamp_a=self._clamp_a)

        if self._clamp_eps is not None:
            self._glp.dlpm.gen_eps.setParams(clamp_eps=self._clamp_eps)

        x = self._glp.p_sample_loop(
            model=self._net_auth,
            shape=(int(n_samples), int(self._dim)),
            progress=bool(kwargs.get("progress", False)),
        )
        return x.to(device=self._device, dtype=self._fdtype)


class FlowMatchingOrigin(Base):
    """Minimal adapter around Meta's flow_matching 2D example components."""
    _family = "vendor"

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 100,
        t_min: float = 0.0,
        t_max: float = 1.0,
        package_root: Optional[str] = None,
        base_or_sample: torch.Tensor = None,
        fdtype: torch.dtype = torch.float32,
        idtype: torch.dtype = torch.int32,
        dtype: Optional[torch.dtype] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """Configure the flow-matching adapter around the vendored solver stack."""
        fdtype = _resolve_fdtype(fdtype, dtype)
        super().__init__(net=net, dim=dim, n_steps=n_steps, base_or_sample=base_or_sample,
                         fdtype=fdtype, idtype=idtype, device=device)
        self._t_min = float(t_min)
        self._t_max = float(t_max)
        if not (0.0 <= self._t_min < self._t_max <= 1.0):
            raise ValueError(f"Need 0 <= t_min < t_max <= 1, got {self._t_min}, {self._t_max}")

        root = Path(package_root).expanduser().resolve() if package_root is not None else (
            Path(__file__).resolve().parent / "_vendor" / "flow_matching"
        )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from flow_matching.path import AffineProbPath  # noqa: E402
        from flow_matching.path.scheduler import CondOTScheduler  # noqa: E402
        from flow_matching.solver import ODESolver  # noqa: E402
        from flow_matching.utils import ModelWrapper  # noqa: E402

        class _WrappedModel(ModelWrapper):
            """Adapt a genkit network to the vendored flow-matching solver API."""

            def forward(adapter_self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
                """Forward solver states through the wrapped genkit network."""
                if t.ndim == 0:
                    t = t.expand(x.size(0))
                if t.ndim == 1:
                    t = t.unsqueeze(-1)
                return self._net(x, t)

        self._path = AffineProbPath(scheduler=CondOTScheduler())
        self._solver = ODESolver(velocity_model=_WrappedModel(self._net))

    def _sample_source_default(self, n_samples: int) -> torch.Tensor:
        """Draw Gaussian source samples for the flow-matching adapter."""
        return sample_gaussian(n_samples, self._dim, device=self._device, dtype=self._fdtype)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Compute the vendored flow-matching loss on a sampled probability path."""
        x_1 = x.to(device=self._device, dtype=self._fdtype)
        x_0 = self._sample_source(x_1.size(0)) if z is None else z.to(device=self._device, dtype=self._fdtype)
        t = self._t_min + (self._t_max - self._t_min) * torch.rand(
            (x_1.size(0),), device=self._device, dtype=self._fdtype
        )
        path_sample = self._path.sample(x_0=x_0, x_1=x_1, t=t)

        v_hat = self._net(path_sample.x_t, path_sample.t.unsqueeze(-1))
        if v_hat.shape != path_sample.dx_t.shape:
            raise ValueError(
                f"Shape mismatch: v_hat={tuple(v_hat.shape)} vs dx_t={tuple(path_sample.dx_t.shape)}"
            )
        return torch.nn.functional.mse_loss(v_hat, path_sample.dx_t, reduction="none").mean()

    @torch.no_grad()
    def sample(self, n_samples: int, **kwargs) -> torch.Tensor:
        """Generate samples with the vendored ODE solver."""
        self._net.eval()

        x_init = self._sample_source(int(n_samples))
        time_grid = torch.tensor([self._t_min, self._t_max], device=self._device, dtype=self._fdtype)
        step_size = (self._t_max - self._t_min) / float(max(self._n_steps, 1))

        x = self._solver.sample(
            x_init=x_init,
            time_grid=time_grid,
            method="midpoint",
            step_size=step_size,
            return_intermediates=False,
        )
        return x.to(device=self._device, dtype=self._fdtype)


class ScoreSDEOrigin(Base):
    """Adapter around yang-song/score_sde_pytorch using a VE-SDE parameterization."""
    _family = "vendor"

    def __init__(
        self,
        net: torch.nn.Module,
        dim: int,
        n_steps: int = 100,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        time_eps: float = 1e-3,
        use_corrector: Optional[bool] = None,
        snr: Optional[float] = None,
        corrector_steps: int = 1,
        package_root: Optional[str] = None,
        base_or_sample: torch.Tensor = None,
        fdtype: torch.dtype = torch.float32,
        idtype: torch.dtype = torch.int32,
        dtype: Optional[torch.dtype] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """Configure the vendored VE-SDE adapter for low- or high-dimensional data."""
        fdtype = _resolve_fdtype(fdtype, dtype)
        super().__init__(net=net, dim=dim, n_steps=n_steps, base_or_sample=base_or_sample,
                         fdtype=fdtype, idtype=idtype, device=device)

        if not (0.0 < float(time_eps) < 1.0):
            raise ValueError(f"time_eps must be in (0,1), got {time_eps}.")

        root = Path(package_root).expanduser().resolve() if package_root is not None else (
            Path(__file__).resolve().parent / "_vendor" / "score_sde_pytorch"
        )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from sde_lib import VESDE  # noqa: E402
        from losses import get_sde_loss_fn  # noqa: E402
        from sampling import ReverseDiffusionPredictor, NoneCorrector, get_pc_sampler  # noqa: E402
        try:
            from sampling import LangevinCorrector  # noqa: E402
        except ImportError:
            LangevinCorrector = None

        low_dim = int(dim) <= 4
        if sigma_min is None:
            sigma_min = 1e-3 if low_dim else 1e-2
        if sigma_max is None:
            sigma_max = 5.0 if low_dim else 50.0

        self._sigma_min = float(sigma_min)
        self._sigma_max = float(sigma_max)
        self._time_eps = float(time_eps)
        if use_corrector is None:
            use_corrector = low_dim and LangevinCorrector is not None
        self._snr = 0.16 if snr is None and use_corrector else 0.0 if snr is None else float(snr)
        self._corrector_steps = int(corrector_steps)
        self._sde = VESDE(sigma_min=self._sigma_min, sigma_max=self._sigma_max, N=int(n_steps))
        self._loss_fn = get_sde_loss_fn(
            self._sde,
            train=True,
            reduce_mean=True,
            continuous=True,
            likelihood_weighting=False,
            eps=self._time_eps,
        )
        self._get_pc_sampler = get_pc_sampler
        self._predictor_cls = ReverseDiffusionPredictor
        self._corrector_cls = LangevinCorrector if use_corrector and LangevinCorrector is not None else NoneCorrector

        class _ScoreModel(torch.nn.Module):
            """Reshape vector networks to the spatial score-model interface."""

            def __init__(self, model: torch.nn.Module, dim: int):
                """Store the wrapped network and its flattened dimensionality."""
                super().__init__()
                self.model = model
                self.dim = int(dim)

            def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                """Map score-model inputs to the wrapped vector network and back."""
                x = x.view(x.size(0), self.dim)
                if labels.ndim == 1:
                    labels = labels.unsqueeze(-1)
                out = self.model(x, labels)
                return out.view(-1, self.dim, 1, 1)

        self._score_model = _ScoreModel(self._net, self._dim)

    def _sample_source_default(self, n_samples: int) -> torch.Tensor:
        """Draw Gaussian prior samples at the VE-SDE terminal scale."""
        return self._sigma_max * sample_gaussian(n_samples, self._dim, device=self._device, dtype=self._fdtype)

    def loss(self, x: torch.Tensor, z: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Delegate training loss computation to the vendored score-SDE objective."""
        if z is not None:
            warnings.warn("In 'ScoreSDEOrigin.loss', input 'z' is ignored (Gaussian perturbations are sampled internally).")

        x = x.to(device=self._device, dtype=self._fdtype).view(-1, self._dim, 1, 1)
        return self._loss_fn(self._score_model, x)

    @torch.no_grad()
    def sample(self, n_samples: int, **kwargs) -> torch.Tensor:
        """Generate samples with the vendored predictor-corrector sampler."""
        self._net.eval()
        sampling_fn = self._get_pc_sampler(
            self._sde,
            (int(n_samples), int(self._dim), 1, 1),
            self._predictor_cls,
            self._corrector_cls,
            inverse_scaler=lambda x: x,
            snr=self._snr,
            n_steps=self._corrector_steps,
            probability_flow=False,
            continuous=True,
            denoise=True,
            eps=self._time_eps,
            device=self._device,
        )
        x, _ = sampling_fn(self._score_model)
        return x.view(int(n_samples), int(self._dim)).to(device=self._device, dtype=self._fdtype)
