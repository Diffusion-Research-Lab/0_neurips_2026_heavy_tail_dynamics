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


def _remove_sys_path_entry(path: Path) -> None:
    """Remove all occurrences of a filesystem path from sys.path."""
    path_str = str(path)
    sys.path[:] = [entry for entry in sys.path if entry != path_str]


def _resolve_vendor_root(package_root: Optional[str], vendor_name: str) -> Path:
    """Resolve the filesystem root for a vendored dependency."""
    if package_root is not None:
        return Path(package_root).expanduser().resolve()
    return Path(__file__).resolve().parent / "_vendor" / vendor_name


def _import_dlpm_vendor(authors_root: Optional[str]):
    """Import the vendored DLPM entrypoint."""
    root = _resolve_vendor_root(authors_root, "DLPM")
    sys.path.insert(0, str(root))
    try:
        from dlpm.methods.GenerativeLevyProcess import GenerativeLevyProcess  # noqa: E402
    finally:
        _remove_sys_path_entry(root)
    return GenerativeLevyProcess


def _import_flow_matching_vendor(package_root: Optional[str]):
    """Import the vendored flow-matching components."""
    root = _resolve_vendor_root(package_root, "flow_matching")
    sys.path.insert(0, str(root))
    try:
        from flow_matching.path import AffineProbPath  # noqa: E402
        from flow_matching.path.scheduler import CondOTScheduler  # noqa: E402
        from flow_matching.solver import ODESolver  # noqa: E402
        from flow_matching.utils import ModelWrapper  # noqa: E402
    finally:
        _remove_sys_path_entry(root)
    return AffineProbPath, CondOTScheduler, ODESolver, ModelWrapper


def _import_score_sde_vendor(package_root: Optional[str]):
    """Import vendored score-SDE components while avoiding global module collisions."""
    root = _resolve_vendor_root(package_root, "score_sde_pytorch")
    generic_vendor_modules = {"sde_lib", "losses", "sampling", "models", "utils"}
    modules_before = set(sys.modules)
    sys.path.insert(0, str(root))
    try:
        from sde_lib import VESDE  # noqa: E402
        from losses import get_sde_loss_fn  # noqa: E402
        from sampling import ReverseDiffusionPredictor, NoneCorrector, get_pc_sampler  # noqa: E402
        try:
            from sampling import LangevinCorrector  # noqa: E402
        except ImportError:
            LangevinCorrector = None
    finally:
        _remove_sys_path_entry(root)
        loaded_vendor_modules = {
            name
            for name in set(sys.modules) - modules_before
            if name in generic_vendor_modules or any(name.startswith(f"{prefix}.") for prefix in generic_vendor_modules)
        }
        for name in loaded_vendor_modules:
            sys.modules.pop(name, None)
    return VESDE, get_sde_loss_fn, ReverseDiffusionPredictor, NoneCorrector, get_pc_sampler, LangevinCorrector


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
        monte_carlo_outer: int = 1,
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
        if base_or_sample is not None:
            warnings.warn("DLPMEpsOrigin ignores base_or_sample during sampling.")

        GenerativeLevyProcess = _import_dlpm_vendor(authors_root)

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
        raise ValueError("In DLPMEpsOrigin sampling is handled internally by the DLPM sampler.")

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
        if float(t_min) != 0.0 or float(t_max) != 1.0:
            warnings.warn(
                "FlowMatchingOrigin always uses t_min=0 and t_max=1; passed values are ignored."
            )
        self._t_min = 0.0
        self._t_max = 1.0

        AffineProbPath, CondOTScheduler, ODESolver, ModelWrapper = _import_flow_matching_vendor(package_root)

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
        if x_0.shape != x_1.shape:
            raise ValueError(f"z must have shape {tuple(x_1.shape)}, got {tuple(x_0.shape)}.")
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
        loss_eps: float = 1e-5,
        sampling_eps: float = 1e-5,
        reduce_mean: bool = False,
        time_eps: Optional[float] = None,
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
        if base_or_sample is not None:
            warnings.warn("ScoreSDEOrigin ignores base_or_sample during sampling.")

        if time_eps is not None:
            warnings.warn("`time_eps` is deprecated in ScoreSDEOrigin; use `loss_eps` and `sampling_eps` instead.")
            loss_eps = float(time_eps)
            sampling_eps = float(time_eps)
        if not (0.0 < float(loss_eps) < 1.0):
            raise ValueError(f"loss_eps must be in (0,1), got {loss_eps}.")
        if not (0.0 < float(sampling_eps) < 1.0):
            raise ValueError(f"sampling_eps must be in (0,1), got {sampling_eps}.")

        (
            VESDE,
            get_sde_loss_fn,
            ReverseDiffusionPredictor,
            NoneCorrector,
            get_pc_sampler,
            LangevinCorrector,
        ) = _import_score_sde_vendor(package_root)

        low_dim = int(dim) <= 4
        if sigma_min is None:
            sigma_min = 1e-3 if low_dim else 1e-2
        if sigma_max is None:
            sigma_max = 5.0 if low_dim else 50.0

        self._sigma_min = float(sigma_min)
        self._sigma_max = float(sigma_max)
        self._loss_eps = float(loss_eps)
        self._sampling_eps = float(sampling_eps)
        if use_corrector is None:
            use_corrector = LangevinCorrector is not None
        self._snr = 0.16 if snr is None and use_corrector else 0.0 if snr is None else float(snr)
        self._corrector_steps = int(corrector_steps)
        self._sde = VESDE(sigma_min=self._sigma_min, sigma_max=self._sigma_max, N=int(n_steps))
        self._loss_fn = get_sde_loss_fn(
            self._sde,
            train=True,
            reduce_mean=bool(reduce_mean),
            continuous=True,
            likelihood_weighting=False,
            eps=self._loss_eps,
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
            eps=self._sampling_eps,
            device=self._device,
        )
        x, _ = sampling_fn(self._score_model)
        return x.view(int(n_samples), int(self._dim)).to(device=self._device, dtype=self._fdtype)
