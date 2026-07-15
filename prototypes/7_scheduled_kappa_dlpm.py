import logging
from pathlib import Path
import torch
from genkit._noise import sample_scaled_scalar_alpha_stable
from genkit.diffusion import DLPMEps
from genkit.training import train
from _constants import DIM, N_MMD, N_STEPS, N_TRIALS
from _utils import (
    add_test_vs_true_sample,
    evaluate_model,
    load_alpha_stable,
    make_net,
    make_train_kwargs,
    save_tail_figure,
    setup,
)

log_path = Path(__file__).with_name("logs") / f"{Path(__file__).stem}.log"
log_path.parent.mkdir(exist_ok=True)
logging.basicConfig(filename=log_path, filemode="w", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


########################################################################################################################
# Additional classes

class ScheduledKappaDLPM(DLPMEps):
    """Time-inhomogeneous DLPM with kappa_t decreasing from 2 to kappa_min."""

    def __init__(self, *args, kappa_min=1.7, kappa_power=1.0, **kwargs):
        kappa_min = float(kappa_min)
        kappa_power = float(kappa_power)
        if not (0.0 < kappa_min <= 2.0):
            raise ValueError(f"kappa_min must be in (0, 2], got {kappa_min}.")
        if kappa_power <= 0.0:
            raise ValueError(f"kappa_power must be positive, got {kappa_power}.")

        kwargs.pop("alpha", None)
        super().__init__(*args, alpha=kappa_min, **kwargs)
        if self._n_steps < 2:
            raise ValueError(f"n_steps must be >= 2, got {self._n_steps}.")

        self.kappa_min = kappa_min
        self.kappa_power = kappa_power
        u = torch.linspace(0.0, 1.0, self._n_steps, device=self._device, dtype=self._fdtype)
        self._kappa_t = 2.0 - (2.0 - self.kappa_min) * u.pow(self.kappa_power)

        self._gamma_t = (1.0 - self._betas).clamp_min(self._eps).pow(1.0 / self._kappa_t)
        self._sigma_t = (1.0 - self._gamma_t.pow(self._kappa_t)).clamp_min(0.0).pow(1.0 / self._kappa_t)
        self._gamma_1_t = torch.ones(self._n_steps + 1, device=self._device, dtype=self._fdtype)
        self._gamma_1_t[1:] = torch.cumprod(self._gamma_t, dim=0)
        self._sigma_1_t = torch.zeros(self._n_steps + 1, device=self._device, dtype=self._fdtype)
        self._sigma_1_t[1:] = (1.0 - self._gamma_1_t[1:].pow(self._kappa_t)).clamp_min(0.0).pow(1.0 / self._kappa_t)

    def draw_A_for_kappa(self, kappa):
        kappa = torch.as_tensor(kappa, device=self._device, dtype=self._fdtype).reshape(-1)
        if ((kappa <= 0.0) | (kappa > 2.0)).any():
            raise ValueError("all kappa values must be in (0, 2].")

        A = torch.empty_like(kappa)
        for value in torch.unique(kappa):
            mask = kappa == value
            A[mask] = sample_scaled_scalar_alpha_stable(
                n_samples=int(mask.sum().item()),
                alpha=float(value.item()),
                device=self._device,
                dtype=self._fdtype,
            ).reshape(-1)
        return A

    def _kappa_for_t(self, t):
        idx = torch.as_tensor(t, device=self._device, dtype=torch.long).reshape(-1).clamp(0, self._n_steps - 1)
        return self._kappa_t.index_select(0, idx)

    def _draw_A(self, n, kappa=None):
        n = int(n)
        if kappa is None:
            kappa = self._kappa_t[-1].expand(n)
        else:
            kappa = torch.as_tensor(kappa, device=self._device, dtype=self._fdtype)
            if kappa.ndim == 0:
                kappa = kappa.expand(n)
            else:
                kappa = kappa.reshape(-1)
            if kappa.numel() != n:
                raise ValueError(f"expected {n} kappa values, got {kappa.numel()}.")
        return self.draw_A_for_kappa(kappa).reshape(n, 1)

    def _sample_source_default(self, n_samples, *, kappa=None, expand_trials=False, return_A=False):
        n_samples = int(n_samples)
        if expand_trials:
            if kappa is None:
                kappa = self._kappa_t[-1].expand(n_samples)
            else:
                kappa = torch.as_tensor(kappa, device=self._device, dtype=self._fdtype).reshape(-1)
            if kappa.numel() != n_samples:
                raise ValueError(f"expected {n_samples} kappa values, got {kappa.numel()}.")

            A = self.draw_A_for_kappa(kappa.view(1, n_samples).expand(self._n_trial_A, n_samples).reshape(-1))
            A = A.view(self._n_trial_A, 1, n_samples).expand(self._n_trial_A, self._n_trial_G, n_samples).reshape(-1)
            G = torch.randn(self._n_trial_A * self._n_trial_G * n_samples, *self._sample_shape, device=self._device, dtype=self._fdtype)
            eps = self._expand_batch_scalar(A, G).sqrt() * G
            return (eps, A) if return_A else eps

        A = self._draw_A(n_samples, kappa=kappa).reshape(n_samples)
        G = torch.randn(n_samples, *self._sample_shape, device=self._device, dtype=self._fdtype)
        eps = self._expand_batch_scalar(A, G).sqrt() * G
        return (eps, A) if return_A else eps

    def _loss(self, x, z=None, t=None):
        x_1 = x.to(device=self._device, dtype=self._fdtype)
        if tuple(x_1.shape[1:]) != self._sample_shape:
            raise ValueError(f"Expected x shape (N, *{self._sample_shape}), got {tuple(x.shape)}")
        self._n = x_1.size(0)

        if t is None:
            t = torch.randint(1, self._n_steps, (self._n,), device=self._device, dtype=self._idtype)
        else:
            t = self._check_t(t, self._n)

        t_e = self._expand(t.view(1, 1, self._n)).reshape(-1)
        t_norm = self._expand((t / self._n_steps).view(1, 1, self._n)).reshape(-1, 1)
        x_1_e = x_1.view(1, 1, self._n, *self._sample_shape).expand(
            self._n_trial_A,
            self._n_trial_G,
            self._n,
            *self._sample_shape,
        ).reshape(-1, *self._sample_shape)

        eps = self._sample_source_default(self._n, kappa=self._kappa_for_t(t), expand_trials=True)
        gamma_1_t = self._expand_batch_scalar(self._gamma_1_t.index_select(0, t_e), x_1_e)
        sigma_1_t = self._expand_batch_scalar(self._sigma_1_t.index_select(0, t_e), x_1_e)
        eps_hat = self._net(gamma_1_t * x_1_e + sigma_1_t * eps, t_norm)
        return self._loss_fn(eps_hat, eps, t)

    @torch.no_grad()
    def _sample(self, n_samples, return_trajectory=True):
        self._net.eval()
        A_path = torch.stack([
            self._draw_A(n_samples, kappa=self._kappa_t[t]).squeeze(-1)
            for t in range(self._n_steps)
        ], dim=0)
        Sigma_1_t = self._Sigma_1_t(A_path)

        eps = self._sample_source_default(n_samples, kappa=self._kappa_t[-1])
        x = self._sigma_1_t[self._n_steps - 1] * eps
        l_x = [x] if return_trajectory else None

        for t in range(self._n_steps - 1, 0, -1):
            t_norm = torch.full((n_samples, 1), t / self._n_steps, device=self._device, dtype=self._fdtype)
            Sigma_hat, gamma_t, Gamma_t = self._g_Sigma_hat_Gamma(Sigma_1_t, t)
            eps_hat = self._net(x, t_norm)
            x = (x - self._expand_batch_scalar(Gamma_t, x) * self._sigma_1_t[t] * eps_hat) / gamma_t
            if t > 1:
                innovation = torch.randn(n_samples, *self._sample_shape, device=self._device, dtype=self._fdtype)
                x = x + self._expand_batch_scalar(Sigma_hat.sqrt(), innovation) * innovation
            if return_trajectory:
                l_x.append(x)
        return x, l_x


########################################################################################################################
# Main

device, dtype = setup()

dim = DIM
alpha, x_train, x_test = load_alpha_stable(device, dtype, dim=dim)

n_tail = x_test.shape[0]
n_mmd = N_MMD
n_trials = N_TRIALS
kappa_min = alpha
kappa_power = 1.0
train_kwargs = make_train_kwargs(device)
model_kwargs = dict(dim=dim, n_steps=N_STEPS, n_trial_A=1, n_trial_G=1, reduce_type="mean", device=device)
logging.info("device=%s dtype=%s dim=%s n_trials=%s n_tail=%s n_mmd=%s kappa_min=%s kappa_power=%s", device, dtype, dim, n_trials, n_tail, n_mmd, kappa_min, kappa_power)

rows = []
for trial in range(1, n_trials + 1):

    torch.manual_seed(trial - 1)
    logging.info("trial %s/%s", trial, n_trials)

    models = {
        "DLPM": DLPMEps(net=make_net(dim=dim, device=device, dtype=dtype), alpha=alpha, **model_kwargs),
        "Scheduled DLPM": ScheduledKappaDLPM(
            net=make_net(dim=dim, device=device, dtype=dtype),
            kappa_min=kappa_min,
            kappa_power=kappa_power,
            **model_kwargs,
        ),
    }

    for name, model in models.items():
        logging.info("train/evaluate %s", name)
        model, _ = train(model, x_train, **train_kwargs)
        rows.append(evaluate_model(name, model, trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

logging.info("add test-vs-true reference")
add_test_vs_true_sample(rows, x_test, n_tail, n_mmd, n_trials, alpha=alpha)


########################################################################################################################
# Plotting

save_tail_figure(rows, "Scheduled-kappa DLPM", "7_scheduled_kappa_dlpm.pdf")
logging.info("saved figure 7_scheduled_kappa_dlpm.pdf")
