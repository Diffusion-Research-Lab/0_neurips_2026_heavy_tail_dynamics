import logging
from pathlib import Path
import torch
from genkit.diffusion import DLPMEps, DDPMV
from genkit.training import train
from genkit._noise import sample_scaled_scalar_alpha_stable
from _constants import N_MMD, N_STEPS, N_TRIALS
from _utils import (
    add_test_vs_true_sample,
    make_net,
    evaluate_model,
    load_alpha_stable,
    make_train_kwargs,
    save_tail_figure,
    setup,
)

log_path = Path(__file__).with_name("logs") / f"{Path(__file__).stem}.log"
log_path.parent.mkdir(exist_ok=True)
logging.basicConfig(filename=log_path, filemode="w", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MC_DIM = 2
MC_N_TRAIN = 5_000
MCDLPM_N_MC = 500_000
MCDLPM_MC_CHUNK_SIZE = 5_000
MCDDPM_N_MC = 1_000

########################################################################################################################
# Additional classes


class MCDLPMEps(DLPMEps):
    """DLPM-Eps sampler using a path-conditioned Monte Carlo eps estimate."""

    def __init__(self, *args, n_mc=1_000, mc_chunk_size=1_000, **kwargs):
        kwargs["net"] = torch.nn.Identity()
        super().__init__(*args, **kwargs)
        self.n_mc = int(n_mc)
        self.mc_chunk_size = int(mc_chunk_size)
        if self.n_mc <= 0:
            raise ValueError(f"n_mc must be positive, got {self.n_mc}.")
        if self.mc_chunk_size <= 0:
            raise ValueError(f"mc_chunk_size must be positive, got {self.mc_chunk_size}.")

    def _mc_eps_coeff(self, observed_radius_sq, path_noise_var, gamma_1_t, sigma_1_t):
        running_max = torch.full_like(path_noise_var, -torch.inf)
        running_den = torch.zeros_like(path_noise_var)
        running_num = torch.zeros_like(path_noise_var)

        for start in range(0, self.n_mc, self.mc_chunk_size):
            size = min(self.mc_chunk_size, self.n_mc - start)
            data_var = sample_scaled_scalar_alpha_stable(size, alpha=self._a, device=self._device, dtype=torch.float64)
            data_var = data_var.reshape(1, -1)
            total_var = gamma_1_t.square() * data_var + path_noise_var
            log_weight = -0.5 * (self._dim * total_var.log() + observed_radius_sq / total_var)
            coeff = path_noise_var / (sigma_1_t * total_var)

            block_max = log_weight.max(dim=1, keepdim=True).values
            new_max = torch.maximum(running_max, block_max)
            old_scale = torch.exp(running_max - new_max)
            block_scale = torch.exp(log_weight - new_max)

            running_den = running_den * old_scale + block_scale.sum(dim=1, keepdim=True)
            running_num = running_num * old_scale + (block_scale * coeff).sum(dim=1, keepdim=True)
            running_max = new_max

        return running_num / running_den.clamp_min(torch.finfo(running_den.dtype).tiny)

    @torch.no_grad()
    def _sample(
        self,
        n_samples: int,
        return_trajectory: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Run the DLPM reverse chain with path-conditioned MC eps estimates."""
        A_path = []
        for _ in range(self._n_steps):
            a_path = sample_scaled_scalar_alpha_stable(n_samples, alpha=self._a, device=self._device, dtype=torch.float64)
            A_path.append(a_path.squeeze(-1))
        A_path = torch.stack(A_path, dim=0)

        Sigma_1_t = self._Sigma_1_t(A_path)
        x = self._sigma_1_t[self._n_steps - 1] * self._sample_source_default(n_samples)
        l_x = [x] if return_trajectory else None

        for t in range(self._n_steps - 1, 0, -1):
            Sigma_hat, gamma_t, Gamma_t = self._g_Sigma_hat_Gamma(Sigma_1_t, t)
            Sigma_hat = Sigma_hat.to(torch.float64)
            gamma_t = gamma_t.to(torch.float64)
            Gamma_t = Gamma_t.to(torch.float64)

            # Conditional model at time t: x_t = gamma_{1,t} x_0 + sigma_{1,t} eps_t,
            # with x_0 | A_data ~ N(0, A_data I) and sigma_{1,t} eps_t | A_path ~ N(0, path_noise_var I).
            x_flat = x.reshape(n_samples, -1).to(torch.float64)
            observed_radius_sq = x_flat.square().sum(dim=1, keepdim=True)

            path_noise_var = Sigma_1_t[t].reshape(-1, 1)
            gamma_1_t = self._gamma_1_t[t].to(torch.float64)
            sigma_1_t = self._sigma_1_t[t].to(torch.float64)

            # E[eps_t | x_t, A_data, A_path] = path_noise_var / (sigma_{1,t} total_var) x_t.
            eps_coeff = self._mc_eps_coeff(observed_radius_sq, path_noise_var, gamma_1_t, sigma_1_t)
            eps_hat = (eps_coeff * x_flat).reshape_as(x).to(dtype=x.dtype)

            gamma_t_data = self._expand_batch_scalar(Gamma_t.to(dtype=x.dtype), x)
            x = (x - gamma_t_data * sigma_1_t.to(dtype=x.dtype) * eps_hat) / gamma_t.to(dtype=x.dtype)

            if t > 1:
                innovation = torch.randn(n_samples, *self._sample_shape, device=self._device, dtype=self._fdtype)
                x = x + self._expand_batch_scalar(Sigma_hat.to(dtype=innovation.dtype).sqrt(), innovation) * innovation

            if return_trajectory:
                l_x.append(x)

        return x, l_x


class MCDDPMV(DDPMV):
    """DDPM-V sampler using a Monte Carlo eps estimate under the alpha-stable data prior."""

    def __init__(self, *args, alpha=1.7, n_mc=1_000, **kwargs):
        kwargs["net"] = torch.nn.Identity()
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.n_mc = int(n_mc)

    @torch.no_grad()
    def _sample(
        self,
        n_samples: int,
        return_trajectory: bool = True,
        sample_source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        n_samples, x = self._resolve_sample_source(n_samples, sample_source)
        l_x = [x] if return_trajectory else None
        source_var = torch.tensor(self._sigma_max**2, device=x.device, dtype=torch.float64)

        for t in range(self._n_steps, 0, -1):
            t_idx = t - 1
            alpha_bar_t = self._alpha_bar[t_idx].to(torch.float64)
            one_minus_alpha_bar_t = 1.0 - alpha_bar_t

            # Conditional model at time t: x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps,
            # with x_0 | A_data ~ N(0, A_data I) and eps ~ N(0, sigma_max^2 I).
            x_flat = x.reshape(n_samples, -1).to(torch.float64)
            observed_radius_sq = x_flat.square().sum(dim=1, keepdim=True)

            data_var = sample_scaled_scalar_alpha_stable(self.n_mc, alpha=self.alpha, device=self._device,
                                                         dtype=torch.float64)
            data_var = data_var.reshape(1, -1)
            noise_var = one_minus_alpha_bar_t * source_var
            total_var = alpha_bar_t * data_var + noise_var
            log_p_xt_given_data_var = -0.5 * (self._dim * total_var.log() + observed_radius_sq / total_var)
            data_var_weight = torch.softmax(log_p_xt_given_data_var, dim=1)

            # E[eps | x_t, A_data] = sqrt(1 - alpha_bar_t) sigma_max^2 / total_var x_t.
            eps_coeff_given_data_var = one_minus_alpha_bar_t.sqrt() * source_var / total_var
            eps_coeff = (data_var_weight * eps_coeff_given_data_var).sum(dim=1, keepdim=True)
            eps_hat = (eps_coeff * x_flat).reshape_as(x).to(dtype=x.dtype)

            scale = (self._betas[t_idx].to(torch.float64) / one_minus_alpha_bar_t.sqrt()).to(dtype=x.dtype)
            alpha_t_sqrt = self._alphas[t_idx].to(torch.float64).sqrt().to(dtype=x.dtype)
            x = (x - scale * eps_hat) / alpha_t_sqrt

            if t > 1:
                x = x + self._sigma_max * self._sqrt_post_var[t_idx] * torch.randn_like(x)

            if return_trajectory:
                l_x.append(x)

        return x, l_x


########################################################################################################################
# Main
device, dtype = setup()

dim = MC_DIM
alpha, x_train, x_test = load_alpha_stable(device, dtype, dim=dim)
x_train = x_train[:MC_N_TRAIN]

n_tail = x_test.shape[0]
n_mmd = N_MMD
n_trials = N_TRIALS
n_steps = N_STEPS
ddpm_sigma_max = 5.0
train_kwargs = make_train_kwargs(device)
logging.info("device=%s dtype=%s dim=%s n_train=%s n_trials=%s n_tail=%s n_mmd=%s n_steps=%s mcdlpm_n_mc=%s mcdlpm_mc_chunk_size=%s mcddpm_n_mc=%s",
             device, dtype, dim, len(x_train), n_trials, n_tail, n_mmd, n_steps, MCDLPM_N_MC, MCDLPM_MC_CHUNK_SIZE, MCDDPM_N_MC)

rows = []
for trial in range(1, n_trials + 1):

    torch.manual_seed(trial - 1)
    logging.info("trial %s/%s", trial, n_trials)

    logging.info("train/evaluate DLPM")
    dlpm = DLPMEps(net=make_net(dim=dim, device=device, dtype=dtype), alpha=alpha, dim=dim, n_steps=n_steps, n_trial_A=1, n_trial_G=1, reduce_type="mean", device=device)
    dlpm, _ = train(dlpm, x_train, **train_kwargs)
    rows.append(evaluate_model("DLPM", dlpm, trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

    logging.info("evaluate DLPM (+MC)")
    mc_dlpm = MCDLPMEps(alpha=alpha, dim=dim, n_steps=n_steps, n_trial_A=1, n_trial_G=1, reduce_type="mean",
                        n_mc=MCDLPM_N_MC, mc_chunk_size=MCDLPM_MC_CHUNK_SIZE, device=device)
    rows.append(evaluate_model("DLPM (+MC)", mc_dlpm, trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

    logging.info("train/evaluate DDPM")
    ddpm = DDPMV(net=make_net(dim=dim, device=device, dtype=dtype), dim=dim, n_steps=n_steps, sigma_max=ddpm_sigma_max, sampler="ddpm", device=device, fdtype=dtype)
    ddpm, _ = train(ddpm, x_train, **train_kwargs)
    rows.append(evaluate_model("DDPM", ddpm, trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

    logging.info("evaluate DDPM (+MC)")
    mc_ddpm = MCDDPMV(alpha=alpha, dim=dim, n_steps=n_steps, sigma_max=ddpm_sigma_max, sampler="ddpm", n_mc=MCDDPM_N_MC, device=device, fdtype=dtype)
    rows.append(evaluate_model("DDPM (+MC)", mc_ddpm, trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

logging.info("add test-vs-true reference")
add_test_vs_true_sample(rows, x_test, n_tail, n_mmd, n_trials, alpha=alpha)

########################################################################################################################
# Plotting

save_tail_figure(rows, "DLPM MC estimates", "2_dlpm_mc_score.pdf")
logging.info("saved figure 2_dlpm_mc_score.pdf")
