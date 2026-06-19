import torch
from _utils import (
    add_test_vs_test,
    evaluate_model,
    load_alpha_stable,
    make_mlp,
    save_tail_figure,
    setup,
)
from genkit.flow_matching import GaussianFlowLinear
from genkit.training import train


device, dtype = setup()


class _HeteroscedasticNet(torch.nn.Module):
    def __init__(self, field, dim, n_noise_levels, gate_width, init_log_var, learn_scales):
        super().__init__()
        self.field = field
        self.noise_gate = torch.nn.Sequential(
            torch.nn.Linear(dim, gate_width),
            torch.nn.SiLU(),
            torch.nn.Linear(gate_width, n_noise_levels),
        )
        if learn_scales:
            self.log_var_multipliers = torch.nn.Parameter(init_log_var.clone())
        else:
            self.register_buffer("log_var_multipliers", init_log_var.clone())

    def forward(self, x, t):
        return self.field(x, t)


class HeteroscedasticGaussianFlowLinear(GaussianFlowLinear):
    def __init__(
        self,
        *args,
        n_noise_levels=4,
        init_var_multipliers=None,
        gate_width=64,
        learn_scales=True,
        temperature=1.0,
        scale_reg=1e-4,
        entropy_reg=0.0,
        ema_decay=0.99,
        min_log_var=-6.0,
        max_log_var=6.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._n_noise_levels = int(n_noise_levels)
        self._temperature = float(temperature)
        self._scale_reg = float(scale_reg)
        self._entropy_reg = float(entropy_reg)
        self._ema_decay = float(ema_decay)
        self._min_log_var = float(min_log_var)
        self._max_log_var = float(max_log_var)
        if init_var_multipliers is None:
            init_var = torch.logspace(-2.0, 2.0, self._n_noise_levels, base=2.0)
        else:
            init_var = torch.as_tensor(init_var_multipliers)
        init_log_var = init_var.to(device=self._device, dtype=self._fdtype)
        init_log_var = init_log_var.log().clamp(self._min_log_var, self._max_log_var)
        self._init_log_var_multipliers = init_log_var.detach().clone()
        self._net = _HeteroscedasticNet(
            self._net,
            self._dim,
            self._n_noise_levels,
            gate_width,
            init_log_var,
            learn_scales,
        )
        self._net.to(device=self._device, dtype=self._fdtype)
        self._noise_probs = torch.full(
            (self._n_noise_levels,),
            1.0 / self._n_noise_levels,
            device=self._device,
            dtype=self._fdtype,
        )
        self._noise_probs_ema = self._noise_probs.clone()
        self._last_entropy_penalty = torch.zeros((), device=self._device, dtype=self._fdtype)

    def _var_multipliers(self):
        return self._net.log_var_multipliers.clamp(self._min_log_var, self._max_log_var).exp()

    def _precompute_loss(self, x, z=None, t=None):
        if z is not None:
            raise ValueError("This prototype samples x0 from the data-dependent source and does not accept z.")
        x_1 = x.to(device=self._device, dtype=self._fdtype)
        if tuple(x_1.shape[1:]) != self._sample_shape:
            raise ValueError(f"Expected x1 shape (N, *{self._sample_shape}), got {tuple(x_1.shape)}")
        batch_size = x_1.shape[0]
        t = self._check_t(t, batch_size)
        logits = self._net.noise_gate(x_1.reshape(batch_size, -1))
        pi = torch.softmax(logits, dim=-1)
        y = torch.nn.functional.gumbel_softmax(logits, tau=self._temperature, hard=True, dim=-1)
        var_mult = (y * self._var_multipliers()).sum(dim=-1)
        eps = torch.randn_like(x_1)
        x_0 = self._sigma_max * self._expand_batch_scalar(torch.sqrt(var_mult.clamp_min(self._eps)), x_1) * eps
        batch_probs = pi.detach().mean(dim=0)
        self._noise_probs.copy_(batch_probs)
        self._noise_probs_ema.mul_(self._ema_decay).add_((1.0 - self._ema_decay) * batch_probs)
        self._noise_probs_ema.div_(self._noise_probs_ema.sum().clamp_min(1e-12))
        self._last_entropy_penalty = -(pi * pi.clamp_min(self._eps).log()).sum(dim=-1).mean()
        t_data = self._expand_batch_scalar(t, x_1)
        x_t = (1.0 - t_data) * x_0 + t_data * x_1
        v_t = x_1 - x_0
        v_t_hat = self._net(x_t, t)
        return v_t_hat, v_t, t

    def _sample_source_default(self, n_samples):
        probs = self._noise_probs_ema / self._noise_probs_ema.sum().clamp_min(1e-12)
        idx = torch.multinomial(probs, int(n_samples), replacement=True)
        var_mult = self._var_multipliers().index_select(0, idx)
        eps = torch.randn(n_samples, *self._sample_shape, device=self._device, dtype=self._fdtype)
        return self._sigma_max * self._expand_batch_scalar(torch.sqrt(var_mult), eps) * eps

    def loss(self, x, z=None, t=None):
        mse_loss = self._reduce(self._loss(x=x, z=z, t=t))
        scale_penalty = (self._net.log_var_multipliers - self._init_log_var_multipliers).pow(2).mean()
        return mse_loss + self._scale_reg * scale_penalty + self._entropy_reg * self._last_entropy_penalty

    @property
    def noise_std_multipliers(self):
        return torch.sqrt(self._var_multipliers()).detach().cpu()

    @property
    def noise_probs_ema(self):
        return self._noise_probs_ema.detach().cpu()


alpha, x_train, x_test = load_alpha_stable(device, dtype)
n_test = x_test.shape[0]
n_tail = n_test
n_mmd = n_test // 10
n_trials = 5
train_kwargs = dict(
    batch_size=128,
    n_epochs=32,
    lr=5e-4,
    device=device,
    use_adamw=False,
    lr_schedule="constant",
    freq_logging=8,
)


print(f"[INFO] n_trials={n_trials} train_kwargs={train_kwargs}")

rows = []
for trial in range(1, n_trials + 1):

    print(f"[INFO] trial {trial}/{n_trials}: start")
    torch.manual_seed(trial - 1)
    print(f"[INFO] building models seed={trial - 1}")
    baseline = GaussianFlowLinear(
        net=make_mlp(device, dtype),
        dim=2,
        n_steps=128,
        sigma_max=1.0,
        device=device,
    )
    hetero = HeteroscedasticGaussianFlowLinear(
        net=make_mlp(device, dtype),
        dim=2,
        n_steps=128,
        n_noise_levels=4,
        sigma_max=1.0,
        gate_width=64,
        device=device,
    )

    print(f"[INFO] trial {trial}/{n_trials}: train GF linear")
    baseline, _ = train(baseline, x_train, **train_kwargs)

    print(f"[INFO] trial {trial}/{n_trials}: train heteroscedastic GF linear")
    hetero, _ = train(hetero, x_train, **train_kwargs)

    print(f"[INFO] trial {trial}/{n_trials}: evaluate GF linear")
    rows.append(evaluate_model("GF linear", baseline, trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

    print(f"[INFO] trial {trial}/{n_trials}: evaluate heteroscedastic GF linear")
    rows.append(evaluate_model("heteroscedastic GF linear", hetero, trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

    print(f"[INFO] trial {trial}/{n_trials}: done")


print("[INFO] heteroscedastic mixture logic")

x_probe = x_train.to(device=device, dtype=dtype)

with torch.no_grad():
    logits = hetero._net.noise_gate(x_probe.reshape(x_probe.shape[0], -1))
    pi = torch.softmax(logits / hetero._temperature, dim=-1)

print("std multipliers:", hetero.noise_std_multipliers)
print("batch mean pi(x):", pi.mean(dim=0).cpu())
print("EMA generation probabilities:", hetero.noise_probs_ema)


add_test_vs_test(rows, x_test, n_tail, n_mmd, n_trials)
save_tail_figure(rows, "Heteroscedastic GF linear", "0_heteroscedastic.pdf")
