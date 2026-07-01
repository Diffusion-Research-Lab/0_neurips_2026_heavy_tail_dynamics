import torch
from _utils import (
    add_test_vs_true_sample,
    evaluation_sizes,
    evaluate_model,
    load_alpha_stable,
    make_mlp,
    make_train_kwargs,
    save_tail_figure,
    setup,
)
from genkit.flow_matching import GaussianFlowLinear
from genkit.training import train


########################################################################################################################
# Setup

device, dtype = setup()


########################################################################################################################
# Additional classes

class HistogramLevels2D:
    def __init__(self, x, n_levels=4, n_bins=80, range_quantile=0.995):
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"Expected 2D samples with shape (N, 2), got {tuple(x.shape)}.")
        self.n_levels = int(n_levels)
        self.n_bins = int(n_bins)
        if self.n_levels <= 0 or self.n_bins <= 1:
            raise ValueError("n_levels must be positive and n_bins must be greater than one.")
        q = (1.0 - float(range_quantile)) / 2.0
        lo = torch.quantile(x, q, dim=0)
        hi = torch.quantile(x, 1.0 - q, dim=0)
        span = (hi - lo).clamp_min(torch.finfo(x.dtype).eps)
        lo, hi = lo - 1e-3 * span, hi + 1e-3 * span
        self.x_edges = torch.linspace(lo[0], hi[0], self.n_bins + 1, device=x.device, dtype=x.dtype)
        self.y_edges = torch.linspace(lo[1], hi[1], self.n_bins + 1, device=x.device, dtype=x.dtype)

        ix, iy = self.cell_indices(x)
        counts = torch.bincount(ix * self.n_bins + iy, minlength=self.n_bins**2).reshape(self.n_bins, self.n_bins).float()
        point_counts = counts[ix, iy]
        thresholds = torch.quantile(point_counts, torch.linspace(0.0, 1.0, self.n_levels + 1, device=x.device))
        self.cell_levels = torch.bucketize(counts.reshape(-1), thresholds[1:-1], right=False).reshape(self.n_bins, self.n_bins).long()
        levels = self.cell_levels[ix, iy]
        self.level_probs = torch.bincount(levels, minlength=self.n_levels).float()
        self.level_probs /= self.level_probs.sum().clamp_min(1.0)

    def to(self, device=None, dtype=None):
        self.x_edges = self.x_edges.to(device=device, dtype=dtype)
        self.y_edges = self.y_edges.to(device=device, dtype=dtype)
        self.cell_levels = self.cell_levels.to(device=device)
        self.level_probs = self.level_probs.to(device=device, dtype=dtype)
        return self

    def cell_indices(self, x):
        ix = torch.bucketize(x[:, 0], self.x_edges[1:-1]).clamp(0, self.n_bins - 1)
        iy = torch.bucketize(x[:, 1], self.y_edges[1:-1]).clamp(0, self.n_bins - 1)
        return ix.long(), iy.long()

    def assign(self, x):
        ix, iy = self.cell_indices(x)
        return self.cell_levels[ix, iy]


class HeteroscedasticGaussianFlowLinear(GaussianFlowLinear):
    def __init__(
        self,
        *args,
        levels,
        init_var_multipliers=None,
        learn_scales=True,
        scale_reg=1e-4,
        min_log_var=-6.0,
        max_log_var=6.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.levels = levels.to(device=self._device, dtype=self._fdtype)
        self._n_noise_levels = int(levels.n_levels)
        self._scale_reg = float(scale_reg)
        self._min_log_var = float(min_log_var)
        self._max_log_var = float(max_log_var)
        if init_var_multipliers is None:
            init_var = torch.ones(self._n_noise_levels)
        else:
            init_var = torch.as_tensor(init_var_multipliers)
        init_log_var = init_var.to(device=self._device, dtype=self._fdtype)
        init_log_var = init_log_var.log().clamp(self._min_log_var, self._max_log_var)
        self._init_log_var_multipliers = init_log_var.detach().clone()
        if learn_scales:
            self.log_var_multipliers = torch.nn.Parameter(init_log_var.clone())
        else:
            self.register_buffer("log_var_multipliers", init_log_var.clone())

    def _var_multipliers(self):
        return self.log_var_multipliers.clamp(self._min_log_var, self._max_log_var).exp()

    def _precompute_loss(self, x, z=None, t=None):
        if z is not None:
            raise ValueError("This prototype samples x0 from the histogram source and does not accept z.")
        x_1 = x.to(device=self._device, dtype=self._fdtype)
        if tuple(x_1.shape[1:]) != self._sample_shape:
            raise ValueError(f"Expected x1 shape (N, *{self._sample_shape}), got {tuple(x_1.shape)}")
        batch_size = x_1.shape[0]
        t = self._check_t(t, batch_size)
        level = self.levels.assign(x_1)
        var_mult = self._var_multipliers().index_select(0, level)
        eps = torch.randn_like(x_1)
        x_0 = self._sigma_max * self._expand_batch_scalar(torch.sqrt(var_mult.clamp_min(self._eps)), x_1) * eps
        t_data = self._expand_batch_scalar(t, x_1)
        x_t = (1.0 - t_data) * x_0 + t_data * x_1
        v_t = x_1 - x_0
        v_t_hat = self._net(x_t, t)
        return v_t_hat, v_t, t

    def _sample_source_default(self, n_samples):
        probs = self.levels.level_probs / self.levels.level_probs.sum().clamp_min(1e-12)
        idx = torch.multinomial(probs, int(n_samples), replacement=True)
        var_mult = self._var_multipliers().index_select(0, idx)
        eps = torch.randn(n_samples, *self._sample_shape, device=self._device, dtype=self._fdtype)
        return self._sigma_max * self._expand_batch_scalar(torch.sqrt(var_mult), eps) * eps

    def loss(self, x, z=None, t=None):
        mse_loss = self._reduce(self._loss(x=x, z=z, t=t))
        scale_penalty = (self.log_var_multipliers - self._init_log_var_multipliers).pow(2).mean()
        return mse_loss + self._scale_reg * scale_penalty

    @property
    def noise_std_multipliers(self):
        return torch.sqrt(self._var_multipliers()).detach().cpu()

    @property
    def noise_level_probs(self):
        return self.levels.level_probs.detach().cpu()


########################################################################################################################
# Main

alpha, x_train, x_test = load_alpha_stable(device, dtype)
n_tail, n_mmd = evaluation_sizes(x_test)
n_trials = 10
n_noise_levels = 4
train_kwargs = make_train_kwargs(device)


levels = HistogramLevels2D(x_train, n_levels=n_noise_levels, n_bins=80)

print(f"[INFO] n_trials={n_trials} n_noise_levels={n_noise_levels} train_kwargs={train_kwargs}")
print("[INFO] fixed histogram level probabilities:", levels.level_probs.cpu())

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
        levels=levels,
        sigma_max=1.0,
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


print("[INFO] fixed histogram heteroscedastic source")
print("std multipliers:", hetero.noise_std_multipliers)
print("level probabilities:", hetero.noise_level_probs)


########################################################################################################################
# Plotting

add_test_vs_true_sample(rows, x_test, n_tail, n_mmd, n_trials, alpha=alpha)
save_tail_figure(rows, "Heteroscedastic GF linear", "0_heteroscedastic.pdf")
