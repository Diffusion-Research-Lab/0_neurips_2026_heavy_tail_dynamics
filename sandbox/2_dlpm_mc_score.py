import time
import torch
from _utils import (
    setup,
)
from genkit._noise import sample_scaled_isotropic_alpha_stable
from genkit.diffusion import DLPMEps
from genkit.metrics import mmd_rbf, tail_coverage_error
from genkit.nn import MLPModel
from genkit.training import train

########################################################################################################################
# Setup

device, dtype = setup()


########################################################################################################################
# Helpers

alpha = 1.7
dim = 2
n_steps = 64
n_train = 2_048
n_eval = 4_096
n_mmd = 2_048
n_mc = 2_048
batch_size = 256
n_epochs = 16
chunk_size = 512
tce_probs = torch.tensor([1e-1, 5e-2, 1e-2], dtype=torch.float64)


class ConditionalMonteCarloEpsNet(torch.nn.Module):
    """MC estimator of E[eps | x_t] for x_t = gamma_t x_0 + sigma_t eps."""

    def __init__(self, model, n_mc=n_mc, chunk_size=chunk_size):
        super().__init__()
        self.model = model
        self.n_mc = int(n_mc)
        self.chunk_size = int(chunk_size)

    @torch.no_grad()
    def forward(self, x, t):
        x = x.to(device=self.model._device, dtype=self.model._fdtype)
        t_idx = (t.reshape(-1).to(device=self.model._device) * self.model._n_steps).round().long()
        if t_idx.numel() == 1:
            t_idx = t_idx.expand(x.shape[0])
        t_idx = t_idx.clamp(1, self.model._n_steps - 1)

        out = torch.empty_like(x)
        tiny = torch.finfo(x.dtype).tiny
        for start in range(0, x.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, x.shape[0])
            x_chunk = x[start:stop]
            x_flat = x_chunk.reshape(x_chunk.shape[0], -1)
            x2 = x_flat.square().sum(dim=1, keepdim=True)

            idx = t_idx[start:stop]
            gamma = self.model._gamma_1_t.index_select(0, idx).reshape(-1, 1)
            sigma = self.model._sigma_1_t.index_select(0, idx).reshape(-1, 1)
            A_data = self.model._draw_A(self.n_mc).reshape(1, -1).clamp_min(tiny)
            A_noise = self.model._draw_A(self.n_mc).reshape(1, -1).clamp_min(tiny)
            variance = (gamma.square() * A_data + sigma.square() * A_noise).clamp_min(tiny)

            logw = -0.5 * (self.model._dim * variance.log() + x2 / variance)
            weights = torch.softmax(logw, dim=1)
            coeff = (weights * sigma * A_noise / variance).sum(dim=1, keepdim=True)
            out[start:stop] = (coeff * x_flat).reshape_as(x_chunk)

        return out


def sample_target(n_samples, seed):
    torch.manual_seed(seed)
    return sample_scaled_isotropic_alpha_stable(
        n_samples=n_samples,
        dim=dim,
        alpha=alpha,
        device=device,
        dtype=dtype,
    )


def make_dlpm(net):
    return DLPMEps(
        net=net,
        dim=dim,
        n_steps=n_steps,
        alpha=alpha,
        n_trial_A=1,
        n_trial_G=1,
        reduce_type="mean",
        fdtype=dtype,
        device=device,
    )


@torch.no_grad()
def evaluate_samples(name, x_ref, x_gen):
    x_ref = x_ref.detach().cpu().to(torch.float64)
    x_gen = x_gen.detach().cpu().to(torch.float64)
    values = {"mmd_rbf": mmd_rbf(x_ref[:n_mmd], x_gen[:n_mmd])}
    for prob in tce_probs:
        q = 100.0 * (1.0 - float(prob))
        values[f"tce_q{q:.1f}"] = tail_coverage_error(
            x_ref,
            x_gen,
            probs=prob.reshape(1),
            tail="upper",
            mode="log",
        )
    print(f"[EVAL] {name}: " + " ".join(f"{key}={value:.3g}" for key, value in values.items()))


########################################################################################################################
# Main

print(
    f"[INFO] alpha={alpha} dim={dim} n_steps={n_steps} n_train={n_train} "
    f"n_eval={n_eval} n_mc={n_mc}"
)
x_train = sample_target(n_train, seed=0)
x_ref = sample_target(n_eval, seed=10_000)
x_test = sample_target(n_eval, seed=20_000)
evaluate_samples("test vs test", x_ref, x_test)

torch.manual_seed(1)
dlpm = make_dlpm(MLPModel(dim=dim, width=64, depth=2).to(device=device, dtype=dtype))
start = time.perf_counter()
dlpm, stats = train(
    dlpm,
    x_train,
    batch_size=batch_size,
    n_epochs=n_epochs,
    lr=5e-4,
    device=device,
    use_adamw=False,
    lr_schedule="constant",
    freq_logging=0,
)
elapsed = time.perf_counter() - start
final_loss = stats["stats"]["training_loss"][-1]
print(f"[TRAIN] DLPM done in {elapsed:.1f}s final_loss={final_loss:.4f}")

########################################################################################################################
# Evaluation

start = time.perf_counter()
torch.manual_seed(30_000)
evaluate_samples("DLPM trained net", x_ref, dlpm.sample(n_eval))
print(f"[TIME] DLPM sampling+eval {time.perf_counter() - start:.1f}s")

mc_dlpm = make_dlpm(torch.nn.Identity())
mc_dlpm._net = ConditionalMonteCarloEpsNet(mc_dlpm)
start = time.perf_counter()
torch.manual_seed(30_000)
evaluate_samples("DLPM conditional MC", x_ref, mc_dlpm.sample(n_eval))
print(f"[TIME] conditional MC sampling+eval {time.perf_counter() - start:.1f}s")
