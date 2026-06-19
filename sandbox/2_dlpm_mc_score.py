import torch
from _utils import (
    add_test_vs_test,
    evaluate_model,
    load_alpha_stable,
    make_mlp,
    save_tail_figure,
    setup,
)
from genkit.diffusion import DLPMEps
from genkit.training import train


device, dtype = setup()


class ZeroNet(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


class DLPMMonteCarloScoreNet(torch.nn.Module):
    def __init__(self, model, n_mc=2048, chunk_size=512):
        super().__init__()
        self.model = model
        self._n_mc = int(n_mc)
        self._chunk_size = int(chunk_size)
        if self._n_mc <= 0:
            raise ValueError("n_mc must be positive.")
        if self._chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

    @torch.no_grad()
    def forward(self, x, t):
        x = x.to(device=self.model._device, dtype=self.model._fdtype)
        t_idx = (t.reshape(-1) * self.model._n_steps).round().long()
        t_idx = t_idx.clamp(1, self.model._n_steps - 1)
        out = torch.empty_like(x)
        tiny = torch.finfo(x.dtype).tiny

        for start in range(0, x.shape[0], self._chunk_size):
            stop = min(start + self._chunk_size, x.shape[0])
            x_chunk = x[start:stop]
            x_flat = x_chunk.reshape(x_chunk.shape[0], -1)
            x2 = x_flat.square().sum(dim=1, keepdim=True)

            A = self.model._draw_A(self._n_mc).reshape(1, -1).clamp_min(tiny)
            logw = -0.5 * (self.model._dim * A.log() + x2 / A)
            weights = torch.softmax(logw, dim=1)
            score = -x_flat * (weights / A).sum(dim=1, keepdim=True)

            sigma = self.model._sigma_1_t.index_select(0, t_idx[start:stop]).reshape(-1, 1)
            eps_hat = -sigma * score
            out[start:stop] = eps_hat.reshape_as(x_chunk)

        return out


alpha, x_train, x_test = load_alpha_stable(device, dtype)
n_test = x_test.shape[0]


n_steps = 64
mc_score_samples = 4096
mc_chunk_size = 1024
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
print(f"[INFO] mc_score_samples={mc_score_samples} mc_chunk_size={mc_chunk_size}")

rows = []
for trial in range(1, n_trials + 1):

    print(f"[INFO] trial {trial}/{n_trials}: start")
    torch.manual_seed(trial - 1)
    print(f"[INFO] building models seed={trial - 1}")
    dlpm = DLPMEps(
        net=make_mlp(device, dtype),
        dim=2,
        n_steps=n_steps,
        alpha=alpha,
        n_trial_A=1,
        n_trial_G=1,
        reduce_type="mean",
        device=device,
    )
    mc_dlpm = DLPMEps(
        net=ZeroNet(),
        dim=2,
        n_steps=n_steps,
        alpha=alpha,
        n_trial_A=1,
        n_trial_G=1,
        reduce_type="mean",
        device=device,
    )
    mc_dlpm._net = DLPMMonteCarloScoreNet(
        mc_dlpm,
        n_mc=mc_score_samples,
        chunk_size=mc_chunk_size,
    ).to(device=device, dtype=dtype)
    models = {"DLPM": dlpm, "DLPM + MC score": mc_dlpm}

    for name, model in models.items():

        if name == "DLPM":
            print(f"[INFO] trial {trial}/{n_trials}: train {name}")
            model, _ = train(model, x_train, **train_kwargs)
        else:
            print(f"[INFO] trial {trial}/{n_trials}: skip training for {name}")

        print(f"[INFO] trial {trial}/{n_trials}: evaluate {name}")
        rows.append(evaluate_model(name, model, trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

    print(f"[INFO] trial {trial}/{n_trials}: done")


add_test_vs_test(rows, x_test, n_tail, n_mmd, n_trials)
save_tail_figure(rows, "DLPM Monte Carlo score", "2_dlpm_mc_score.pdf")
