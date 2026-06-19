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
from genkit.flow_matching import GaussianFlowLinear
from genkit.training import train


device, dtype = setup()


class DLPMEpsPowerLoss(DLPMEps):
    def __init__(self, *args, loss_power=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_power = float(loss_power)
        if self._loss_power <= 0.0:
            raise ValueError(f"loss_power must be positive, got {loss_power}.")

    def _loss_fn(self, eps_hat, eps, t):
        loss_values = torch.nn.functional.mse_loss(eps_hat, eps, reduction="none")
        loss_values = loss_values.mean(dim=tuple(range(1, loss_values.ndim)))
        return loss_values.clamp_min(self._eps).pow(self._loss_power)

alpha, x_train, x_test = load_alpha_stable(device, dtype)
n_test = x_test.shape[0]


loss_powers = [0.1, 0.25, 0.5]
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


print(f"[INFO] n_trials={n_trials} loss_powers={loss_powers} train_kwargs={train_kwargs}")

rows = []
for trial in range(1, n_trials + 1):

    print(f"[INFO] trial {trial}/{n_trials}: start")
    torch.manual_seed(trial - 1)
    print(f"[INFO] building models seed={trial - 1}")
    models = {
        "GF linear": GaussianFlowLinear(
            net=make_mlp(device, dtype),
            dim=2,
            n_steps=128,
            sigma_max=1.0,
            device=device,
        ),
    }
    for r in loss_powers:
        models[f"DLPM r={r:g}"] = DLPMEpsPowerLoss(
            net=make_mlp(device, dtype),
            dim=2,
            n_steps=64,
            alpha=alpha,
            n_trial_A=1,
            n_trial_G=1,
            reduce_type="mean",
            loss_power=r,
            device=device,
        )

    for name, model in models.items():
        print(f"[INFO] trial {trial}/{n_trials}: train {name}")
        model, _ = train(model, x_train, **train_kwargs)

        print(f"[INFO] trial {trial}/{n_trials}: evaluate {name}")
        rows.append(evaluate_model(name, model, trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

    print(f"[INFO] trial {trial}/{n_trials}: done")


add_test_vs_test(rows, x_test, n_tail, n_mmd, n_trials)
save_tail_figure(rows, "DLPM loss power", "1_dlpm_loss_power.pdf")
