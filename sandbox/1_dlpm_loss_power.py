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
from genkit.diffusion import DLPMEps
from genkit.training import train


########################################################################################################################
# Setup

device, dtype = setup()


########################################################################################################################
# Additional classes

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


########################################################################################################################
# Main

alpha, x_train, x_test = load_alpha_stable(device, dtype)
n_tail, n_mmd = evaluation_sizes(x_test)
loss_powers = [0.1, 0.25, 0.5, 0.75]
n_trials = 10
train_kwargs = make_train_kwargs(device)


print(f"[INFO] n_trials={n_trials} loss_powers={loss_powers} train_kwargs={train_kwargs}")

rows = []
for trial in range(1, n_trials + 1):

    print(f"[INFO] trial {trial}/{n_trials}: start")
    torch.manual_seed(trial - 1)
    print(f"[INFO] building models seed={trial - 1}")
    models = {}
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


########################################################################################################################
# Plotting

add_test_vs_true_sample(rows, x_test, n_tail, n_mmd, n_trials, alpha=alpha)
save_tail_figure(rows, "DLPM loss power", "1_dlpm_loss_power.pdf")
