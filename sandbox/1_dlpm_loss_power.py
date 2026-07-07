import logging
from pathlib import Path
import torch
from genkit.diffusion import DLPMEps
from genkit.training import train
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

class DLPMEpsPowerLoss(DLPMEps):
    def __init__(self, *args, loss_power=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_power = float(loss_power)

    def _loss_fn(self, eps_hat, eps, t):
        loss_values = torch.nn.functional.mse_loss(eps_hat, eps, reduction="none")
        loss_values = loss_values.mean(dim=tuple(range(1, loss_values.ndim)))
        return loss_values.clamp_min(self._eps).pow(self._loss_power)


########################################################################################################################
# Main
device, dtype = setup()

dim = 15
alpha, x_train, x_test = load_alpha_stable(device, dtype, dim=dim)

n_tail = x_test.shape[0]
n_mmd = 10_000
n_trials = 10
loss_powers = [0.1, 0.25, 0.5, 0.75]
train_kwargs = make_train_kwargs(device)
logging.info("device=%s dtype=%s dim=%s n_trials=%s n_tail=%s n_mmd=%s loss_powers=%s", device, dtype, dim, n_trials, n_tail, n_mmd, loss_powers)

rows = []
for trial in range(1, n_trials + 1):

    torch.manual_seed(trial - 1)
    logging.info("trial %s/%s", trial, n_trials)

    models = {}
    for reduce_type in ["mean", "median"]:
        for r in loss_powers:
            model = DLPMEpsPowerLoss(net=make_net(dim=dim, device=device, dtype=dtype), dim=dim, n_steps=64, alpha=alpha,
                                     n_trial_A=1, n_trial_G=1, reduce_type=reduce_type, loss_power=r,
                                     device=device)
            models[f"DLPM r={r:g} reduce_type={reduce_type}"] = model

    for name, model in models.items():
        logging.info("train/evaluate %s", name)
        model, _ = train(model, x_train, **train_kwargs)
        rows.append(evaluate_model(name, model, trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

logging.info("add test-vs-true reference")
add_test_vs_true_sample(rows, x_test, n_tail, n_mmd, n_trials, alpha=alpha)

########################################################################################################################
# Plotting

save_tail_figure(rows, "DLPM loss power", "1_dlpm_loss_power.pdf")
logging.info("saved figure 1_dlpm_loss_power.pdf")
