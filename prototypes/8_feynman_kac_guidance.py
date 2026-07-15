import logging
from pathlib import Path
import torch
from genkit.diffusion import DDPMV, DLPMEps
from genkit.flow_matching import GaussianFlowLinear
from genkit.training import train
from _constants import ALPHA, DIM, FLOW_N_STEPS, FLOW_SAMPLE_STEPS, N_MMD, N_STEPS, N_TRIALS, SAMPLE_CHUNK_SIZE
from _guidance import Guidance, make_guidance_dataset
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

DDPM_SIGMA_MAX = 5.0
GUIDANCE_BATCH_SIZE = 128
GUIDANCE_N_SAMPLES = 10_000
GUIDANCE_N_ROLLOUTS = 16
GUIDANCE_EPOCHS = 128
GUIDANCE_LR = 5e-4
GUIDANCE_TEMPERATURE = 5.0
GUIDANCE_STRENGTHS = (10.0,)
GUIDANCE_BASE_N_SAMPLES = 100_000
MAX_ABS_REWARD = 5.0


########################################################################################################################
# Additional classes and functions


class GuidedSampler:
    def __init__(self, guidance: Guidance, strength: float):
        self.guidance = guidance
        self.strength = float(strength)

    def sample(self, n_samples: int):
        return self.guidance.sample(n_samples, strength=self.strength)


def sample_model(model, n_samples, chunk_size=SAMPLE_CHUNK_SIZE):
    chunks = []
    for start in range(0, n_samples, chunk_size):
        chunks.append(model.sample(min(chunk_size, n_samples - start)).detach().cpu())
    return torch.cat(chunks, dim=0)


def make_tail_ratio_reward(target_radius, base_samples, device):
    target_radius = target_radius.detach().sort().values.to(device)
    base_radius = base_samples.detach().reshape(base_samples.shape[0], -1).norm(dim=1).sort().values.to(device)
    eps = 0.5 / min(len(target_radius), len(base_radius))

    def empirical_survival(sorted_radius, radius):
        sorted_radius = sorted_radius.to(device=radius.device, dtype=radius.dtype)
        index = torch.searchsorted(sorted_radius, radius.contiguous(), right=False)
        return ((len(sorted_radius) - index).to(radius.dtype) / len(sorted_radius)).clamp_min(eps)

    def reward(x):
        radius = x.reshape(x.shape[0], -1).norm(dim=1)
        target_survival = empirical_survival(target_radius, radius)
        base_survival = empirical_survival(base_radius, radius)
        return (target_survival.log() - base_survival.log()).clamp(-MAX_ABS_REWARD, MAX_ABS_REWARD)

    return reward


########################################################################################################################
# Main

device, dtype = setup()

dim = DIM
alpha, x_train, x_test = load_alpha_stable(device, dtype, dim=dim)

n_tail = x_test.shape[0]
n_mmd = N_MMD
n_trials = N_TRIALS
target_radius = x_train.flatten(1).norm(dim=1)
train_kwargs = make_train_kwargs(device)
dlpm_kwargs = dict(alpha=alpha, dim=dim, n_steps=N_STEPS, n_trial_A=1, n_trial_G=1, reduce_type="mean", device=device)
flow_kwargs = dict(dim=dim, n_steps=FLOW_N_STEPS, t_min=0.0, t_max=1.0, sigma_max=1.0, sampler="euler", sample_steps=FLOW_SAMPLE_STEPS, device=device)
ddpm_kwargs = dict(dim=dim, n_steps=N_STEPS, sigma_max=DDPM_SIGMA_MAX, sampler="ddpm", device=device, fdtype=dtype)
logging.info("device=%s dtype=%s dim=%s n_trials=%s n_tail=%s n_mmd=%s guidance_samples=%s guidance_rollouts=%s guidance_strengths=%s", device, dtype, dim, n_trials, n_tail, n_mmd, GUIDANCE_N_SAMPLES, GUIDANCE_N_ROLLOUTS, GUIDANCE_STRENGTHS)

rows = []
for trial in range(1, n_trials + 1):

    torch.manual_seed(trial - 1)
    logging.info("trial %s/%s", trial, n_trials)

    logging.info("train/evaluate DDPM")
    ddpm = DDPMV(net=make_net(dim=dim, device=device, dtype=dtype), **ddpm_kwargs)
    ddpm, _ = train(ddpm, x_train, **train_kwargs)
    rows.append(evaluate_model("DDPM", ddpm, trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

    logging.info("train/evaluate DLPM")
    dlpm = DLPMEps(net=make_net(dim=dim, device=device, dtype=dtype), **dlpm_kwargs)
    dlpm, _ = train(dlpm, x_train, **train_kwargs)
    rows.append(evaluate_model("DLPM", dlpm, trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

    logging.info("train/evaluate GFL")
    gfl = GaussianFlowLinear(net=make_net(dim=dim, device=device, dtype=dtype), **flow_kwargs)
    gfl, _ = train(gfl, x_train, **train_kwargs)
    rows.append(evaluate_model("GFL", gfl, trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

    logging.info("fit guidance")
    ddpm_reward_samples = sample_model(ddpm, min(GUIDANCE_BASE_N_SAMPLES, n_tail))
    guidance = Guidance(ddpm, make_net(dim=dim, output_dim=1, device=device, dtype=dtype))
    guidance_dataset = make_guidance_dataset(
        ddpm,
        make_tail_ratio_reward(target_radius, ddpm_reward_samples, device),
        temperature=GUIDANCE_TEMPERATURE,
        n_samples=GUIDANCE_N_SAMPLES,
        n_rollouts=GUIDANCE_N_ROLLOUTS,
        batch_size=GUIDANCE_BATCH_SIZE,
    )
    guidance.fit(guidance_dataset, epochs=GUIDANCE_EPOCHS, batch_size=GUIDANCE_BATCH_SIZE, lr=GUIDANCE_LR)

    for strength in GUIDANCE_STRENGTHS:
        name = f"DDPM + guidance (strength={strength:g})"
        logging.info("evaluate %s", name)
        rows.append(evaluate_model(name, GuidedSampler(guidance, strength), trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

logging.info("add test-vs-true reference")
add_test_vs_true_sample(rows, x_test, n_tail, n_mmd, n_trials, alpha=alpha)


########################################################################################################################
# Plotting

save_tail_figure(rows, "Feynman-Kac DDPM guidance", "8_feynman_kac_guidance.pdf")
logging.info("saved figure 8_feynman_kac_guidance.pdf")
