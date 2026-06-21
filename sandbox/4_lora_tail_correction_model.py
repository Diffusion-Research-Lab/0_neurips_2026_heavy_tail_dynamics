import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from _utils import (
    add_test_vs_test,
    evaluate_model_with_source,
    load_alpha_stable,
    make_mlp,
    save_tail_figure,
    setup,
)
from genkit.flow_matching import GaussianFlowLinear
from genkit.nn import MLPModel
from genkit.training import train


device, dtype = setup()


class LoRALinear(nn.Module):
    def __init__(self, linear, rank=8, alpha=1.0):
        super().__init__()
        rank = int(rank)
        if rank <= 0:
            raise ValueError("rank must be positive.")
        if float(alpha) <= 0.0:
            raise ValueError("alpha must be positive.")

        self.linear = linear
        self.A = nn.Linear(linear.in_features, rank, bias=False)
        self.B = nn.Linear(rank, linear.out_features, bias=False)
        self.scale = float(alpha) / rank
        for param in self.linear.parameters():
            param.requires_grad_(False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x, gate):
        correction = self.scale * self.B(self.A(x))
        while gate.ndim < correction.ndim:
            gate = gate.unsqueeze(-1)
        return self.linear(x) + gate * correction


class GatedLoRABlock(nn.Module):
    def __init__(self, block, rank, alpha):
        super().__init__()
        self.norm = block.norm
        self.fc1 = LoRALinear(block.fc1, rank=rank, alpha=alpha)
        self.fc2 = LoRALinear(block.fc2, rank=rank, alpha=alpha)
        self.tproj = block.tproj
        self.drop = block.drop
        self.act = block.act

    def forward(self, x, temb, gate):
        h = self.fc1(self.norm(x), gate) + self.tproj(temb)
        h = self.fc2(self.drop(self.act(h)), gate)
        return x + h


class GatedLoRAMLP(nn.Module):
    def __init__(self, net, rank=8, alpha=1.0, gate_tau=1.0, gate_sharpness=5.0):
        super().__init__()
        if not isinstance(net, MLPModel):
            raise TypeError(f"GatedLoRAMLP expects MLPModel, got {type(net).__name__}.")
        if float(gate_sharpness) <= 0.0:
            raise ValueError("gate_sharpness must be positive.")

        net = copy.deepcopy(net)
        for param in net.parameters():
            param.requires_grad_(False)

        self.input_dim = net.input_dim
        self.output_dim = net.output_dim
        self.time_dim = net.time_dim
        self.gate_tau = float(gate_tau)
        self.gate_sharpness = float(gate_sharpness)
        self.inp = LoRALinear(net.inp, rank=rank, alpha=alpha)
        self.time = net.time
        self.blocks = nn.ModuleList(GatedLoRABlock(block, rank=rank, alpha=alpha) for block in net.blocks)
        self.out = LoRALinear(net.out, rank=rank, alpha=alpha)
        self.act = net.act

    def forward(self, x, t):
        x_flat = x.reshape(x.shape[0], -1)
        if x_flat.shape[1] != self.input_dim:
            raise ValueError(f"Expected flattened input dimension {self.input_dim}, got {x_flat.shape[1]}.")
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        elif t.ndim != 1:
            raise ValueError(f"Expected t with shape [B] or [B, 1], got {t.shape}.")

        gate = torch.sigmoid(self.gate_sharpness * (x_flat.norm(dim=1, keepdim=True) - self.gate_tau))
        temb = MLPModel._timestep_embedding(t, self.time_dim).to(dtype=x.dtype, device=x.device)
        h = self.act(self.inp(x_flat, gate))
        temb = self.time(temb)
        for block in self.blocks:
            h = block(h, temb, gate)
        return self.out(h, gate).reshape(x.shape[0], self.output_dim)


class TailLoRAFlow(GaussianFlowLinear):
    def __init__(
        self,
        base,
        tau,
        rank=8,
        alpha=1.0,
        loss_tail_weight=2.0,
        loss_sharpness=4.0,
        gate_tau=None,
        gate_sharpness=4.0,
    ):
        if float(loss_tail_weight) <= 0.0:
            raise ValueError("loss_tail_weight must be positive.")
        if float(loss_sharpness) <= 0.0:
            raise ValueError("loss_sharpness must be positive.")
        if gate_tau is None:
            gate_tau = tau

        super().__init__(
            GatedLoRAMLP(base._net, rank=rank, alpha=alpha, gate_tau=gate_tau, gate_sharpness=gate_sharpness),
            dim=base._sample_shape,
            n_steps=base._n_steps,
            t_min=base._t_min,
            t_max=base._t_max,
            sigma_max=base._sigma_max,
            sampler=base._sampler,
            schedule=base._schedule,
            sample_steps=base._sample_steps,
            device=base._device,
            fdtype=base._fdtype,
            idtype=base._idtype,
        )
        self.tau = float(tau)
        self.loss_tail_weight = float(loss_tail_weight)
        self.loss_sharpness = float(loss_sharpness)
        self.gate_tau = float(gate_tau)
        self.gate_sharpness = float(gate_sharpness)

    def _loss(self, x, z=None, t=None):
        x_0, x_1, t = self._latent(x_1=x, x_0=z, t=t)
        t_data = self._expand_batch_scalar(t, x_0)
        x_t = (1.0 - t_data) * x_0 + t_data * x_1
        v_t = x_1 - x_0
        v_t_hat = self._net(x_t, t)
        if v_t_hat.shape != v_t.shape:
            raise ValueError(f"Network output has shape {tuple(v_t_hat.shape)}, expected {tuple(v_t.shape)}.")

        state_norm = x_t.flatten(1).norm(dim=1, keepdim=True)
        weight = 1.0 + (self.loss_tail_weight - 1.0) * torch.sigmoid(self.loss_sharpness * (state_norm - self.tau))
        return weight.view(-1, *([1] * (x.ndim - 1))) * F.mse_loss(v_t_hat, v_t, reduction="none")


def build_base_model():
    return GaussianFlowLinear(
        net=make_mlp(device, dtype),
        dim=2,
        n_steps=128,
        t_min=0.0,
        t_max=1.0,
        sigma_max=1.0,
        sampler="euler",
        sample_steps=128,
        device=device,
    )


alpha, x_train, x_test = load_alpha_stable(device, dtype)
n_test = x_test.shape[0]
n_tail = n_test
n_mmd = n_test // 10
n_trials = 5
sample_chunk_size = 10_000
train_kwargs = dict(
    batch_size=128,
    n_epochs=32,
    lr=5e-4,
    device=device,
    use_adamw=False,
    lr_schedule="constant",
    freq_logging=8,
)
lora_specs = [
    {"name": "gLoRA q90", "tau_q": 0.90, "gate_q": 0.90, "rank": 6, "alpha": 4.0, "loss_tail_weight": 2.0, "sharpness": 5.0},
]

tail_norms = x_train.flatten(1).norm(dim=1)
tail_quantiles = sorted({spec["tau_q"] for spec in lora_specs} | {spec["gate_q"] for spec in lora_specs})
tail_taus = {q: torch.quantile(tail_norms, q).item() for q in tail_quantiles}
print(f"[INFO] n_trials={n_trials} train_kwargs={train_kwargs}")
print(f"[INFO] sample_chunk_size={sample_chunk_size}")
print("[INFO] tail norm quantiles:", {q: round(value, 4) for q, value in tail_taus.items()})

rows = []
for trial in range(1, n_trials + 1):
    print(f"[INFO] trial {trial}/{n_trials}: start")
    torch.manual_seed(trial - 1)
    print(f"[INFO] building models seed={trial - 1}")
    models = {"GF linear": build_base_model()}

    print(f"[INFO] trial {trial}/{n_trials}: train GF linear")
    models["GF linear"], _ = train(models["GF linear"], x_train, **train_kwargs)

    for spec in lora_specs:
        name = spec["name"]
        start_time = time.time()
        print(f"[INFO] trial {trial}/{n_trials}: train {name}")
        models[name] = TailLoRAFlow(
            models["GF linear"],
            tau=tail_taus[spec["tau_q"]],
            rank=spec["rank"],
            alpha=spec["alpha"],
            loss_tail_weight=spec["loss_tail_weight"],
            loss_sharpness=spec["sharpness"],
            gate_tau=tail_taus[spec["gate_q"]],
            gate_sharpness=spec["sharpness"],
        )
        models[name], _ = train(models[name], x_train, **train_kwargs)
        print(f"[INFO] trial {trial}/{n_trials}: trained {name} in {time.time() - start_time:.1f} s")

    x_source = models["GF linear"]._sample_source(n_tail)
    for name, model in models.items():
        rows.append(evaluate_model_with_source(name, model, trial, x_test, x_source, n_tail, n_mmd, chunk_size=sample_chunk_size))

    print(f"[INFO] trial {trial}/{n_trials}: done")


add_test_vs_test(rows, x_test, n_tail, n_mmd, n_trials)
save_tail_figure(rows, "Gated LoRA tail correction", "4_lora_tail_correction_model.pdf")
