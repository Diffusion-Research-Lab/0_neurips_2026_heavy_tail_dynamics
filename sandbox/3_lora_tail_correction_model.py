import copy
import math
import torch
from genkit.flow_matching import GaussianFlowLinear
from genkit.nn import MLPModel
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


########################################################################################################################
# Additional classes

class LoRALinear(torch.nn.Module):
    def __init__(self, linear, rank=6, alpha=4.0):
        super().__init__()
        self.linear = linear
        self.scale = float(alpha) / int(rank)
        self.down = torch.nn.Linear(linear.in_features, int(rank), bias=False)
        self.up = torch.nn.Linear(int(rank), linear.out_features, bias=False)
        for param in self.linear.parameters():
            param.requires_grad_(False)
        torch.nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, x, gate):
        return self.linear(x) + gate * self.scale * self.up(self.down(x))


class TailLoRAMLP(torch.nn.Module):
    def __init__(self, net, tau, rank=6, alpha=4.0, sharpness=5.0):
        super().__init__()
        if not isinstance(net, MLPModel):
            raise TypeError(f"TailLoRAMLP expects MLPModel, got {type(net).__name__}.")

        net = copy.deepcopy(net)
        for param in net.parameters():
            param.requires_grad_(False)

        self.input_dim = net.input_dim
        self.output_dim = net.output_dim
        self.time_dim = net.time_dim
        self.tau = float(tau)
        self.sharpness = float(sharpness)
        self.inp = LoRALinear(net.inp, rank=rank, alpha=alpha)
        self.time = net.time
        self.blocks = net.blocks
        self.out = LoRALinear(net.out, rank=rank, alpha=alpha)
        self.act = net.act

        for block in self.blocks:
            block.fc1 = LoRALinear(block.fc1, rank=rank, alpha=alpha)
            block.fc2 = LoRALinear(block.fc2, rank=rank, alpha=alpha)

    def forward(self, x, t):
        x_flat = x.reshape(x.shape[0], -1)
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]

        gate = torch.sigmoid(self.sharpness * (x_flat.norm(dim=1, keepdim=True) - self.tau))
        temb = MLPModel._timestep_embedding(t, self.time_dim).to(dtype=x.dtype, device=x.device)
        temb = self.time(temb)
        h = self.act(self.inp(x_flat, gate))

        for block in self.blocks:
            residual = h
            h = block.fc1(block.norm(h), gate) + block.tproj(temb)
            h = block.fc2(block.drop(block.act(h)), gate)
            h = residual + h

        return self.out(h, gate).reshape(x.shape[0], self.output_dim)


class TailLoRAFlow(GaussianFlowLinear):
    def __init__(self, base, tau, rank=6, alpha=4.0, sharpness=5.0, tail_weight=2.0):
        super().__init__(
            net=TailLoRAMLP(base._net, tau=tau, rank=rank, alpha=alpha, sharpness=sharpness),
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
        self.sharpness = float(sharpness)
        self.tail_weight = float(tail_weight)

    def _loss(self, x, z=None, t=None):
        x_0, x_1, t = self._latent(x_1=x, x_0=z, t=t)
        t_data = self._expand_batch_scalar(t, x_0)
        x_t = (1.0 - t_data) * x_0 + t_data * x_1
        v_t = x_1 - x_0
        v_t_hat = self._net(x_t, t)

        radius = x_t.flatten(1).norm(dim=1)
        weight = 1.0 + (self.tail_weight - 1.0) * torch.sigmoid(self.sharpness * (radius - self.tau))
        loss = torch.nn.functional.mse_loss(v_t_hat, v_t, reduction="none")
        return self._expand_batch_scalar(weight, loss) * loss


########################################################################################################################
# Main

device, dtype = setup()

dim = 15
alpha, x_train, x_test = load_alpha_stable(device, dtype, dim=dim)

n_tail = x_test.shape[0]
n_mmd = 10_000
n_trials = 10
tail_quantile = 0.90
train_kwargs = make_train_kwargs(device)
flow_kwargs = dict(dim=dim, n_steps=128, t_min=0.0, t_max=1.0, sigma_max=1.0, sampler="euler", sample_steps=128, device=device)
tau = torch.quantile(x_train.flatten(1).norm(dim=1), tail_quantile).item()

rows = []
for trial in range(1, n_trials + 1):

    torch.manual_seed(trial - 1)

    baseline = GaussianFlowLinear(net=make_net(dim=dim, device=device, dtype=dtype), **flow_kwargs)
    baseline, _ = train(baseline, x_train, **train_kwargs)
    rows.append(evaluate_model("GF linear", baseline, trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

    lora = TailLoRAFlow(baseline, tau=tau, rank=6, alpha=4.0, sharpness=5.0, tail_weight=2.0)
    lora, _ = train(lora, x_train, **train_kwargs)
    rows.append(evaluate_model("GF linear + tail LoRA", lora, trial, x_test, n_tail=n_tail, n_mmd=n_mmd))

add_test_vs_true_sample(rows, x_test, n_tail, n_mmd, n_trials, alpha=alpha)


########################################################################################################################
# Plotting

save_tail_figure(rows, "Tail LoRA GF linear", "3_lora_tail_correction_model.pdf")
