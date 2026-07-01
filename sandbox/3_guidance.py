import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flowbench-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from _utils import (
    add_test_vs_true_sample,
    evaluation_sizes,
    evaluate_model_with_source,
    figure_path,
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

class GuidedGaussianFlowLinear:
    def __init__(
        self,
        flow,
        tail_radius,
        tail_scale,
        lambda_schedule=None,
        guidance_scale=1.0,
    ):
        self.flow = flow.flow if isinstance(flow, GuidedGaussianFlowLinear) else flow
        if not isinstance(self.flow, GaussianFlowLinear):
            raise TypeError("flow must be a trained GaussianFlowLinear or GuidedGaussianFlowLinear.")

        self.tail_radius = float(tail_radius)
        self.tail_scale = float(tail_scale)
        if self.tail_scale <= 0.0:
            raise ValueError("tail_scale must be positive.")
        self.lambda_schedule = lambda_schedule
        self.guidance_scale = float(guidance_scale)

    def __getattr__(self, name):
        if "flow" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.flow, name)

    def drift(self, x, t):
        with torch.no_grad():
            velocity = self.flow._net(x, t)

        if self.guidance_scale == 0.0:
            return velocity

        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            norm = x_req.flatten(1).norm(dim=1)
            tail_potential = torch.nn.functional.softplus((norm - self.tail_radius) / self.tail_scale)
            grad = torch.autograd.grad(tail_potential.sum(), x_req)[0]

        lbda = 1.0 if self.lambda_schedule is None else self.lambda_schedule(t)
        lbda = torch.as_tensor(lbda, device=x.device, dtype=x.dtype)
        while lbda.ndim < x.ndim:
            lbda = lbda.unsqueeze(-1)
        return velocity + self.guidance_scale * lbda * grad

    @torch.no_grad()
    def _sample(self, n_samples=None, return_trajectory=True, sample_source=None, chunk_size=None):
        self.flow._net.eval()
        _, x = self.flow._resolve_sample_source(n_samples, sample_source)

        if chunk_size is not None and int(chunk_size) < x.size(0):
            outputs, trajectory_chunks = [], None
            for x_chunk in x.split(int(chunk_size)):
                x_out, trajectory = self._sample(return_trajectory=return_trajectory, sample_source=x_chunk)
                outputs.append(x_out)
                if return_trajectory:
                    if trajectory_chunks is None:
                        trajectory_chunks = [[] for _ in trajectory]
                    for chunks_at_time, value in zip(trajectory_chunks, trajectory):
                        chunks_at_time.append(value)
            x_out = torch.cat(outputs, dim=0)
            trajectory_out = None if trajectory_chunks is None else [torch.cat(chunks, dim=0) for chunks in trajectory_chunks]
            return x_out, trajectory_out

        timesteps = torch.linspace(
            self.flow._t_min,
            self.flow._t_max,
            self.flow._sample_steps + 1,
            device=self.flow._device,
            dtype=self.flow._fdtype,
        )
        trajectory = [x.detach()] if return_trajectory else None
        for t0, t1 in zip(timesteps[:-1], timesteps[1:]):
            t = t0.reshape(1, 1).expand(x.size(0), 1)
            x = x + (t1 - t0) * self.drift(x, t)
            if trajectory is not None:
                trajectory.append(x.detach())
        return x, trajectory

    @torch.no_grad()
    def sample(self, n_samples=None, sample_source=None, chunk_size=None):
        x, _ = self._sample(n_samples=n_samples, return_trajectory=False, sample_source=sample_source, chunk_size=chunk_size)
        return x


def build_models(x_train, train_kwargs, tail_radius, tail_scale):
    base = GaussianFlowLinear(
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
    print("[INFO] train GFL")
    base, diagnostics = train(base, x_train, **train_kwargs)
    models = {"GF linear": base}
    for scale in [0.5, 1.0]:
        models[f"guided GF scale={scale:g}"] = GuidedGaussianFlowLinear(
            base,
            tail_radius=tail_radius,
            tail_scale=tail_scale,
            guidance_scale=scale,
        )
    return models, diagnostics


########################################################################################################################
# Plotting helpers

def collect_visuals(models, target_preview, *, lim=4.0, n_vis=120, n_trails=120, n_grid=15):
    sample_source = next(iter(models.values()))._sample_source(n_vis)
    visuals = {}
    for name, model in models.items():
        grid_axis = torch.linspace(-lim, lim, n_grid, device=model._device, dtype=model._fdtype)
        field_points = torch.cartesian_prod(grid_axis, grid_axis)
        _, trajectory_steps = model._sample(sample_source=sample_source)

        trajectory = torch.stack(trajectory_steps).detach()
        flow_times = torch.linspace(model._t_min, model._t_max, len(trajectory_steps), device=model._device, dtype=model._fdtype)
        field_times = [t.expand(field_points.shape[0], 1) for t in flow_times]

        if isinstance(model, GuidedGaussianFlowLinear):
            field_vectors = torch.stack([model.drift(field_points, t) for t in field_times]).detach()
            with torch.no_grad():
                base_vectors = torch.stack([model.flow._net(field_points, t) for t in field_times]).detach()
        else:
            with torch.no_grad():
                field_vectors = torch.stack([model._net(field_points, t) for t in field_times]).detach()
            base_vectors = field_vectors

        correction_vectors = field_vectors - base_vectors
        visuals[name] = {
            "trajectory": trajectory.cpu().numpy(),
            "field_points": field_points.cpu().numpy(),
            "field_vectors": field_vectors.cpu().numpy(),
            "flow_times": flow_times.cpu().numpy(),
            "correction_vectors": correction_vectors.cpu().numpy(),
            "correction_norm": float(correction_vectors.norm(dim=-1).mean().item()),
        }
    return visuals, target_preview.detach().cpu().numpy(), n_trails


def save_guidance_gif(models, x_test):
    print("[INFO] rendering guidance animation")
    visuals, target_np, n_trails = collect_visuals(models, x_test[:3_000])
    n_frames = min(data["trajectory"].shape[0] for data in visuals.values())
    model_items = list(visuals.items())
    row_labels = ["trajectory", "vector field", "correction"]
    lim = 4.0

    fig, axes = plt.subplots(
        len(row_labels),
        len(model_items),
        figsize=(4 * len(model_items), 4 * len(row_labels)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    panels = []
    for col, (name, data) in enumerate(model_items):
        trajectory = data["trajectory"]
        field_points = data["field_points"]
        field_vectors = data["field_vectors"]
        correction_vectors = data["correction_vectors"]
        flow_times = data["flow_times"]
        trail_idx = np.unique(np.linspace(0, trajectory.shape[1] - 1, min(n_trails, trajectory.shape[1]), dtype=int))

        trajectory_ax, field_ax, correction_ax = axes[:, col]
        trajectory_ax.scatter(target_np[:, 0], target_np[:, 1], s=7, color="0.6", alpha=0.2, linewidths=0)
        trail_lines = LineCollection([], colors="tab:blue", linewidths=2.0, alpha=0.1)
        trajectory_ax.add_collection(trail_lines)
        sample_cloud = trajectory_ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1], s=11, color="tab:blue", alpha=0.4, linewidths=0)

        field_arrows = field_ax.quiver(
            field_points[:, 0],
            field_points[:, 1],
            field_vectors[0, :, 0],
            field_vectors[0, :, 1],
            angles="xy",
            scale_units="xy",
            scale=5,
            width=0.002,
            color="tab:blue",
            alpha=0.4,
        )
        correction_arrows = correction_ax.quiver(
            field_points[:, 0],
            field_points[:, 1],
            correction_vectors[0, :, 0],
            correction_vectors[0, :, 1],
            angles="xy",
            scale_units="xy",
            scale=5,
            width=0.002,
            color="tab:red",
            alpha=0.5,
        )
        correction_ax.text(
            0.02,
            0.98,
            f"mean |correction| = {data['correction_norm']:.2f}",
            transform=correction_ax.transAxes,
            ha="left",
            va="top",
        )
        time_texts = [ax.text(0.02, 0.90, "", transform=ax.transAxes, ha="left", va="top") for ax in axes[:, col]]
        axes[0, col].set_title(name)
        panels.append((trajectory, field_vectors, correction_vectors, flow_times, trail_idx, trail_lines, sample_cloud, field_arrows, correction_arrows, time_texts))

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label)
    for ax in axes.ravel():
        ax.set(aspect="equal", xlim=(-2 * lim, 2 * lim), ylim=(-2 * lim, 2 * lim))
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()

    def update(frame):
        artists = []
        for panel in panels:
            trajectory, field_vectors, correction_vectors, flow_times = panel[:4]
            trail_idx, trail_lines, sample_cloud, field_arrows, correction_arrows, time_texts = panel[4:]
            i = int(frame)
            trail_lines.set_segments([trajectory[: i + 1, j, :] for j in trail_idx])
            sample_cloud.set_offsets(trajectory[i])
            field_arrows.set_UVC(field_vectors[i, :, 0], field_vectors[i, :, 1])
            correction_arrows.set_UVC(correction_vectors[i, :, 0], correction_vectors[i, :, 1])
            for time_text in time_texts:
                time_text.set_text(f"t = {flow_times[i]:.3f}")
            artists.extend([trail_lines, sample_cloud, field_arrows, correction_arrows, *time_texts])
        return artists

    anim = FuncAnimation(fig, update, frames=n_frames, interval=80, repeat=True, blit=False)
    path = figure_path("3_guidance.gif")
    anim.save(path, writer="pillow", fps=12)
    plt.close(fig)
    print(f"[INFO] wrote animation to {path}")


########################################################################################################################
# Main

alpha, x_train, x_test = load_alpha_stable(device, dtype)
n_tail, n_mmd = evaluation_sizes(x_test)
n_trials = 10
sample_chunk_size = 10_000
tail_radius = float(torch.quantile(x_train.flatten(1).norm(dim=1), 0.90).item())
tail_scale = 0.25 * tail_radius
train_kwargs = make_train_kwargs(device)

print(f"[INFO] n_trials={n_trials} train_kwargs={train_kwargs}")
print(f"[INFO] tail_radius={tail_radius:.4g} tail_scale={tail_scale:.4g} sample_chunk_size={sample_chunk_size}")

rows = []
last_models = None
for trial in range(1, n_trials + 1):
    print(f"[INFO] trial {trial}/{n_trials}: start")
    torch.manual_seed(trial - 1)
    print(f"[INFO] building models seed={trial - 1}")
    start_time = time.time()
    models, _ = build_models(x_train, train_kwargs, tail_radius, tail_scale)
    print(f"[INFO] trial {trial}/{n_trials}: trained in {time.time() - start_time:.1f} s")

    x_source = models["GF linear"]._sample_source(n_tail)
    for name, model in models.items():
        rows.append(evaluate_model_with_source(name, model, trial, x_test, x_source, n_tail, n_mmd, chunk_size=sample_chunk_size))

    last_models = models
    print(f"[INFO] trial {trial}/{n_trials}: done")


########################################################################################################################
# Plotting

add_test_vs_true_sample(rows, x_test, n_tail, n_mmd, n_trials, alpha=alpha)
save_tail_figure(rows, "Tail guidance", "3_guidance.pdf")
save_guidance_gif(last_models, x_test)
