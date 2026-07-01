import time
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import torch
from genkit.diffusion import DDPMV, DLPMEps
from genkit.metrics import tail_coverage_error
from genkit.training import train
from _utils import (
    evaluation_sizes,
    fetch_alpha_stable_reference,
    figure_path,
    load_alpha_stable,
    make_mlp,
    make_train_kwargs,
    setup,
)

########################################################################################################################
# Setup

device, dtype = setup()


########################################################################################################################
# Helpers

model_names = ("DLPM", "DDPM")
model_colors = {"DLPM": "tab:orange", "DDPM": "tab:purple"}
model_markers = {"DLPM": "o", "DDPM": "^"}

plot_quantile = 0.999
tail_min_prob = float(np.round(1.0 - plot_quantile, 12))
tail_quantiles = np.array([0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 0.995, plot_quantile])
map_quantiles = np.array([0.50, 0.90, 0.95, 0.99, plot_quantile])
tce_probs = torch.logspace(-1.0, np.log10(tail_min_prob), 20, dtype=torch.float64)
tce_quantiles = 1.0 - tce_probs.numpy()

n_bins = 160
min_plot_lim = 6.0
plot_lim_pad = 1.10
smooth_sigma = 2.0
pseudocount = 0.5
ratio_clip_quantile = 98.0
tail_ratio_ylim = (0.05, 1.5)


def build_model(name, alpha):
    if name == "DLPM":
        return DLPMEps(
            net=make_mlp(device, dtype),
            dim=2,
            n_steps=64,
            alpha=alpha,
            n_trial_A=1,
            n_trial_G=1,
            reduce_type="mean",
            device=device,
            fdtype=dtype,
        )
    if name == "DDPM":
        return DDPMV(
            net=make_mlp(device, dtype),
            dim=2,
            n_steps=128,
            sigma_max=5.0,
            sampler="ddpm",
            device=device,
            fdtype=dtype,
        )
    raise ValueError(f"unknown model {name!r}")


def smooth2d(count):
    radius = max(1, int(np.ceil(3.0 * smooth_sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / float(smooth_sigma)) ** 2)
    kernel /= kernel.sum()
    smoothed = count.astype(float)
    for axis in (0, 1):
        smoothed = np.apply_along_axis(lambda values: np.convolve(values, kernel, mode="same"), axis, smoothed)
    return smoothed


def tail_mass(radius, tail_radii):
    return np.array([(radius > r).mean() for r in tail_radii], dtype=float)


def tail_ratio(radius, data_tail, tail_radii, eps):
    return (tail_mass(radius, tail_radii) + eps) / (data_tail + eps)


def tce_curve(x_ref, x_gen):
    tce = tail_coverage_error(x_ref, x_gen, probs=tce_probs, tail="upper", mode="log", reduction="none")
    return tce.mean(dim=1).cpu().numpy()


def sorted_tail_arrays(ratio, ratio_std, tail_probs):
    arrays = [np.asarray(tail_probs, dtype=float), np.asarray(ratio, dtype=float), np.asarray(ratio_std, dtype=float)]
    valid = np.logical_and.reduce([np.isfinite(array) for array in arrays])
    valid &= (arrays[0] > 0.0) & (arrays[1] > 0.0)
    order = np.argsort(arrays[0][valid])[::-1]
    return [array[valid][order] for array in arrays]


def fmt_ratio(x):
    return f"{x:.2f}"


########################################################################################################################
# Main

alpha, x_train, x_test = load_alpha_stable(device, dtype)
n_tail, n_mmd = evaluation_sizes(x_test)
n_trials = 10
train_kwargs = make_train_kwargs(device)

x_ref = x_test[:n_tail].detach().cpu()
x_ref_np = x_ref.numpy()
ref_radius = np.linalg.norm(x_ref_np, axis=1)
tail_radii = np.quantile(ref_radius, tail_quantiles)
data_tail = tail_mass(ref_radius, tail_radii)
tail_probs = 1.0 - tail_quantiles
tail_ratio_eps = pseudocount / n_tail

map_lim = max(min_plot_lim, plot_lim_pad * np.quantile(ref_radius, plot_quantile))
edges = np.linspace(-map_lim, map_lim, n_bins + 1)
x_centers = 0.5 * (edges[:-1] + edges[1:])
extent = [x_centers[0], x_centers[-1], x_centers[0], x_centers[-1]]
data_count = np.histogram2d(x_ref_np[:, 0], x_ref_np[:, 1], bins=(edges, edges))[0]
data_smooth = smooth2d(data_count)
data_mass = (data_smooth + pseudocount) / (n_tail + pseudocount * data_smooth.size)
data_contours = np.where(data_smooth > 0.2, np.log10(data_mass), np.nan)
map_quantile_radii = np.quantile(ref_radius, map_quantiles)

model_counts = {name: np.zeros_like(data_count, dtype=float) for name in model_names}
tail_ratio_trials = {name: [] for name in model_names}
tce_trials = {name: [] for name in ("test vs true sample", *model_names)}
baseline_tail_ratio_trials = []

print(f"[INFO] n_trials={n_trials} n_tail={n_tail} n_mmd={n_mmd} train_kwargs={train_kwargs}")
for trial in range(1, n_trials + 1):
    print(f"[INFO] trial {trial}/{n_trials}: start")

    x_true = fetch_alpha_stable_reference(n_tail, device=torch.device("cpu"), dtype=dtype, alpha=alpha, seed=10_000 + trial)
    tce_trials["test vs true sample"].append(tce_curve(x_ref, x_true))
    baseline_tail_ratio_trials.append(
        tail_ratio(np.linalg.norm(x_true.numpy(), axis=1), data_tail, tail_radii, tail_ratio_eps)
    )

    for model_index, name in enumerate(model_names):
        torch.manual_seed(trial - 1 + 1_000 * model_index)
        model = build_model(name, alpha)

        start = time.time()
        print(f"[INFO] trial {trial}/{n_trials}: train {name}")
        model, _ = train(model, x_train, **train_kwargs)
        print(f"[INFO] trial {trial}/{n_trials}: trained {name} in {time.time() - start:.1f} s")

        torch.manual_seed(20_000 + trial + 1_000 * model_index)
        start = time.time()
        x_gen = model.sample(n_tail).detach().cpu()
        print(f"[INFO] trial {trial}/{n_trials}: sampled {name} in {time.time() - start:.1f} s")

        tce_trials[name].append(tce_curve(x_ref, x_gen))
        x_gen_np = x_gen.numpy()
        model_counts[name] += np.histogram2d(x_gen_np[:, 0], x_gen_np[:, 1], bins=(edges, edges))[0] / n_trials
        tail_ratio_trials[name].append(tail_ratio(np.linalg.norm(x_gen_np, axis=1), data_tail, tail_radii, tail_ratio_eps))

    print(f"[INFO] trial {trial}/{n_trials}: done")


########################################################################################################################
# Plotting

tce_mean = {name: np.stack(values).mean(axis=0) for name, values in tce_trials.items()}
tce_std = {name: np.stack(values).std(axis=0) for name, values in tce_trials.items()}

model_mass_ratio = {}
for name in model_names:
    model_smooth = smooth2d(model_counts[name])
    model_mass = (model_smooth + pseudocount) / (n_tail + pseudocount * model_smooth.size)
    model_mass_ratio[name] = model_mass / data_mass

finite_ratio = np.concatenate([ratio[np.isfinite(ratio) & (ratio > 0.0)] for ratio in model_mass_ratio.values()])
ratio_scale = np.nanpercentile(np.maximum(finite_ratio, 1.0 / finite_ratio), ratio_clip_quantile) if finite_ratio.size else 2.0
ratio_scale = max(float(ratio_scale), 1.1)

baseline_tail_ratio = np.stack(baseline_tail_ratio_trials)
baseline_tail_ratio_mean = baseline_tail_ratio.mean(axis=0)
baseline_tail_ratio_std = baseline_tail_ratio.std(axis=0)
model_tail_ratio_mean = {name: np.stack(tail_ratio_trials[name]).mean(axis=0) for name in model_names}
model_tail_ratio_std = {name: np.stack(tail_ratio_trials[name]).std(axis=0) for name in model_names}

fig = plt.figure(figsize=(16.4, 4), layout="constrained")
gs = fig.add_gridspec(1, 5, width_ratios=(1.0, 1.0, 0.045, 1.08, 1.08), wspace=0.14)
ax_maps = [fig.add_subplot(gs[0, i]) for i in range(2)]
cax, ax_tail, ax_tce = [fig.add_subplot(gs[0, i]) for i in range(2, 5)]
norm = mcolors.Normalize(vmin=-np.log(ratio_scale), vmax=np.log(ratio_scale), clip=True)
theta = np.linspace(0.0, 2.0 * np.pi, 512)
label_angle = np.deg2rad(35.0)

for ax, name in zip(ax_maps, model_names):
    ratio = model_mass_ratio[name]
    log_ratio = np.where(np.isfinite(ratio) & (ratio > 0.0), np.log(ratio), np.nan)
    im = ax.imshow(log_ratio.T, origin="lower", extent=extent, cmap="coolwarm", norm=norm, aspect="equal", interpolation="nearest", rasterized=True)
    ax.contour(x_centers, x_centers, data_contours.T, levels=6, colors="0.15", linewidths=0.7, alpha=0.35)
    for q, r in zip(map_quantiles, map_quantile_radii):
        if np.isfinite(r) and r <= map_lim:
            ax.plot(r * np.cos(theta), r * np.sin(theta), color="0.1", linewidth=0.8, alpha=0.55)
            ax.text(r * np.cos(label_angle), r * np.sin(label_angle), f"{100.0 * q:.2f}%", fontsize=8, ha="left", va="bottom", bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.0))
    ax.set(title=f"{name}: local mass ratio", xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]))
    ax.set_aspect("equal", adjustable="box")

tick_ratios = np.array([1.0 / ratio_scale, 1.0, ratio_scale])
cbar = fig.colorbar(im, cax=cax, ticks=np.log(tick_ratios))
cbar.ax.set_yticklabels([fmt_ratio(t) for t in tick_ratios])
cbar.ax.set_title("ratio", fontsize=10, pad=6)

tail, ratio, ratio_std = sorted_tail_arrays(baseline_tail_ratio_mean, baseline_tail_ratio_std, tail_probs)
ax_tail.fill_between(tail, np.clip(ratio - ratio_std, 1e-12, None), ratio + ratio_std, color="black", alpha=0.12, linewidth=0)
ax_tail.axhline(1.0, color="0.1", linewidth=0.9)
ax_tail.plot(tail, ratio, color="black", linestyle="--", marker="s", markersize=4, linewidth=1.6, label="test vs true sample")
for name in model_names:
    tail, ratio, ratio_std = sorted_tail_arrays(model_tail_ratio_mean[name], model_tail_ratio_std[name], tail_probs)
    color = model_colors[name]
    ax_tail.fill_between(tail, np.clip(ratio - ratio_std, 1e-12, None), ratio + ratio_std, color=color, alpha=0.18, linewidth=0)
    ax_tail.plot(tail, ratio, color=color, marker=model_markers[name], markersize=4, linewidth=1.8, label=name)
ax_tail.set(xscale="log", yscale="log", title="Radial tail mass ratio", xlabel=r"tail probability $1 - q$", ylabel="model / data mass ratio")
ax_tail.set_xlim(1.0, tail_min_prob)
ax_tail.set_ylim(*tail_ratio_ylim)
ax_tail.yaxis.set_major_locator(mticker.FixedLocator([0.05, 0.10, 0.20, 0.50, 1.00, 1.50]))
ax_tail.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_ratio(x)))
ax_tail.yaxis.set_minor_formatter(mticker.NullFormatter())
ax_tail.xaxis.set_major_locator(mticker.LogLocator(base=10))
ax_tail.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
ax_tail.xaxis.set_minor_formatter(mticker.NullFormatter())
ax_tail.grid(True, which="major", axis="both", alpha=0.28, linewidth=0.8)
ax_tail.grid(True, which="minor", axis="x", alpha=0.12, linewidth=0.6)
ax_tail.legend(frameon=False, fontsize=8)

ax_tce.plot(tce_quantiles, tce_mean["test vs true sample"], marker="s", markevery=4, linewidth=1.6, linestyle="--", color="black", label="test vs true sample")
ax_tce.fill_between(tce_quantiles, np.clip(tce_mean["test vs true sample"] - tce_std["test vs true sample"], 1e-12, None), tce_mean["test vs true sample"] + tce_std["test vs true sample"], color="black", alpha=0.12)
for name in model_names:
    color = model_colors[name]
    ax_tce.plot(tce_quantiles, tce_mean[name], marker=model_markers[name], markevery=4, linewidth=1.8, color=color, label=name)
    ax_tce.fill_between(tce_quantiles, np.clip(tce_mean[name] - tce_std[name], 1e-12, None), tce_mean[name] + tce_std[name], color=color, alpha=0.2)
ax_tce.set(xscale="logit", yscale="log", title="TCE by tail quantile", xlabel="tail quantile (%)", ylabel="TCE, upper tail log error")
ax_tce.set_xlim(0.89, plot_quantile + 2e-4)
ax_tce.set_xticks([0.90, 0.99, 0.999])
ax_tce.set_xticklabels(["90", "99", "99.9"])
ax_tce.grid(True, which="both", alpha=0.28, linewidth=0.8)
ax_tce.legend(frameon=False, fontsize=8)

path = figure_path("7_error.pdf")
fig.savefig(path, bbox_inches="tight")
plt.close(fig)
print(f"[INFO] wrote figure to {path}")
