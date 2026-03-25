"""Benchmark 5 figure creation from saved results."""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from genkit.plotting import PRETTY_RCPARAMS
from _utils import (average_trial_curve, format_mean_std_latex, loss_linestyle,
                    resolve_run_dir, save_figure, source_color)


BENCHMARK_NAME = "bench_5"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("_results"))
    parser.add_argument("--fig-root", type=Path, default=Path("_figures"))
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    plt.rcParams.update(PRETTY_RCPARAMS)

    run_dir = resolve_run_dir(args.out_root, BENCHMARK_NAME, args.run_dir)
    fig_dir = args.fig_root / BENCHMARK_NAME / run_dir.name
    fig_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))

    models = payload.get("models", [])
    if not models:
        raise ValueError("results.json does not contain any models.")
    if not all("trials" in model for model in models):
        raise ValueError(
            "results.json is missing per-trial entries. Re-run benchmark 05 with the updated experiment script."
        )
    if not all("WASS" in model.get("metrics", {}) for model in models):
        raise ValueError(
            "results.json is missing the WASS metric. Re-run benchmark 05 with the updated experiment script."
        )

    source_order = list(dict.fromkeys(model["source_distri"] for model in models))
    loss_order = list(dict.fromkeys(model["loss"] for model in models))
    loss_alpha_values = list(dict.fromkeys(float(model["loss_alpha"]) for model in models))

    source_colors = {source_name: source_color(source_name) for source_name in source_order}
    loss_ls = {alpha_value: loss_linestyle(alpha_value) for alpha_value in loss_alpha_values}

    max_loss_epochs = max(len(average_trial_curve(model["trials"], "training_loss")) for model in models)
    max_grad_epochs = max(len(average_trial_curve(model["trials"], "grad_variance_epoch")) for model in models)
    fig, axis = plt.subplots(1, 2, figsize=(5.0, 2.5), squeeze=False)
    for model in models:
        avg_loss = average_trial_curve(model["trials"], "training_loss")
        avg_grad_var = average_trial_curve(model["trials"], "grad_variance_epoch")
        if avg_loss.size == 0 or avg_grad_var.size == 0:
            continue
        tt_loss = np.linspace(1, len(avg_loss), len(avg_loss), dtype=int)
        tt_grad = np.linspace(1, len(avg_grad_var), len(avg_grad_var), dtype=int)
        color = source_colors[model["source_distri"]]
        linestyle = loss_ls[float(model["loss_alpha"])]
        axis[0, 0].plot(tt_loss, avg_loss, color=color, ls=linestyle, lw=2.0, alpha=0.7)
        axis[0, 1].plot(tt_grad, avg_grad_var, color=color, ls=linestyle, lw=2.0, alpha=0.7)

    axis[0, 0].set_xscale("log")
    axis[0, 0].set_yscale("log")
    axis[0, 0].set_xlim(1, max_loss_epochs)
    axis[0, 0].set_xticks([1, max_loss_epochs], ["1", f"{max_loss_epochs}"])
    axis[0, 0].set_xlabel("Epochs")
    axis[0, 0].set_ylabel("Loss")

    axis[0, 1].set_yscale("log")
    axis[0, 1].set_xlim(1, max_grad_epochs)
    axis[0, 1].set_xticks([1, max_grad_epochs], ["1", f"{max_grad_epochs}"])
    axis[0, 1].set_xlabel("Epochs")
    axis[0, 1].set_ylabel("Var. Grad.")

    legend_handles = [
        Line2D([0], [0], color=color, lw=3.0, label=source_name)
        for source_name, color in source_colors.items()
    ]
    legend_handles += [
        Line2D(
            [0],
            [0],
            color="black",
            lw=3.0,
            ls=loss_ls[alpha_value],
            label=rf"$\alpha = {int(alpha_value) if float(alpha_value).is_integer() else alpha_value:g}$",
        )
        for alpha_value in loss_alpha_values
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=max(2, min(4, len(legend_handles))),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1.0, 0.83))
    for saved_path in save_figure(fig, fig_dir, "training_curves"):
        print(f"[INFO] Saved figure artifact: {saved_path}")
    plt.close(fig)

    rows = []
    for model in models:
        row = {
            "source_distri": model["source_distri"],
            "loss": model["loss"],
            "loss_alpha": float(model["loss_alpha"]),
            "MSSLE_95": float(model["metrics"]["MSSLE_95"]["mean"]),
            "MSSLE_95_std": float(model["metrics"]["MSSLE_95"]["std"]),
            "WASS": float(model["metrics"]["WASS"]["mean"]),
            "WASS_std": float(model["metrics"]["WASS"]["std"]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["source_distri"] = pd.Categorical(df["source_distri"], categories=source_order, ordered=True)
    df["loss"] = pd.Categorical(df["loss"], categories=loss_order, ordered=True)
    df = df.sort_values(["source_distri", "loss_alpha"]).reset_index(drop=True)

    caption_items = {
        "(a) MSSLE(95)": ("MSSLE_95", "MSSLE_95_std"),
        "(b) Wasserstein distance": ("WASS", "WASS_std"),
    }
    tables = {}
    for caption, (mean_key, std_key) in caption_items.items():
        mean_df = df.pivot_table(
            index="source_distri",
            columns="loss",
            values=mean_key,
            aggfunc="mean",
            observed=False,
        )
        std_df = df.pivot_table(
            index="source_distri",
            columns="loss",
            values=std_key,
            aggfunc="mean",
            observed=False,
        )
        tables[caption] = {
            "mean": mean_df.reindex(index=source_order, columns=loss_order),
            "std": std_df.reindex(index=source_order, columns=loss_order),
        }

    fig, axis = plt.subplots(len(tables), 1, figsize=(1.8 * len(loss_order), 2.8), squeeze=False)
    for i, (caption, table_stats) in enumerate(tables.items()):
        axis[i, 0].axis("off")
        mean_df = table_stats["mean"]
        std_df = table_stats["std"]
        row_min_mask = mean_df.eq(mean_df.min(axis=1), axis=0)
        table_min_mask = mean_df.eq(mean_df.min().min())
        formatted = mean_df.copy().astype(object)
        for row_label in mean_df.index:
            for col_label in mean_df.columns:
                mean = mean_df.loc[row_label, col_label]
                std = std_df.loc[row_label, col_label]
                is_best = bool(table_min_mask.loc[row_label, col_label]) if pd.notna(mean) else False
                formatted.loc[row_label, col_label] = (
                    "-"
                    if pd.isna(mean)
                    else format_mean_std_latex(mean, std, bold=is_best)
                )

        table = axis[i, 0].table(
            cellText=formatted.values,
            rowLabels=formatted.index,
            colLabels=formatted.columns,
            cellLoc="center",
            rowLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.35, 1.6)

        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0.5)
            cell.set_text_props(alpha=0.5)
            if row == 0 or col == -1:
                cell.set_text_props(weight="bold", alpha=1.0)
            elif row > 0 and col >= 0 and row_min_mask.iloc[row - 1, col]:
                cell.set_text_props(alpha=1.0)

        axis[i, 0].text(
            0.5,
            -0.25,
            caption,
            transform=axis[i, 0].transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.08, hspace=0.8)
    for saved_path in save_figure(fig, fig_dir, "summary_tables"):
        print(f"[INFO] Saved figure artifact: {saved_path}")
    plt.close(fig)
