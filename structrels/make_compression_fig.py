import os

import wandb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _wandb_project_and_entity() -> tuple[str, str | None]:
    project = os.environ.get("WANDB_PROJECT", "")
    entity = os.environ.get("WANDB_ENTITY", "")
    return project, entity


def get_runs():

    # 1) Login & API init ────────────────────────────────────────────
    project, entity = _wandb_project_and_entity()

    wandb.login()
    api = wandb.Api()

    wandb.init(project=project, entity=entity)

    # 2) Fetch runs that belong to *either* tag bucket
    runs = api.runs(
        f"{entity}/{project}" if entity else project,
        filters={"tags": {"$in": ["PUT_TAGS_HERE"]}},
    )

    records = []
    for run in runs:
        summary = run.summary._json_dict
        # convenience shortcuts ------------------------------------------------
        optim_type = run.config.get("optim_type")
        with_nn    = run.config.get("with_nn")
        lr         = run.config.get("lr")
        model      = run.config.get("model")
        inner_dim_r  = run.config.get("inner_dim_r")
        inner_dim_so = run.config.get("inner_dim_s")
        target       = run.config.get("train/jacobi_train_mean")
        TAG_POOL = ["PUT_TAGS_HERE"]

        def pick_tag(run):
            for t in TAG_POOL:
                if t in run.tags:
                    return t
            return None

        tag = pick_tag(run)

        # only keep the sweep configuration of interest ------------------------
        if (
            optim_type != "SGD"
            or lr != 0.001
            or "train/tensor_train_mean" not in summary
            or model == "triangle"
        ):
            continue

        records.append(
            {
                "num_params": float(summary["num_params"]),
                "train_mean": float(summary["train/tensor_train_mean"]),
                "inner_dim_so": inner_dim_so,
                "inner_dim_r": inner_dim_r,
                "model": model,
                "with_nn": with_nn,
                "baseline": target,
                "tag": tag,
            }
        )

    return records


# ──────────────────────────────────────────────────────────────────────────────
# P L O T T I N G
# ──────────────────────────────────────────────────────────────────────────────

def plot_train_mean_scatter(records):

    df = pd.DataFrame(records)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    marker_map = {"individual_matrices": "^", "block": "s", "triangle": "^", "other": "o"}

    tags = sorted([t for t in df["tag"].dropna().unique()])
    palette = dict(zip(tags, sns.color_palette("tab10", n_colors=len(tags))))

    for with_nn in (True, False):
        for tag in tags:
            sub_tag = df[(df["with_nn"] == with_nn) & (df["tag"] == tag)]

            if sub_tag.empty:
                continue
            for model, marker in marker_map.items():
                sub = sub_tag[sub_tag["model"] == model]
                if sub.empty:
                    continue

                ax.scatter(
                    sub["num_params"],
                    sub["train_mean"],
                    marker=marker,
                    s=80,
                    linewidth=1.2,
                    facecolor=palette[tag] if with_nn else "none",
                    edgecolor="none" if with_nn else palette[tag],
                    alpha=0.95 if with_nn else 1.0,
                    zorder=2,
                )

    # Axes, baseline, ticks ----------------------------------------------------
    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters (log scale)", fontsize=14)
    ax.set_ylabel("Faithfulness", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14, length=8, width=1.5)

    # Legends ------------------------------------------------------------------
    # (A) Model shapes
    model_handles = [
        Line2D([0], [0], marker=marker_map[k], linestyle="None",
               markerfacecolor="#666666", markeredgecolor="#666666", markersize=10,
               label=lbl)
        for k, lbl in [
            ("block", "Simple 3‑way tensor network"),
            ("triangle", "Triangle tensor network"),
        ]
        if k in df["model"].unique()
    ]

    # (B) With/without extra relation embedder (fill vs. outline)
    nn_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor="#666666", markeredgecolor="#666666", markersize=10,
               label="With additional relation embedder"),
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor="none", markeredgecolor="#666666", markersize=10,
               label="Without additional relation embedder"),
    ]

    # (C) Tag colors
    tag_handles = [
        Line2D([0], [0], marker="s", linestyle="None",
               markerfacecolor=palette[t], markeredgecolor=palette[t], markersize=10,
               label=t)
        for t in tags
    ]

    # Place legends: first the model + NN legend together…
    leg1 = ax.legend(handles=(model_handles + nn_handles), loc="upper right", title=None)
    ax.add_artist(leg1)
    # …then the tag legend (place it under/left so they don't overlap)
    ax.legend(handles=tag_handles, loc="lower right", title="Tag")

    plt.grid(which="major", linestyle="-", linewidth=0.75)
    plt.grid(which="minor", linestyle="--", linewidth=0.5)
    plt.minorticks_on()
    plt.tight_layout()

    # save / log --------------------------------------------------------------
    wandb.log({"img": wandb.Image(plt)})
    plt.savefig("compression_fig.pdf", format="pdf", bbox_inches="tight")

    return fig, ax


if __name__ == "__main__":
    records = get_runs()
    plot_train_mean_scatter(records)
