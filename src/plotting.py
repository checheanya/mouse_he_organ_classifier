"""Training figures. SVG text stays editable, and the legend sits off the axes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from organs import COLORS, ORGAN_ORDER


def _present(organs: list[str]) -> list[str]:
    return [name for name in ORGAN_ORDER if name in organs] + [
        name for name in organs if name not in ORGAN_ORDER
    ]


def plot_counts(counts: pd.Series, path: Path) -> None:
    names = _present(list(counts.index))
    values = [int(counts.get(name, 0)) for name in names]
    colors = [COLORS.get(name, COLORS["unknown"]) for name in names]
    fig_h = max(3.2, 0.42 * len(names) + 1.2)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    ypos = np.arange(len(names))
    ax.barh(ypos, values, color=colors, height=0.72, zorder=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    span = max(values) if values else 1
    ax.set_xlim(0, span * 1.18)
    for y, value in zip(ypos, values):
        ax.text(value + span * 0.02, y, str(value), va="center", ha="left", fontsize=9, color="#333333")
    ax.set_xlabel("Tissue pieces")
    ax.set_title("Annotated pieces in the training set")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def plot_umap(embedding: np.ndarray, organs: list[str], path: Path) -> None:
    names = _present(sorted(set(organs)))
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    for name in names:
        mask = np.array([item == name for item in organs])
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=18,
            c=COLORS.get(name, COLORS["unknown"]),
            alpha=0.85,
            linewidths=0,
            label=name,
            zorder=2,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Mean-pooled UNI features by annotated organ")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        title="Organ",
        borderaxespad=0.0,
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    fig.savefig(path, format="svg")
    plt.close(fig)


def plot_confusion(matrix: np.ndarray, names: list[str], path: Path) -> None:
    fig_w = max(6.5, 0.55 * len(names) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Annotated")
    ax.set_title("Leave-group-out predictions")
    peak = float(matrix.max()) if matrix.size else 1.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            if value == 0:
                continue
            color = "white" if value > 0.55 * peak else "#1d1d1d"
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8, color=color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.set_ylabel("Pieces", rotation=90, va="bottom")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def legend_handles(names: list[str]) -> list[Line2D]:
    ordered = _present(names)
    return [
        Line2D([0], [0], color=COLORS.get(name, COLORS["unknown"]), lw=2.2, label=name)
        for name in ordered
    ]
