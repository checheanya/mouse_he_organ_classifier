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

MASTER = Path("/data/labs/sinha_lab/achechenina/data/261015_master_organ_metadata.csv")

SEX_ORDER = ("Male", "Female", "unknown")
SEX_COLORS = {
    "Male": "#5b8fa8",
    "Female": "#ba7999",
    "unknown": "#b7b1a8",
}


def _present(organs: list[str]) -> list[str]:
    return [name for name in ORGAN_ORDER if name in organs] + [
        name for name in organs if name not in ORGAN_ORDER
    ]


def _normalize_sex(value) -> str:
    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value).strip()
    if text in {"M", "male", "Male"}:
        return "Male"
    if text in {"F", "female", "Female"}:
        return "Female"
    return "unknown"


def attach_sex(examples: pd.DataFrame, master_path: Path = MASTER) -> pd.DataFrame:
    """Add a ``sex`` column. Blank or uncertain values become ``unknown``."""
    frame = examples.drop(columns=["sex"], errors="ignore").copy()
    master = pd.read_csv(master_path, dtype=str)
    by_slide = (
        master[["slide_id", "sex"]]
        .dropna(subset=["slide_id"])
        .drop_duplicates("slide_id")
        .rename(columns={"sex": "sex_slide"})
    )
    def _majority(values: pd.Series) -> str:
        known = values.dropna().astype(str).str.strip()
        known = known[known.ne("") & known.ne("nan")]
        if known.empty:
            return ""
        return str(known.mode().iloc[0])

    by_mouse = (
        master[["mouse_id", "sex"]]
        .dropna(subset=["mouse_id"])
        .rename(columns={"sex": "sex_mouse"})
        .groupby("mouse_id")["sex_mouse"]
        .agg(_majority)
        .reset_index()
    )
    frame = frame.merge(by_slide, on="slide_id", how="left")
    frame = frame.merge(by_mouse, on="mouse_id", how="left")
    sex = frame["sex_slide"].where(frame["sex_slide"].notna() & frame["sex_slide"].ne(""), frame["sex_mouse"])
    frame["sex"] = sex.map(_normalize_sex)
    return frame.drop(columns=["sex_slide", "sex_mouse"])


def mice_by_sex_table(examples: pd.DataFrame) -> pd.DataFrame:
    """One row per organ and sex: unique mice contributing that organ."""
    frame = examples if "sex" in examples.columns else attach_sex(examples)
    rows = []
    for organ in _present(sorted(frame["organ"].unique())):
        sub = frame.loc[frame["organ"] == organ]
        for sex in SEX_ORDER:
            n = int(sub.loc[sub["sex"] == sex, "mouse_id"].nunique())
            rows.append({"organ": organ, "sex": sex, "n_mice": n})
    return pd.DataFrame(rows)


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


def plot_mice_by_sex(examples: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Horizontal stacked bars: unique mice per organ, split by sex."""
    table = mice_by_sex_table(examples)
    names = _present(sorted(table["organ"].unique()))
    matrix = {
        sex: [int(table.loc[(table.organ == organ) & (table.sex == sex), "n_mice"].sum()) for organ in names]
        for sex in SEX_ORDER
    }
    totals = [sum(matrix[sex][i] for sex in SEX_ORDER) for i in range(len(names))]
    fig_h = max(3.2, 0.42 * len(names) + 1.6)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    ypos = np.arange(len(names))
    left = np.zeros(len(names), dtype=float)
    for sex in SEX_ORDER:
        values = np.asarray(matrix[sex], dtype=float)
        if not values.any():
            continue
        ax.barh(
            ypos, values, left=left, color=SEX_COLORS[sex], height=0.72,
            label=sex, zorder=2, edgecolor="none",
        )
        left = left + values
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    span = max(totals) if totals else 1
    ax.set_xlim(0, span * 1.18)
    for y, total in zip(ypos, totals):
        ax.text(total + span * 0.02, y, str(total), va="center", ha="left", fontsize=9, color="#333333")
    ax.set_xlabel("Unique mice")
    ax.set_title("Mice contributing annotated tissues, by sex")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        title="Sex",
        borderaxespad=0.0,
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    fig.savefig(path, format="svg")
    plt.close(fig)
    return table


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
