"""Draw predicted organ names on a slide thumbnail. The legend sits beside the image."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image

from features import polygons
from organs import COLORS
from plotting import legend_handles


def _tiff_size(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as handle:
            hdr = handle.read(8)
            if hdr[:2] not in (b"II", b"MM") or int.from_bytes(hdr[2:4], "little" if hdr[:2] == b"II" else "big") != 42:
                return None
            endian = "little" if hdr[:2] == b"II" else "big"
            handle.seek(int.from_bytes(hdr[4:8], endian))
            n = int.from_bytes(handle.read(2), endian)
            width = height = None
            for _ in range(n):
                ent = handle.read(12)
                tag = int.from_bytes(ent[0:2], endian)
                typ = int.from_bytes(ent[2:4], endian)
                val = int.from_bytes(ent[8:12], endian)
                if typ == 3:
                    val &= 0xFFFF
                if tag == 256:
                    width = val
                elif tag == 257:
                    height = val
        if width and height:
            return width, height
    except OSError:
        return None
    return None


def slide_size(path: Path) -> tuple[int, int]:
    found = _tiff_size(path)
    if found:
        return found
    import openslide

    slide = openslide.OpenSlide(str(path))
    try:
        return slide.dimensions
    finally:
        slide.close()


def draw_slide(
    thumbnail: Path,
    geojson: Path,
    wsi: Path,
    calls: list[tuple[int, str, float]],
    out: Path,
    title: str,
) -> None:
    slide_w, slide_h = slide_size(wsi)
    image = np.asarray(Image.open(thumbnail).convert("RGB"))
    thumb_h, thumb_w = image.shape[:2]
    sx = thumb_w / slide_w
    sy = thumb_h / slide_h
    rings = {tid: path.vertices for tid, path, _area in polygons(geojson)}

    fig_h = 8.0
    fig_w = fig_h * (thumb_w / thumb_h) + 2.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(image)
    present = []
    for piece_id, organ, confidence in calls:
        ring = rings.get(piece_id)
        if ring is None:
            continue
        color = COLORS.get(organ, COLORS["unknown"])
        xs = ring[:, 0] * sx
        ys = ring[:, 1] * sy
        ax.add_patch(MplPolygon(
            np.column_stack([xs, ys]), closed=True, fill=True,
            facecolor=color, edgecolor="none", alpha=0.16, lw=0, zorder=2,
        ))
        ax.plot(np.r_[xs, xs[0]], np.r_[ys, ys[0]], color=color, lw=2.0, zorder=3)
        cx, cy = float(xs.mean()), float(ys.mean())
        ax.text(
            cx, cy, f"{piece_id}  {organ}  {confidence:.2f}",
            ha="center", va="center", fontsize=8, color=color, zorder=4,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": color, "linewidth": 0.6, "alpha": 0.92},
        )
        if organ not in present:
            present.append(organ)
    ax.legend(
        handles=legend_handles(present),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        title="Organ",
        borderaxespad=0.0,
    )
    ax.set_xlim(0, thumb_w)
    ax.set_ylim(thumb_h, 0)
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=11, pad=8)
    fig.subplots_adjust(left=0.02, right=0.78, top=0.92, bottom=0.03)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg")
    plt.close(fig)
