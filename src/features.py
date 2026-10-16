"""Mean-pool UNI patch embeddings inside an unrefined tissue contour."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.path as mpath
import numpy as np

from organs import MIN_PATCHES, PATCH


def mean_pool(feats: np.ndarray) -> np.ndarray | None:
    """Mean of L2-normalized patch vectors, then L2-normalized.

    This is the same pooling used for the Warren piece prototypes. It does
    not apply the later tissue-color refinement.
    """
    if feats.size == 0:
        return None
    feats = np.asarray(feats, dtype=np.float32)
    ok = ~np.isnan(feats).any(axis=1)
    feats = feats[ok]
    if len(feats) == 0:
        return None
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.clip(norms, 1e-8, None)
    mean = feats.mean(axis=0)
    mean = mean / max(float(np.linalg.norm(mean)), 1e-8)
    return mean.astype(np.float32)


def read_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        feats = np.asarray(handle["features"][:], dtype=np.float32)
        coords = np.asarray(handle["coords"][:]) if "coords" in handle else np.zeros((len(feats), 2))
    return feats, coords


def polygons(geo_path: Path) -> list[tuple[int, mpath.Path, float]]:
    data = json.loads(Path(geo_path).read_text())
    out = []
    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        tid = int(feat.get("properties", {}).get("tissue_id", len(out)))
        rings = []
        if geom.get("type") == "Polygon":
            rings = [geom["coordinates"][0]]
        elif geom.get("type") == "MultiPolygon":
            rings = [part[0] for part in geom["coordinates"]]
        if not rings:
            continue
        ring = max(rings, key=len)
        xy = np.asarray(ring, dtype=np.float64)
        if len(xy) < 3:
            continue
        area = float(abs(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1))) / 2)
        out.append((tid, mpath.Path(xy), area))
    return out


def assign_patches(coords: np.ndarray, pieces: list[tuple[int, mpath.Path, float]]) -> np.ndarray:
    """Patch index -> tissue id. A patch in several contours keeps the smaller one."""
    centers = coords.astype(np.float64) + PATCH / 2
    assigned = np.full(len(centers), -1, dtype=np.int32)
    best_area = np.full(len(centers), np.inf)
    for tid, path, area in pieces:
        inside = path.contains_points(centers)
        take = inside & (area < best_area)
        assigned[take] = tid
        best_area[take] = area
    return assigned


def piece_vectors(
    feature_h5: Path,
    geojson: Path | None,
    piece_id: str | None,
    min_patches: int = MIN_PATCHES,
) -> list[tuple[int | None, int, np.ndarray, str]]:
    """Return (piece_id, n_patches, vector, how) for one annotation row.

    ``how`` is ``piece`` when the contour split worked, ``piece_h5`` when the
    file is already one tissue, and ``slide_mean`` when the contour split
    assigned nothing and the whole unrefined H5 was used instead.
    """
    feats, coords = read_h5(feature_h5)
    if geojson is None or not Path(geojson).is_file():
        vec = mean_pool(feats)
        if vec is None or len(feats) < min_patches:
            return []
        pid = int(piece_id) if piece_id not in (None, "", "nan") else None
        return [(pid, int(len(feats)), vec, "piece_h5")]

    pieces = polygons(Path(geojson))
    wanted = None if piece_id in (None, "", "nan") else int(float(piece_id))
    if wanted is not None:
        pieces = [item for item in pieces if item[0] == wanted]
    if not pieces or len(coords) != len(feats):
        vec = mean_pool(feats)
        if vec is None or len(feats) < min_patches:
            return []
        return [(wanted, int(len(feats)), vec, "slide_mean")]

    owned = assign_patches(coords, pieces)
    rows = []
    for tid, _path, _area in pieces:
        idx = np.flatnonzero(owned == tid)
        if len(idx) < min_patches:
            continue
        vec = mean_pool(feats[idx])
        if vec is not None:
            rows.append((tid, int(len(idx)), vec, "piece"))
    if rows:
        return rows
    vec = mean_pool(feats)
    if vec is None or len(feats) < min_patches:
        return []
    return [(wanted, int(len(feats)), vec, "slide_mean")]
