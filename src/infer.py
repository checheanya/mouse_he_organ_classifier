#!/usr/bin/env python3
"""Score tissue pieces with the saved organ classifier.

Give one of:

- ``--features`` a UNI H5, or a directory of H5s that are already one tissue each
- ``--sheet`` a CSV with a column of H5 paths
- ``--trident-dir`` a Trident job. Each contour is scored on its own.

The H5s must be UNI v1 (1024-d). Other encoders of the same width will run,
but this model was not trained on them. Overlays are written only when a
thumbnail, a contour file, and a WSI path are all available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import piece_vectors  # noqa: E402
from organs import ORGAN_ORDER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _class_names(bundle: dict) -> list[str]:
    present = list(bundle["organs"])
    ordered = [name for name in ORGAN_ORDER if name in present]
    return ordered + [name for name in present if name not in ordered]


def _predict_rows(bundle: dict, records: list[dict]) -> pd.DataFrame:
    names = _class_names(bundle)
    index = {name: i for i, name in enumerate(bundle["organs"])}
    rows = []
    vectors = []
    for record in records:
        geo = record.get("contour_geojson") or ""
        piece = record.get("piece_id") or ""
        found = piece_vectors(
            Path(record["feature_h5"]),
            Path(geo) if geo else None,
            piece or None,
        )
        for pid, n_patches, vector, how in found:
            vectors.append(vector)
            rows.append({
                "slide_id": record.get("slide_id") or Path(record["feature_h5"]).stem,
                "piece_id": "" if pid is None else str(pid),
                "n_patches": n_patches,
                "pooling": how,
                "feature_h5": record["feature_h5"],
                "contour_geojson": geo,
                "thumbnail": record.get("thumbnail", ""),
                "wsi_path": record.get("wsi_path", ""),
            })
    if not rows:
        raise RuntimeError("no tissue piece had enough patches to score")
    x = bundle["scaler"].transform(np.vstack(vectors))
    proba = bundle["clf"].predict_proba(x)
    # clf.classes_ are label-encoder integers. bundle["organs"] is the class list
    # in encoder.classes_ order, which is also predict_proba column order.
    order = [index[name] for name in names]
    chosen = proba.argmax(axis=1)
    frame = pd.DataFrame(rows)
    frame["predicted_organ"] = [bundle["organs"][int(i)] for i in chosen]
    frame["confidence"] = proba.max(axis=1).round(4)
    for column, src in zip(names, order):
        frame[f"proba_{column}"] = proba[:, src].round(4)
    return frame


def _from_features(path: Path) -> list[dict]:
    files = [path] if path.is_file() else sorted(path.glob("*.h5"))
    if not files:
        raise SystemExit(f"no H5 files in {path}")
    return [{"feature_h5": str(item), "slide_id": item.stem} for item in files]


def _from_sheet(path: Path, column: str) -> list[dict]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if column not in frame.columns:
        raise SystemExit(f"{path} has no column {column}")
    rows = []
    for record in frame.to_dict(orient="records"):
        rows.append({
            "feature_h5": record[column],
            "slide_id": record.get("slide_id") or record.get("slide_name") or Path(record[column]).stem,
            "piece_id": record.get("piece_id", ""),
            "contour_geojson": record.get("contour_geojson", ""),
            "thumbnail": record.get("thumbnail", ""),
            "wsi_path": record.get("wsi_path", ""),
        })
    return rows


def _from_trident(root: Path, wsi_csv: Path | None) -> list[dict]:
    feature_dir = None
    for candidate in (
        root / "20x_256px_0px_overlap" / "features_uni_v1",
        root / "features_uni_official",
        root,
    ):
        if candidate.is_dir() and any(candidate.glob("*.h5")):
            feature_dir = candidate
            break
    if feature_dir is None:
        raise SystemExit(f"no UNI H5s under {root}")
    geo_dir = root / "contours_geojson"
    thumb_dir = root / "thumbnails"
    wsi = {}
    if wsi_csv is not None:
        sheet = pd.read_csv(wsi_csv, dtype=str).fillna("")
        key = "slide_id" if "slide_id" in sheet.columns else "slide_name"
        wsi = dict(zip(sheet[key], sheet["wsi_path"]))
    rows = []
    for feature in sorted(feature_dir.glob("*.h5")):
        stem = feature.stem
        geo = geo_dir / f"{stem}.geojson"
        thumb = next((thumb_dir / f"{stem}{ext}" for ext in (".jpg", ".png", ".tif") if (thumb_dir / f"{stem}{ext}").is_file()), None)
        rows.append({
            "feature_h5": str(feature),
            "slide_id": stem,
            "contour_geojson": str(geo) if geo.is_file() else "",
            "thumbnail": str(thumb) if thumb else "",
            "wsi_path": wsi.get(stem, ""),
        })
    return rows


def _overlays(frame: pd.DataFrame, out_dir: Path) -> None:
    from overlay import draw_slide

    wrote = 0
    for slide_id, group in frame.groupby("slide_id", sort=True):
        thumbnail = group["thumbnail"].iloc[0]
        geojson = group["contour_geojson"].iloc[0]
        wsi = group["wsi_path"].iloc[0]
        if not thumbnail or not geojson or not wsi:
            continue
        calls = []
        for row in group.itertuples(index=False):
            if row.piece_id == "":
                continue
            calls.append((int(row.piece_id), row.predicted_organ, float(row.confidence)))
        if not calls:
            continue
        draw_slide(
            Path(thumbnail),
            Path(geojson),
            Path(wsi),
            calls,
            out_dir / f"{slide_id}.svg",
            str(slide_id),
        )
        wrote += 1
    print(f"wrote {wrote} overlays to {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "results" / "model.joblib")
    parser.add_argument("--features", type=Path, default=None, help="One H5, or a directory of one-tissue H5s.")
    parser.add_argument("--sheet", type=Path, default=None, help="CSV of H5 paths.")
    parser.add_argument("--feature-column", default="feature_h5")
    parser.add_argument("--trident-dir", type=Path, default=None, help="Trident job with features and contours.")
    parser.add_argument("--wsi-csv", type=Path, default=None, help="slide_id,wsi_path. Required for overlays.")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "inference" / "predictions.csv")
    parser.add_argument("--overlay-dir", type=Path, default=None)
    args = parser.parse_args()
    chosen = [args.features, args.sheet, args.trident_dir]
    if sum(item is not None for item in chosen) != 1:
        raise SystemExit("pass exactly one of --features, --sheet, or --trident-dir")

    if args.features is not None:
        records = _from_features(args.features)
    elif args.sheet is not None:
        records = _from_sheet(args.sheet, args.feature_column)
    else:
        records = _from_trident(args.trident_dir, args.wsi_csv)

    bundle = joblib.load(args.model)
    frame = _predict_rows(bundle, records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.drop(columns=["thumbnail", "wsi_path", "contour_geojson"]).to_csv(args.out, index=False)
    print(f"wrote {len(frame)} rows to {args.out}", flush=True)
    if args.overlay_dir is not None:
        _overlays(frame, args.overlay_dir)


if __name__ == "__main__":
    main()
