#!/usr/bin/env python3
"""Write the training manifest of annotated, unrefined tissue segmentations.

Single-organ slides contribute one row per slide. The trainer then mean-pools
each contour on that slide and labels every piece with the slide organ.
Those slides have one segmentation; none of them has a v2 or v3 contour.

The ten manually labeled Warren slides contribute one row per labeled piece.
Those ten slides were not redrawn in v2 or v3.

eGPS survey spleen and pancreas contribute the v3 contours. The H5 is the
patches inside that contour, before the tissue-color cut, and the geojson is
the opened v3 outline. Extra rows in ``data/extra_annotations.csv`` are appended
as they are.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from organs import PENDING_ORGANS, SINGLE_ORGAN_COHORTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MASTER = Path("/data/labs/sinha_lab/achechenina/data/261015_master_organ_metadata.csv")
MANUAL = Path("/data/labs/sinha_lab/anamikay/uw_slide_tissue_numbers.csv")
PIECE_DIR = Path(
    "/data/labs/sinha_lab/achechenina/embeddings_uni/warren_slam/features_uni_official"
)
JAX_ROOT = Path("/data/labs/sinha_lab/achechenina/embeddings_uni/jax_kidneys")
JAX_META = Path("/data/labs/sinha_lab/achechenina/data/261013_jax_kidney_known_age_meta.csv")
EXTRA = ROOT / "data" / "extra_annotations.csv"

COLUMNS = [
    "slide_id",
    "mouse_id",
    "cohort",
    "organ",
    "feature_h5",
    "contour_geojson",
    "piece_id",
    "source",
]


def _piece_numbers(value) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def single_organ_rows(master: pd.DataFrame) -> pd.DataFrame:
    keep = master["cohort"].isin(SINGLE_ORGAN_COHORTS)
    keep &= ~master["organ"].isin(PENDING_ORGANS)
    keep &= master["analysis_primary"].fillna(False).astype(bool)
    frame = master.loc[keep].copy()
    frame = frame[frame["embedding_path"].fillna("").map(lambda path: Path(path).is_file())]
    rows = pd.DataFrame({
        "slide_id": frame["slide_id"].astype(str),
        "mouse_id": frame["mouse_id"].astype(str),
        "cohort": frame["cohort"].astype(str),
        "organ": frame["organ"].astype(str),
        "feature_h5": frame["embedding_path"].astype(str),
        "contour_geojson": frame["contour_geojson"].fillna("").astype(str),
        "piece_id": "",
        "source": "single_organ_slide",
    })
    return rows


def manual_rows() -> pd.DataFrame:
    labels = pd.read_csv(MANUAL)
    records = []
    organs = [name for name in ("Lung", "Liver", "Heart", "Kidney") if name in labels.columns]
    for row in labels.itertuples(index=False):
        slide = str(row.slide_name).strip()
        for organ in organs:
            for number in _piece_numbers(getattr(row, organ)):
                path = PIECE_DIR / f"{slide}_{number}.h5"
                if not path.is_file():
                    print(f"missing manual piece {path.name}")
                    continue
                records.append({
                    "slide_id": slide,
                    "mouse_id": slide,
                    "cohort": "Warren_UW",
                    "organ": organ,
                    "feature_h5": str(path),
                    "contour_geojson": "",
                    "piece_id": number,
                    "source": "manual_piece",
                })
    return pd.DataFrame(records, columns=COLUMNS)


def v3_spleen_pancreas_rows() -> pd.DataFrame:
    """eGPS survey pieces whose outline is the v3 call, including opened fusions."""
    meta = pd.read_csv(JAX_META)
    mouse = dict(zip(meta["slide_name"].astype(str), meta["sample_id"].astype(str)))
    slide_ids = dict(zip(meta["slide_name"].astype(str), meta["slide_id"].astype(str)))
    records = []
    for organ, folder in (("Spleen", "spleen"), ("Pancreas", "pancreas")):
        piece_dir = JAX_ROOT / f"{folder}_pieces_uni_v1"
        geo_dir = JAX_ROOT / f"{folder}_refined" / "contours"
        for feature in sorted(piece_dir.glob("*.h5")):
            geo = geo_dir / f"{feature.stem}.geojson"
            records.append({
                "slide_id": slide_ids.get(feature.stem, feature.stem),
                "mouse_id": mouse.get(feature.stem, feature.stem),
                "cohort": "eGPS",
                "organ": organ,
                "feature_h5": str(feature),
                "contour_geojson": str(geo) if geo.is_file() else "",
                "piece_id": "",
                "source": "v3_contour",
            })
    return pd.DataFrame(records, columns=COLUMNS)


def extra_rows(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(path, dtype=str).fillna("")
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[frame["organ"].ne("") & frame["feature_h5"].ne("")]
    frame.loc[frame["source"].eq(""), "source"] = "extra"
    return frame[COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=MASTER)
    parser.add_argument("--extra", type=Path, default=EXTRA)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "annotations.csv")
    args = parser.parse_args()

    master = pd.read_csv(args.master)
    parts = [single_organ_rows(master), manual_rows(), v3_spleen_pancreas_rows(), extra_rows(args.extra)]
    table = pd.concat(parts, ignore_index=True)
    table = table.drop_duplicates(["feature_h5", "piece_id", "organ"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"wrote {len(table)} rows to {args.out}")
    print(table.groupby(["source", "organ"]).size().to_string())


if __name__ == "__main__":
    main()
