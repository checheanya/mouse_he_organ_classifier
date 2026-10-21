#!/usr/bin/env python3
"""Name each tissue on the JAX ITP canagliflozin slides.

The saved organ classifier scores every contour. Three low-quality slides are
left out. On UT12343_3, piece 5 is left out: that contour is the upper heart
together with the kidney a dark bubble crosses. CPU only. A GPU does not help.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import _from_trident, _overlays, _predict_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRIDENT = Path("/data/labs/sinha_lab/achechenina/embeddings_uni/jax_itp_canagliflozin")
WSI_CSV = Path(
    "/data/labs/sinha_lab/achechenina/projects/local_aging/slurm_runs/prep_data/"
    "jax_itp_canagliflozin_prep_data/slides_jax_itp_canagliflozin_usable.csv"
)
OUT = TRIDENT / "organ_calls"

# Whole slides. The tissue is too torn or too full of holes to call.
DROP_SLIDES = {"JL17926_3", "JL18136_1", "UM85925_1"}
# slide_id, piece_id. Piece 5 is the upper heart fused to the bubbled kidney.
DROP_PIECES = {("UT12343_3", "5")}


def main() -> None:
    usable = pd.read_csv(WSI_CSV, dtype=str).fillna("")
    wsi_csv = OUT / "wsi_paths.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    usable.rename(columns={"wsi": "wsi_path", "slide_name": "slide_id"})[
        ["slide_id", "wsi_path"]
    ].to_csv(wsi_csv, index=False)

    records = [
        row for row in _from_trident(TRIDENT, wsi_csv)
        if row["slide_id"] not in DROP_SLIDES
    ]
    bundle = joblib.load(ROOT / "results" / "model.joblib")
    frame = _predict_rows(bundle, records)
    keep = [
        (row.slide_id, str(row.piece_id)) not in DROP_PIECES
        for row in frame.itertuples(index=False)
    ]
    frame = frame.loc[keep].reset_index(drop=True)
    out_csv = OUT / "predictions.csv"
    frame.drop(columns=["thumbnail", "wsi_path", "contour_geojson"]).to_csv(out_csv, index=False)
    excluded = pd.DataFrame(
        [{"slide_id": slide, "piece_id": "", "reason": "low quality slide"} for slide in sorted(DROP_SLIDES)]
        + [{"slide_id": slide, "piece_id": piece, "reason": "bubble overlaps upper heart and kidney"}
           for slide, piece in sorted(DROP_PIECES)]
    )
    excluded.to_csv(OUT / "excluded.csv", index=False)
    print(f"wrote {len(frame)} rows to {out_csv}", flush=True)
    _overlays(frame, OUT / "overlays")


if __name__ == "__main__":
    main()
