"""Organ names, colors, and the cohorts used to build the training set.

Colors match the organ overlays already used for these slides. Spleen and
pancreas stay in the palette so a later run uses the same colors, and they
are left out of training until that curation is finished.
"""

from __future__ import annotations

PENDING_ORGANS = ("Spleen", "Pancreas")

# Single-organ slides. The organ name is the cassette or the sample sheet,
# and the Trident H5 is the unrefined tissue segmentation. Multi-organ slides
# (Warren UW, eGPS survey, eGPS SLAM) are not in this list: a slide-level
# mean would mix organs, and the piece names there are still being reviewed.
SINGLE_ORGAN_COHORTS = (
    "Adams",
    "Shabnam",
    "Kirkland",
    "Warren_October",
    "JAX_NSC",
)

ORGAN_ORDER = (
    "Brain",
    "Heart",
    "Lung",
    "Liver",
    "Kidney",
    "Gut",
    "Adrenal",
    "Pancreas",
    "Spleen",
    "Muscle",
    "Skin",
    "Prostate",
    "Testis",
)

COLORS = {
    "Kidney": "#6b4c9a",
    "Liver": "#0f85a0",
    "Heart": "#ed8b00",
    "Lung": "#dd4124",
    "Gut": "#ba7999",
    "Brain": "#5b8fa8",
    "Adrenal": "#c4a35a",
    "Skin": "#d4a574",
    "Prostate": "#7a9e7e",
    "Testis": "#8f6b8a",
    "Muscle": "#4c6b7a",
    "Spleen": "#a84b6a",
    "Pancreas": "#6e8b3d",
    "unknown": "#8d8d8d",
}

PATCH = 256
MIN_PATCHES = 200
FEATURE_DIM = 1024
