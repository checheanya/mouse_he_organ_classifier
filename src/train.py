#!/usr/bin/env python3
"""Train the organ classifier on unrefined annotated tissue pieces.

Each example is the mean of the UNI v1 patch embeddings inside one tissue
contour, with no tissue-color refinement. Evaluation is stratified by organ
and grouped by mouse, so pieces from the same mouse stay in one fold.
A GPU does not help.
Classifier is a logistic regression model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import piece_vectors  # noqa: E402
from organs import FEATURE_DIM, MIN_PATCHES, ORGAN_ORDER  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _expand(record: dict) -> list[dict]:
    geo = record.get("contour_geojson") or ""
    piece = record.get("piece_id") or ""
    try:
        # Hand-labeled pieces are kept even when they are smaller than the
        # fragment cut used for single-organ slides.
        # Hand-labeled pieces are kept at any size. v3 contours use 150 so the
        # smallest curated pancreas (190 patches) stays, and drawing crumbs do not.
        if record.get("source") in {"manual_piece", "extra"}:
            minimum = 1
        elif record.get("source") == "v3_contour":
            minimum = 150
        else:
            minimum = MIN_PATCHES
        pieces = piece_vectors(
            Path(record["feature_h5"]),
            Path(geo) if geo else None,
            piece or None,
            min_patches=minimum,
        )
    except Exception as exc:  # noqa: BLE001 — one bad slide should not stop the roster
        return [{"error": f"{record.get('slide_id')}: {exc}"}]
    rows = []
    for pid, n_patches, vector, how in pieces:
        rows.append({
            "slide_id": record["slide_id"],
            "mouse_id": record.get("mouse_id") or record["slide_id"],
            "cohort": record.get("cohort", ""),
            "organ": record["organ"],
            "feature_h5": record["feature_h5"],
            "piece_id": "" if pid is None else str(pid),
            "n_patches": n_patches,
            "pooling": how,
            "source": record.get("source", ""),
            "vector": vector,
        })
    return rows


def _collect(annotations: pd.DataFrame, workers: int) -> tuple[pd.DataFrame, np.ndarray]:
    records = annotations.fillna("").to_dict(orient="records")
    rows: list[dict] = []
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, found in enumerate(pool.map(_expand, records, chunksize=4), start=1):
            for item in found:
                if "error" in item:
                    errors.append(item["error"])
                else:
                    rows.append(item)
            if i % 100 == 0 or i == len(records):
                print(f"pooled {i}/{len(records)} slides, {len(rows)} pieces", flush=True)
    if errors:
        print(f"{len(errors)} slides failed to open; first: {errors[0]}", flush=True)
    if not rows:
        raise RuntimeError("no tissue pieces passed the patch minimum")
    vectors = np.vstack([item.pop("vector") for item in rows]).astype(np.float32)
    if vectors.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"expected {FEATURE_DIM}-d UNI vectors, got {vectors.shape[1]}")
    return pd.DataFrame(rows), vectors


def _n_splits(y: np.ndarray) -> int:
    splits = 5
    counts = pd.Series(y).value_counts()
    while splits > 2 and int(counts.min()) < splits:
        splits -= 1
    return splits


def _cv(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, int]:
    splits = _n_splits(y)
    folder = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=0)
    pred = np.empty(len(y), dtype=object)
    for train_idx, test_idx in folder.split(x, y, groups):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        clf = _classifier()
        clf.fit(x_train, y[train_idx])
        pred[test_idx] = clf.predict(x_test)
    return pred, splits


def _classifier() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
    )


def _ordered(names: list[str]) -> list[str]:
    return [name for name in ORGAN_ORDER if name in names] + [name for name in names if name not in ORGAN_ORDER]


def _report(examples: pd.DataFrame, y: np.ndarray, pred: np.ndarray, splits: int, out: Path) -> None:
    labels = _ordered(sorted(set(y)))
    matrix = confusion_matrix(y, pred, labels=labels)
    per_organ = []
    for name in labels:
        mask = y == name
        hit = pred[mask] == name
        support = int(mask.sum())
        recall = float(hit.mean()) if support else 0.0
        predicted = pred == name
        precision = float((y[predicted] == name).mean()) if predicted.any() else 0.0
        per_organ.append({
            "organ": name,
            "n_pieces": support,
            "n_mice": int(examples.loc[mask, "mouse_id"].nunique()),
            "n_slides": int(examples.loc[mask, "slide_id"].nunique()),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
        })
    metrics = {
        "n_pieces": int(len(examples)),
        "n_slides": int(examples["slide_id"].nunique()),
        "n_mice": int(examples["mouse_id"].nunique()),
        "n_organs": len(labels),
        "cv_splits": splits,
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 3),
        "macro_f1": round(float(f1_score(y, pred, average="macro", labels=labels)), 3),
        "pooling": examples["pooling"].value_counts().to_dict(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame(per_organ).to_csv(out / "per_organ_cv.csv", index=False)
    lines = [
        "# Organ classifier training",
        "",
        "Unrefined UNI v1 segmentations. Each row is one tissue piece: the mean of",
        "its patch embeddings, with no tissue-color cut. eGPS spleen and pancreas",
        "use the v3 contours. The other organs use the original contour, because",
        "those slides have no v2 or v3 segmentation.",
        "",
        f"- Pieces: {metrics['n_pieces']}",
        f"- Slides: {metrics['n_slides']}",
        f"- Mice: {metrics['n_mice']}",
        f"- Organs: {metrics['n_organs']}",
        f"- Grouped stratified CV: {splits} folds, grouped by mouse",
        f"- Balanced accuracy: {metrics['balanced_accuracy']}",
        f"- Macro F1: {metrics['macro_f1']}",
        "",
        "Pooling: " + ", ".join(f"{key} {value}" for key, value in metrics["pooling"].items()),
        "",
        "| Organ | Pieces | Mice | Slides | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in per_organ:
        lines.append(
            f"| {row['organ']} | {row['n_pieces']} | {row['n_mice']} | {row['n_slides']} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} |"
        )
    lines += [
        "",
        "The saved model is `model.joblib`. It was fit on every piece above, after",
        "the cross-validation. Figures: `counts.svg`, `mice_by_sex.svg`, `umap_organ.svg`,",
        "`confusion.svg`.",
        "",
    ]
    (out / "TRAINING_REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ROOT / "results" / "annotations.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    annotations = pd.read_csv(args.annotations, dtype=str).fillna("")
    examples, vectors = _collect(annotations, args.workers)
    examples.to_csv(args.out / "training_pieces.csv", index=False)
    np.save(args.out / "training_features.npy", vectors)

    y = examples["organ"].to_numpy()
    groups = examples["mouse_id"].to_numpy()
    pred, splits = _cv(vectors, y, groups)
    examples = examples.copy()
    examples["cv_predicted"] = pred
    examples.to_csv(args.out / "training_pieces.csv", index=False)

    from plotting import attach_sex, plot_confusion, plot_counts, plot_mice_by_sex, plot_umap

    examples = attach_sex(examples)
    examples.to_csv(args.out / "training_pieces.csv", index=False)
    labels = _ordered(sorted(set(y)))
    plot_counts(examples["organ"].value_counts(), args.out / "counts.svg")
    mice_table = plot_mice_by_sex(examples, args.out / "mice_by_sex.svg")
    mice_table.to_csv(args.out / "mice_by_sex.csv", index=False)
    plot_confusion(confusion_matrix(y, pred, labels=labels), labels, args.out / "confusion.svg")

    import umap

    embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(vectors)
    pd.DataFrame({"umap_1": embedding[:, 0], "umap_2": embedding[:, 1], "organ": y}).to_csv(
        args.out / "umap_coords.csv", index=False
    )
    plot_umap(embedding, list(y), args.out / "umap_organ.svg")

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    scaler = StandardScaler()
    clf = _classifier()
    clf.fit(scaler.fit_transform(vectors), y_enc)
    joblib.dump(
        {"scaler": scaler, "clf": clf, "organs": list(encoder.classes_), "feature_dim": int(vectors.shape[1])},
        args.out / "model.joblib",
    )
    _report(examples, y, pred, splits, args.out)


if __name__ == "__main__":
    main()
