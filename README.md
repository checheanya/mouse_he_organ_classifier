## Mouse organ classification from H&E slides

A logistic regression on the mean UNI v1 embedding of one tissue piece. It is the same idea as the Warren piece prototypes in `data/Warrens_SLAM/UW_slides_v1`: each segmented tissue is one vector, and the model names that tissue. The difference is a trained classifier with a probability for every organ in the training set, instead of a nearest prototype.

The vector is the mean of the L2-normalized UNI v1 patch embeddings inside the tissue contour. This is the unrefined segmentation. The later tissue-color cut used by the slide clocks is not applied. If a slide has several tissues, each contour is classified on its own.

Spleen and pancreas are in the model. On the eGPS survey they use the v3 contours, including the pieces that were opened where two organs had been fused. Warren October spleen and pancreas slides use the original contour, because those slides were not redrawn.

Tissues with large artefacts or lesions, including a big tumor region, can be assigned to the wrong organ.

Training and inference live in `src/`. The fit, the cross-validation table, and the figures live in `results/`.

### What is in the training set

Single-organ slides, where the cassette or the sample sheet names the organ, plus the manually labeled pieces from the ten Warren slides in `uw_slide_tissue_numbers.csv`.

| Source | Organs |
| --- | --- |
| Adams, Shabnam | Liver |
| Kirkland | Brain, Gut, Kidney, Liver, Lung |
| Warren October | Adrenal, Brain, Heart, Kidney, Liver, Lung, Muscle, Pancreas, Prostate, Skin, Spleen, Testis |
| JAX NSC | Kidney |
| Warren manual pieces | Heart, Kidney, Liver, Lung |
| eGPS survey v3 contours | Spleen, Pancreas |

The single-organ slides and the ten Warren slides have no v2 or v3 contour, so those rows use the original segmentation. Where a later contour exists, the manifest points at it: eGPS spleen and pancreas use `spleen_pieces_uni_v1` and `pancreas_pieces_uni_v1` with the v3 geojson under `spleen_refined/contours` and `pancreas_refined/contours`. Warren kidney v2 and v3 files belong to other slides and are not labels in this model.

The rest of the eGPS survey, Warren UW, and eGPS SLAM are multi-organ slides. A slide-level mean would mix organs. A reviewed piece can be added as a row in `data/extra_annotations.csv`.

Pieces smaller than 200 patches are dropped on single-organ slides, so fragments are not used as that organ. Hand-labeled pieces are kept at any size.

Liver is the largest class. The logistic regression uses balanced class weights so a class is not ignored because it has fewer slides.

### 0. Environment

On this machine the packages are already in conda env `ds_python310_env`:

```bash
source /net/fs001/pool0/home/achechenina/miniforge3/etc/profile.d/conda.sh
conda activate ds_python310_env
```

On another machine, create that env from `environment.yml`:

```bash
mamba env create -f environment.yml
conda activate ds_python310_env
```

A GPU does not help training or inference. Both are CPU mean-pooling and a linear model.

### 1. Training

From this directory:

```bash
sbatch slurm/train_organ_classifier.slurm
```

The job builds `results/annotations.csv`, mean-pools every piece, and writes:

- `results/model.joblib` — scaler, logistic regression, and the organ list
- `results/training_pieces.csv` — one row per piece, with the cross-validated call
- `results/per_organ_cv.csv` and `results/TRAINING_REPORT.md` — precision and recall by organ
- `results/counts.svg` — number of pieces per organ
- `results/umap_organ.svg` — UMAP of the mean-pooled vectors, colored by organ
- `results/confusion.svg` — mouse-grouped cross-validation

The cross-validation is stratified by organ and grouped by mouse, so every piece from one mouse stays in the same fold. The saved model is then fit on all of the pieces.

### Current fit

Job `1703534` finished on 19 Oct 2026. It used 3,618 pieces from 1,669 slides and 13 organs. Mouse-grouped 5-fold cross-validation: balanced accuracy 0.986, macro F1 0.983. Twenty-six pieces were wrong. Every spleen was correct. Four other pieces were called spleen (two adrenal, two liver). Four pancreas pieces were called prostate, skin, or spleen, and nothing else was called pancreas.

| Organ | Pieces | Mice | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Brain | 74 | 47 | 1.000 | 1.000 |
| Heart | 55 | 28 | 1.000 | 0.964 |
| Lung | 146 | 60 | 0.973 | 0.986 |
| Liver | 2451 | 1190 | 0.999 | 0.997 |
| Kidney | 169 | 89 | 0.976 | 0.982 |
| Gut | 82 | 30 | 0.976 | 0.988 |
| Adrenal | 44 | 26 | 0.933 | 0.955 |
| Pancreas | 180 | 73 | 1.000 | 0.978 |
| Spleen | 112 | 69 | 0.957 | 1.000 |
| Muscle | 78 | 18 | 0.975 | 1.000 |
| Skin | 49 | 21 | 0.980 | 1.000 |
| Prostate | 107 | 21 | 0.972 | 0.963 |
| Testis | 71 | 21 | 1.000 | 1.000 |

Spleen is 52 v3 eGPS pieces plus 60 Warren October pieces. Pancreas is 84 v3 eGPS pieces plus 96 Warren October pieces. The other organs are the same single-organ slides and 49 hand-labeled Warren pieces as before. This is not a test of unlabeled pieces on a mixed slide. A piece with a large lesion can still be wrong. Full table: `results/TRAINING_REPORT.md`.

To add another reviewed piece, append a row to `data/extra_annotations.csv` and resubmit the same job. Columns:

```text
slide_id,mouse_id,cohort,organ,feature_h5,contour_geojson,piece_id,source
```

`feature_h5` is either one tissue's H5, or a slide H5 with `contour_geojson` set. `piece_id` can be left blank to use every contour in that file. `source` should be `extra`. Point `contour_geojson` at the v3 file when that slide has one.

### 2. Inference

The model expects a UNI v1 H5 (`features` and `coords`, 1024 dimensions). It has not been tested on another encoder. A file of the same width will run, and the probabilities should not be treated as organ calls.

Run one of the three inputs below. The output CSV has one row per tissue, the predicted organ, the confidence (the highest probability), and a `proba_<organ>` column for every trained organ.

**A directory of H5s that are already one tissue each** (a piece file, or a single-organ slide):

```bash
python src/infer.py \
  --model results/model.joblib \
  --features /path/to/h5_directory \
  --out results/inference/predictions.csv
```

`--features` also accepts a single H5.

**A sheet of paths.** The sheet needs a `feature_h5` column. Optional columns: `slide_id`, `piece_id`, `contour_geojson`, `thumbnail`, `wsi_path`.

```bash
python src/infer.py \
  --model results/model.joblib \
  --sheet /path/to/sheet.csv \
  --feature-column feature_h5 \
  --out results/inference/predictions.csv
```

**A raw Trident directory.** The script looks for `20x_256px_0px_overlap/features_uni_v1/*.h5` (or `features_uni_official/*.h5`) and `contours_geojson/<stem>.geojson`, and scores each contour separately.

```bash
python src/infer.py \
  --model results/model.joblib \
  --trident-dir /path/to/trident_job \
  --wsi-csv /path/to/slide_id_wsi_path.csv \
  --out results/inference/predictions.csv \
  --overlay-dir results/inference/overlays
```

`--overlay-dir` writes an SVG per slide: the thumbnail, each contour, and the predicted organ with its confidence. The legend is beside the image. Overlays are written only when that slide has a thumbnail under `<trident-dir>/thumbnails/`, a contour file, and a row in `--wsi-csv` (`slide_id`, `wsi_path`) so the contour can be scaled to the thumbnail. Without those, the CSV is still written.
