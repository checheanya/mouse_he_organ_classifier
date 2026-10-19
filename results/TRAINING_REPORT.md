# Organ classifier training

Unrefined UNI v1 segmentations. Each row is one tissue piece: the mean of
its patch embeddings, with no tissue-color cut. eGPS spleen and pancreas
use the v3 contours. The other organs use the original contour, because
those slides have no v2 or v3 segmentation.

- Pieces: 3618
- Slides: 1669
- Mice: 1296
- Organs: 13
- Grouped stratified CV: 5 folds, grouped by mouse
- Balanced accuracy: 0.986
- Macro F1: 0.983

Pooling: piece 3568, piece_h5 49, slide_mean 1

| Organ | Pieces | Mice | Slides | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Brain | 74 | 47 | 47 | 1.000 | 1.000 |
| Heart | 55 | 28 | 28 | 1.000 | 0.964 |
| Lung | 146 | 60 | 88 | 0.973 | 0.986 |
| Liver | 2451 | 1190 | 1191 | 0.999 | 0.997 |
| Kidney | 169 | 89 | 89 | 0.976 | 0.982 |
| Gut | 82 | 30 | 56 | 0.976 | 0.988 |
| Adrenal | 44 | 26 | 26 | 0.933 | 0.955 |
| Pancreas | 180 | 73 | 73 | 1.000 | 0.978 |
| Spleen | 112 | 69 | 69 | 0.957 | 1.000 |
| Muscle | 78 | 18 | 18 | 0.975 | 1.000 |
| Skin | 49 | 21 | 21 | 0.980 | 1.000 |
| Prostate | 107 | 21 | 21 | 0.972 | 0.963 |
| Testis | 71 | 21 | 21 | 1.000 | 1.000 |

The saved model is `model.joblib`. It was fit on every piece above, after
the cross-validation. Figures: `counts.svg`, `umap_organ.svg`, `confusion.svg`.
