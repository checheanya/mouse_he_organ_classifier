# Organ classifier training

Unrefined UNI v1 segmentations. Each row is one tissue piece: the mean of
its patch embeddings, with no tissue-color cut. Spleen and pancreas are
not in this fit.

- Pieces: 3326
- Slides: 1576
- Mice: 1239
- Organs: 11
- Grouped stratified CV: 5 folds, grouped by mouse
- Balanced accuracy: 0.989
- Macro F1: 0.983

Pooling: piece 3276, piece_h5 49, slide_mean 1

| Organ | Pieces | Mice | Slides | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Brain | 74 | 47 | 47 | 1.000 | 1.000 |
| Heart | 55 | 28 | 28 | 1.000 | 0.964 |
| Lung | 146 | 60 | 88 | 0.973 | 0.993 |
| Liver | 2451 | 1190 | 1191 | 1.000 | 0.996 |
| Kidney | 169 | 89 | 89 | 0.982 | 0.988 |
| Gut | 82 | 30 | 56 | 0.976 | 1.000 |
| Adrenal | 44 | 26 | 26 | 0.898 | 1.000 |
| Muscle | 78 | 18 | 18 | 0.975 | 0.987 |
| Skin | 49 | 21 | 21 | 0.980 | 1.000 |
| Prostate | 107 | 21 | 21 | 0.971 | 0.953 |
| Testis | 71 | 21 | 21 | 1.000 | 1.000 |

The saved model is `model.joblib`. It was fit on every piece above, after
the cross-validation. Figures: `counts.svg`, `umap_organ.svg`, `confusion.svg`.
