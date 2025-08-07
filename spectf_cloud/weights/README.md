# Model Weights

This directory contains versions of the SpecTf EMIT Cloud model.

- `v0.0.1`
    - Published model version
- `v0.1.0`
    - Updated model trained on a larger dataset, in preparation for EMIT cloud
    mask product delivery

## Performance Metrics

_F1-Scores on each test set. *Reported in publication._

| Model  | Threshold | Labelbox | MMGIS | MMGIS-2 | L+M    | L+M+M2 |
| ------ | --------- | -------- | ----- | ------- | ------ | ------ |
| v0.0.1 | 0.52      | 0.919    | 0.988 | 0.860   | 0.952* | 0.915  |
| v0.1.0 | 0.51      | 0.912    | 0.996 | 0.943   | 0.952  | 0.948  |