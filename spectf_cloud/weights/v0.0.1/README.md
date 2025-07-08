## Files:
- weights.pt
    - md5sum: c54fe5ef078b44182d3cc626008959ab
- model_no_bandcat_bf16.onnx
    - md5sum: b5b9c355e045eeebcdec751bdbcbd327
- model.engine
    - md5sum: 439cf0d5aa33d64b2340c74527f16aef 

## Datasets:
- [labelbox](https://zenodo.org/records/15833303/files/spectf_cloud_lbox.hdf5?download=1)
    - samples:
- [mmgis](https://zenodo.org/records/15833303/files/spectf_cloud_mmgis.hdf5?download=1)
    - samples:

## Test Best-F1 Scores:
| Model  | **Lbox + MMGISv1** | Thresh |
| ------ | ------------------ | ------ |
| v0.0.1 | 0.9519             | 0.52   |