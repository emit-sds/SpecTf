## Files:
- weights.pt
    - md5sum: 85d969fb86708175fcd2aaf488a19162
- model_no_bandcat_bf16.onnx
    - md5sum: c62d4d1f365ee233bc41daa5d1c660dd
- model.engine
    - md5sum: 99ac6abb33dff3b08f241cd7cdeab828

## Datasets:
- [labelbox](https://zenodo.org/records/15833303/files/spectf_cloud_lbox.hdf5?download=1)
    - samples:
- [mmgis](https://zenodo.org/records/15833303/files/spectf_cloud_mmgis.hdf5?download=1)
    - samples:
- [mmgis-v2](https://zenodo.org/records/15833303/files/spectf_cloud_mmgis_2.hdf5?download=1)
    - samples:

## Test Best-F1 Scores:
| Model             | **Lbox + MMGISv1 + MMGISv2** | Thresh | **Lbox + MMGISv1** | Thresh | **MMGISv2** | Thresh |
| ----------------- | ---------------------------- | ------ | ------------------ | ------ | ----------- | ------ |
| v0.0.1            | 0.9152                       | 0.55   | 0.9519             | 0.52   | 0.8604      | 0.53   |
| v0.1.0 (this one) | 0.9484                       | 0.51   | 0.9524             | 0.52   | 0.9428      | 0.51   |
