# Datasets

This directory contains dataset-related files.

The ML-ready dataset of TOA reflectance spectra, labels, and metadata are available on Zenodo:

> Lee, J., Kiper, M., Thompson, D. R., & Brodrick, P. (2025). Dataset for "Spectroscopic Transformer for Improved EMIT Cloud Masks" [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14607938

For usage examples, the HDF5 files should be downloaded and saved in this directory. The CSV files are identical.

The `test_fids.csv` and `train_fids.csv` define the train-test split by scene. Ensuring that scenes don't contribute spectra to both datasets ensures that overfitting is prevented, as spectra from adjacent pixels can be very similar.

We are not releasing the label rasters for the scenes in question because labeles were produced ***sparsely*** - that is, not every pixel in every scene was labeled, and a limited number of pixels were sampled from each class from each scene. This makes the raster dataset unsuitable for typical image segmentation evaluation. Please refer to the paper for more details.