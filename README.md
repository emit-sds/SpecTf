<h1 align="center">Spectroscopic Transformer</h1>

This repository contains code for the Spectroscopic Transformer, a pixel-wise deep learning model for spectral tasks.

Subdirectories of this repository contain research code for specific publications and/or applications.

## Included Packages

Installation of these packages require an [existing installation of GDAL](https://gdal.org/en/stable/download.html).
- libgdal needs to be version 3.9.3: `conda install conda-forge::libgdal==3.9.3`

### [SpecTf Cloud Masking Model](https://github.com/emit-sds/SpecTf/tree/main/spectf_cloud)
<p align="center">
  <img src="https://raw.githubusercontent.com/emit-sds/SpecTf/refs/heads/dev/spectf_cloud/figures/fig4.png" alt="SpecTf Cloud Model Image" width="300" height="250">
</p>

#### Decription
This package includes both an importable <ins>Python package</ins> and a <ins>CLI interface</ns> for model deployment and training.

#### Getting Started
Check out the docs [here](https://github.com/emit-sds/SpecTf/blob/dev/spectf_cloud/README.md)!


## Publications
### SpecTf: Transformers Enable Data-Driven Imaging Spectroscopy Cloud Detection

Jake H. Lee, Michael Kiper, David R. Thompson, Philip G. Brodrick. *In Review* 

Preprint: https://arxiv.org/abs/2501.04916

For details see the [cloud directory](https://github.com/emit-sds/SpecTf/tree/dev/spectf_cloud)

## Envrionment Variables
This section is to provide a list of all of the envrionment variables related to this package and how they impact the functional behavior.

### SpecTf

### SpecTf Cloud
`*` -> replace with the uppercase name of the parameter and any dashes `-` to be underscores `_`. (i.e. `--arch-proj-dim` -> `SPECTF_TRAIN_ARCH_PROJ_DIM`)
| Evr Var Name | Description |
|------|-------------|
| **SPECTF_TRAIN_\*** | Set to override the default parameters values of the SpecTf model training function. (See `spectf-cloud train -h` for list of parameters and their defaults) |
| **SPECTF_DEPLOY_\*** | Set to override the default parameters values of the SpecTf model deploy function. (See `spectf-cloud deploy -h` for list of parameters and their defaults) |
| **SPECTF_EVAL_L2A_\*** | Set to override the default parameters values of the L2A evaluation function. (See `spectf-cloud cloud-eval l2a -h` for list of parameters and their defaults) |
| **SPECTF_EVAL_SPECTF_\*** | Set to override the default parameters values of the SpecTf evaluation function. (See `spectf-cloud cloud-eval spectf -h` for list of parameters and their defaults) |
| **RESNET_TRAIN_\*** | Set to override the default parameters values of the ResNet training function. (See `spectf-cloud train-comparison resnet -h` for list of parameters and their defaults) |
| **XGBOOST_TRAIN_\*** | Set to override the default parameters values of the XGBoost training function. (See `spectf-cloud train-comparison xgboost -h` for list of parameters and their defaults) |

