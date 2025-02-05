<h1 align="center">Spectroscopic Transformer</h1>

This repository contains code for the Spectroscopic Transformer, a pixel-wise deep learning model for spectral tasks.

Subdirectories of this repository contain research code for specific publications and/or applications.

> [!NOTE]
> Active development on a more formalized package for easier installation and deployment is ongoing in the [`dev`](https://github.com/emit-sds/SpecTf/tree/dev) branch.

## Dependencies

> [!NOTE]
> Conda environment definitions will be provided in the future.

- [Pytorch](https://pytorch.org/get-started/locally/)
- [ISOFIT](https://isofit.readthedocs.io/en/latest/custom/installation.html)
- [Spectral Python](https://www.spectralpython.net/installation.html)
- [Rasterio](https://rasterio.readthedocs.io/en/stable/installation.html)
- [Schedulefree](https://github.com/facebookresearch/schedule_free)
- `sklearn`
- `pyyaml`
- `numpy`
- (for training logging) [WandB](https://docs.wandb.ai/quickstart/)

## Publications

### SpecTf: Transformers Enable Data-Driven Imaging Spectroscopy Cloud Detection

Jake H. Lee, Michael Kiper, David R. Thompson, Philip G. Brodrick. *In Review* 

Preprint: https://arxiv.org/abs/2501.04916

For details see the [cloud directory](https://github.com/emit-sds/SpecTf/tree/main/cloud)
