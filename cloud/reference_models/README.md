# Reference Models

This directory contains source code of the reference models developed for "Spectroscopic Transformer for Improved EMIT
Cloud Masks."

The Gradient Boosted Tree (GBT) model referenced in the paper is referred to as "XGBoost" here. The Artificial Neural Network (ANN) model referenced in the paper is referred to as "ResNet" here. This is **not** the more common ResNet architecture for computer vision tasks - it is a small network of fully connected layers with residual connections.

## Usage

**GBT**

```python
from xgb import make_xgb_model

model = make_xgb_model(
    arch_yml = 'xgboost_arch.yml', 
    arch_subkeys = ["architecture"],        # optional - tells the function where to start parsing the yaml file
    weight_file = 'xgboost_reference.json'  # optional- loads in this xgboost model
)
```

**ANN**

```python
from ResNet import make_model

model = make_model(
    arch_yml= 'resnet_arch.yml',
    input_dim=268,
    num_classes=2,
    arch_subkeys=['architecture'],      # this is optional - tells the funtion where to start parsing the yaml file
    weight_file='resnet_reference.pt'   # this is optional
)
```

## Training

GBT and ANN models can be reproduced with the scripts provided in this repository.

**GBT**

```
$ python train_xgb.py -data_path dataset.hdf5 -model_yaml xgboost_arch.yml -train_split train_fids.csv -test_split test_fids.csv
```

**ANN**

```
$ python train_resnet.py -data_path dataset.hdf5 -model_yaml resnet_arch.yml -train_split train_fids.csv -test_split test_fids.csv
```