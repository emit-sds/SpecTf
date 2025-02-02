import rich_click as click
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from sklearn.metrics import fbeta_score, roc_auc_score

from spectf.model import SpecTfEncoder
from spectf.dataset import SpectraDataset
from spectf.utils import seed
from spectf_cloud.evaluation import cloud_eval

ENV_VAR_PREFIX = "SPECTF_EVAL_SPECTF_"

torch.autograd.set_detect_anomaly(True)

@click.argument(
    "dataset",
    nargs=-1,  # allow multiple dataset paths
    type=click.Path(exists=True),
    required=True,
    envvar=f"{ENV_VAR_PREFIX}DATASET",
)
@click.option(
    "--weights",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Filepath to model weights.",
    envvar=f"{ENV_VAR_PREFIX}WEIGHTS"
)
@click.option(
    "--test-csv",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Filepath to test FID csv.",
    envvar=f"{ENV_VAR_PREFIX}TEST_CSV"
)
@click.option(
    "--batch",
    default=1024,
    type=int,
    show_default=True,
    help="Batchsize for eval. Default 1024.",
    envvar=f"{ENV_VAR_PREFIX}BATCH"
)
@click.option(
    "--gpu",
    default=-1,
    type=int,
    show_default=True,
    help="GPU device to use. Default -1 (CPU).",
    envvar=f"{ENV_VAR_PREFIX}GPU"
)
@click.option(
    "--arch-ff",
    default=64,
    type=int,
    show_default=True,
    help="Feed-forward dimensions.",
    envvar=f"{ENV_VAR_PREFIX}ARCH_FF"
)
@click.option(
    "--arch-heads",
    default=8,
    type=int,
    show_default=True,
    help="Number of heads for multihead attention.",
    envvar=f"{ENV_VAR_PREFIX}ARCH_HEADS"
)
@click.option(
    "--arch-proj-dim",
    default=64,
    type=int,
    show_default=True,
    help="Projection dimensions.",
    envvar=f"{ENV_VAR_PREFIX}ARCH_PROJ_DIM"
)
@click.option(
    "--thresh",
    default=0.52,
    type=float,
    show_default=True,
    help="Cloud classification posterior score threshold.",
    envvar=f"{ENV_VAR_PREFIX}THRESH"
)
@cloud_eval.command(
    add_help_option=True,
    help="Evaluate the SpecTf model with test data."
)
def spectf(dataset, weights, test_csv, batch, gpu, arch_ff, arch_heads, arch_proj_dim, thresh):

    seed()

    # Importing the dataset
    for i, ds in enumerate(dataset):
        if i == 0:
            f = h5py.File(ds, 'r')
            labels = f['labels'][:]
            fids = f['fids'][:]
            spectra = f['spectra'][:]
            bands = f.attrs['bands']
        else:
            f = h5py.File(ds, 'r')
            labels = np.concatenate((labels, f['labels'][:]))
            fids = np.concatenate((fids, f['fids'][:]))
            spectra = np.concatenate((spectra, f['spectra'][:]))

    print("Loaded dataset with shape:", spectra.shape)

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # apple silicon
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Generate a train/test split that prevents FID leakage
    if test_csv:
        # Open test csv
        with open(test_csv, 'r') as f:
            test_fids = [line.strip() for line in f.readlines()]
            test_fids = set(test_fids)
        # Only keep indices of fids that are in the test set
        test_i = [i for i, f in enumerate(fids) if f.decode('utf-8') in test_fids]
    else:
        raise ValueError("Must specify test csv.")

    # Define train/test split
    test_X = spectra[test_i]
    test_y = labels[test_i]

    # Define model
    n_cls = 2
    banddef = torch.tensor(bands, dtype=torch.float32).to(device)
    model = SpecTfEncoder(banddef,
                          num_classes=n_cls,
                          num_heads=arch_heads,
                          dim_proj=arch_proj_dim,
                          dim_ff=arch_ff,
                          dropout=0).to(device)

    # Load weights
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    
    # Define datasets - replace OnDevice if not using GPU where the dataset can fit in the VRAM
    test_dataset = SpectraDataset(test_X, test_y, transform=None, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    
    test_true = []
    test_pred = []
    test_proba = []
    test_proba_full = []
    for idx, batch_ in enumerate(tqdm(test_dataloader, desc="Test", total=len(test_dataloader))):
        spectra = batch_['spectra'].to(device)
        labels = batch_['label'].to(device)

        with torch.no_grad():
            pred = model(spectra)

            # For multiclass
            proba = nn.functional.softmax(pred, dim=1)
            predcls = torch.argmax(proba, dim=1).cpu().tolist()

            labels_cpu = labels.cpu().tolist()
            proba_cpu = proba.cpu().tolist()

            test_true += labels_cpu
            #test_pred += predcls
            test_pred += [p[1] >= thresh for p in proba_cpu]
            test_proba += [p[1] for p in proba_cpu]
            test_proba_full += proba_cpu

    test_true = np.array(test_true) 
    test_pred = np.array(test_pred)
    test_proba = np.array(test_proba)

    # Accuracy
    test_acc = accuracy_score(test_true, test_pred)

    # Precision, Recall, F1
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(test_true, test_pred, pos_label=1, average='binary')
    test_ap = average_precision_score(test_true, test_proba)
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall:    {test_rec:.4f}")
    print(f"Test AP: {test_ap:.4f}")

    # TPR, FPR, ROC
    test_auc = roc_auc_score(test_true, test_proba)

    print(f"Test AUC: {test_auc:.4f}")

    # True Positive Rate
    test_tpr = np.sum((test_true == 1) & (test_pred == 1)) / np.sum(test_true == 1)
    # False Positive Rate
    test_fpr = np.sum((test_true == 0) & (test_pred == 1)) / np.sum(test_true == 0)
    print(f"Test TPR: {test_tpr:.4f}")
    print(f"Test FPR: {test_fpr:.4f}")

    print(f"{thresh:.2f} F1 : {fbeta_score(test_true, test_pred, beta=1)}")
    print(f"{thresh:.2f} F05 : {fbeta_score(test_true, test_pred, beta=0.5)}")
    print(f"{thresh:.2f} F025: {fbeta_score(test_true, test_pred, beta=0.25)}")
    print(f"{thresh:.2f} F01 : {fbeta_score(test_true, test_pred, beta=0.10)}")

    # Best threshold for F1, F0.5, F0.25
    thresholds = np.around(np.arange(0, 1.01, 0.01), 2)
    f1s = []
    f05s = []
    f025s = []
    f01s = []
    for t in thresholds:
        curr_pred = (test_proba >= t).astype(int)
        f1s.append(fbeta_score(test_true, curr_pred, beta=1))
        f05s.append(fbeta_score(test_true, curr_pred, beta=0.5))
        f025s.append(fbeta_score(test_true, curr_pred, beta=0.25))
        f01s.append(fbeta_score(test_true, curr_pred, beta=0.1))
    test_best_f1 = np.max(f1s)
    test_best_f05 = np.max(f05s)
    test_best_f025 = np.max(f025s)
    test_best_f01 = np.max(f01s)
    test_best_th_f1 = thresholds[np.argmax(f1s)]
    test_best_th_f05 = thresholds[np.argmax(f05s)]
    test_best_th_f025 = thresholds[np.argmax(f025s)]
    test_best_th_f01 = thresholds[np.argmax(f01s)]

    print(f"Test F1   : {test_best_f1:.4f} @ {test_best_th_f1:.2f}")
    print(f"Test F0.5 : {test_best_f05:.4f} @ {test_best_th_f05:.2f}")
    print(f"Test F0.25: {test_best_f025:.4f} @ {test_best_th_f025:.2f}")
    print(f"Test F0.1 : {test_best_f01:.4f} @ {test_best_th_f01:.2f}")
