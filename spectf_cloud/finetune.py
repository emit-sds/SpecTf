import sys
import os
from datetime import datetime
import h5py
import rich_click as click

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import schedulefree

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, 
    average_precision_score, 
    fbeta_score, 
    roc_auc_score, 
    accuracy_score
)
import wandb

from spectf.dataset import SpectraDataset
from spectf.model import SpecTfEncoder
from spectf.utils import seed as useed
from spectf.utils import get_device
from spectf_cloud.cli import spectf_cloud, MAIN_CALL_ERR_MSG

os.environ["WANDB__SERVICE_WAIT"] = "300"
torch.autograd.set_detect_anomaly(True)

ENV_VAR_PREFIX = 'SPECTF_FINETUNE_'

@click.argument(
    "dataset",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    envvar=f'{ENV_VAR_PREFIX}DATASET'
)
@click.option(
    "--weights",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Filepath to base model weights.",
    envvar=f'{ENV_VAR_PREFIX}WEIGHTS'
)
@click.option(
    "--freeze",
    required=True,
    type=str,
    help="Freeze level. Freeze up to 'attention' or the entire 'encoder'",
    envvar=f'{ENV_VAR_PREFIX}FREEZE'
)
@click.option(
    "--train-csv",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Filepath to train FID csv.",
    envvar=f'{ENV_VAR_PREFIX}TRAIN_CSV'
)
@click.option(
    "--test-csv",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Filepath to test FID csv.",
    envvar=f'{ENV_VAR_PREFIX}TEST_CSV'
)
@click.option(
    "--outdir",
    default="./outdir",
    show_default=True,
    help="Output directory for models.",
    envvar=f'{ENV_VAR_PREFIX}OUTDIR'
)
@click.option(
    "--wandb-entity",
    default="",
    show_default=True,
    help="WandB entity.",
    envvar=f'{ENV_VAR_PREFIX}WANDB_ENTITY'
)
@click.option(
    "--wandb-project",
    default="",
    show_default=True,
    help="WandB project to be logged to.",
    envvar=f'{ENV_VAR_PREFIX}WANDB_PROJECT'
)
@click.option(
    "--wandb-name",
    default="",
    show_default=True,
    help="Project name to be appended to timestamp for wandb name.",
    envvar=f'{ENV_VAR_PREFIX}WANDB_NAME'
)
@click.option(
    "--epochs",
    default=10,
    type=int,
    show_default=True,
    help="Number of epochs for training.",
    envvar=f'{ENV_VAR_PREFIX}EPOCHS'
)
@click.option(
    "--batch",
    default=256,
    type=int,
    show_default=True,
    help="Batch size for training.",
    envvar=f'{ENV_VAR_PREFIX}BATCH'
)
@click.option(
    "--lr",
    default=0.0001,
    type=float,
    show_default=True,
    help="Learning rate for training.",
    envvar=f'{ENV_VAR_PREFIX}LR'
)
@click.option(
    "--gpu",
    default=None,
    type=int,
    show_default=True,
    help="GPU device to use.",
    envvar=f'{ENV_VAR_PREFIX}GPU'
)
@click.option(
    "--arch-ff",
    default=64,
    type=int,
    show_default=True,
    help="Feed-forward dimensions.",
    envvar=f'{ENV_VAR_PREFIX}ARCH_FF'
)
@click.option(
    "--arch-heads",
    default=8,
    type=int,
    show_default=True,
    help="Number of heads for multihead attention.",
    envvar=f'{ENV_VAR_PREFIX}ARCH_HEADS'
)
@click.option(
    "--arch-dropout",
    default=0.1,
    type=float,
    show_default=True,
    help="Dropout percentage for overfit prevention.",
    envvar=f'{ENV_VAR_PREFIX}ARCH_DROPOUT'
)
@click.option(
    "--arch-agg",
    default="max",
    type=click.Choice(["mean", "max", "flat"]),
    show_default=True,
    help="Aggregate method prior to classification.",
    envvar=f'{ENV_VAR_PREFIX}ARCH_AGG'
)
@click.option(
    "--arch-proj-dim",
    default=64,
    type=int,
    show_default=True,
    help="Projection dimensions.",
    envvar=f'{ENV_VAR_PREFIX}ARCH_PROJ_DIM'
)
@click.option(
    "--seed",
    default=42,
    type=int,
    show_default=True,
    help="Training run seed.",
    envvar=f'{ENV_VAR_PREFIX}SEED'
)
@click.option(
    "--save-every-epoch",
    is_flag=True,
    default=False,
    help="Save the model's state every epoch.",
    envvar=f'{ENV_VAR_PREFIX}SAVE_EVERY_EPOCH'
)

@spectf_cloud.command(
    add_help_option=True,
    help="Finetune the SpecTf Hyperspectral Transformer Model."
)
def finetune(
    dataset: list,
    weights: str,
    freeze: str,
    train_csv: str,
    test_csv: str,
    outdir: str,
    wandb_entity: str,
    wandb_project: str,
    wandb_name: str,
    epochs: int,
    batch: int,
    lr: float,
    gpu: int,
    arch_ff: int,
    arch_heads: int,
    arch_dropout: float,
    arch_agg: str,
    arch_proj_dim: int,
    seed: int,
    save_every_epoch:bool,
):
    # Set seed
    useed(seed)

    # Importing the dataset
    for i, ds in enumerate(dataset):
        f = h5py.File(ds, 'r')
        bands = f.attrs['bands']
        if i == 0:
            labels = f['labels'][:]
            fids = f['fids'][:]
            spectra = f['spectra'][:]
        else:
            labels = np.concatenate((labels, f['labels'][:]))
            fids = np.concatenate((fids, f['fids'][:]))
            spectra = np.concatenate((spectra, f['spectra'][:]))

    print("Loaded dataset with shape:", spectra.shape)

    # Output directory
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # Device
    device = get_device(gpu)

    print("Using specified train/test splits.")
    # Open train csv
    with open(train_csv, 'r', encoding='utf-8') as f:
        train_fids = [line.strip() for line in f.readlines()]
        train_fids = set(train_fids)
    # Only keep indices of fids that are in the train set
    train_i = [i for i, f in enumerate(fids) if f.decode('utf-8') in train_fids]

    # Open test csv
    with open(test_csv, 'r', encoding='utf-8') as f:
        test_fids = [line.strip() for line in f.readlines()]
        test_fids = set(test_fids)
    # Only keep indices of fids that are in the test set
    test_i = [i for i, f in enumerate(fids) if f.decode('utf-8') in test_fids]

    # Define train/test split
    train_X = spectra[train_i]
    train_y = labels[train_i]
    test_X = spectra[test_i]
    test_y = labels[test_i]

    print(f"Split Train: {len(train_X)} Test: {len(test_X)}")

    # Define model
    n_cls = 2
    criterion = nn.CrossEntropyLoss()

    banddef = torch.tensor(bands, dtype=torch.float32).to(device)
    model = SpecTfEncoder(banddef,
                          dim_output=n_cls,
                          num_heads=arch_heads,
                          dim_proj=arch_proj_dim,
                          dim_ff=arch_ff,
                          dropout=arch_dropout,
                          agg=arch_agg,
                          use_residual=False,
                          num_layers=1).to(device)

    model.load_state_dict(torch.load(weights, map_location=device))

    if freeze == "attention":
        for p in model.spectral_embed.parameters():
            p.requires_grad = False
        for p in model.layers[0].attention.parameters():
            p.requires_grad = False
    elif freeze == "encoder":
        for p in model.spectral_embed.parameters():
            p.requires_grad = False
        for p in model.layers[0].attention.parameters():
            p.requires_grad = False
        for p in model.layers[0].ff.parameters():
            p.requires_grad = False

    optimizer = schedulefree.AdamWScheduleFree((p for p in model.parameters() if p.requires_grad), lr=lr, warmup_steps=1000)

    # Define datasets - set device to CPU if model cannot fit on GPU
    train_dataset = SpectraDataset(train_X, train_y, transform=None, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_dataset = SpectraDataset(test_X, test_y, transform=None, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    # Define wandb
    timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S_%f_{wandb_name}")
    try:
        run = wandb.init(
            project = wandb_project,
            entity = wandb_entity,
            name = timestamp,
            dir = './',
            config = {
                'dataset_path': dataset,
                'lr': lr,
                'epochs': epochs,
                'batch': batch,
                'arch_ff': arch_ff,
                'arch_heads': arch_heads,
                'arch_dropout': arch_dropout,
                'arch_proj_dim': arch_proj_dim,
                'arch_agg': arch_agg
            },
            settings=wandb.Settings(_service_wait=300)
        )
    except Exception as e:
        print("WandB error!")
        print(e)
        sys.exit(1)

    # Training loop
    for epoch in range(epochs):
        # Set model and optimizer to train mode
        model.train()
        optimizer.train()

        train_epoch_loss = 0
        for batch_ in train_dataloader:
            spectra = batch_['spectra'].to(device).float()
            labels = batch_['label'].to(device)

            optimizer.zero_grad()
            pred = model(spectra)
            loss = criterion(pred, labels)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_epoch_loss += loss.cpu().item()
            run.log({
                "loss_train": loss.cpu().item()
            })

        train_epoch_loss /= len(train_dataloader)

        ## MODEL EVALUATION
        model.eval()
        optimizer.eval()
        train_true = []
        train_pred = []
        train_proba = []
        train_proba_full = []
        for idx, batch_ in enumerate(train_dataloader):
            spectra = batch_['spectra'].to(device)
            labels = batch_['label'].to(device)

            with torch.no_grad():
                pred = model(spectra)

                # For multiclass
                proba = nn.functional.softmax(pred, dim=1)
                predcls = torch.argmax(proba, dim=1).cpu().tolist()

                labels_cpu = labels.cpu().tolist()
                proba_cpu = proba.cpu().tolist()

                train_true += labels_cpu
                train_pred += predcls
                train_proba += [p[1] for p in proba_cpu]
                train_proba_full += proba_cpu

        train_true = np.array(train_true) 
        train_pred = np.array(train_pred)
        train_proba = np.array(train_proba)

        # Accuracy
        train_acc = accuracy_score(train_true, train_pred)

        # Precision, Recall, F1
        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(train_true, train_pred, pos_label=1, average='binary')
        train_pr_curve = wandb.plot.pr_curve(train_true, train_proba_full, labels=['clear', 'cloud'], classes_to_plot=[1])
        train_ap = average_precision_score(train_true, train_proba)

        # TPR, FPR, ROC
        train_roc = wandb.plot.roc_curve(train_true, train_proba_full, labels=['clear', 'cloud'], classes_to_plot=[1])
        train_auc = roc_auc_score(train_true, train_proba)

        # True Positive Rate
        train_tpr = np.sum((train_true == 1) & (train_pred == 1)) / np.sum(train_true == 1)
        # False Positive Rate
        train_fpr = np.sum((train_true == 0) & (train_pred == 1)) / np.sum(train_true == 0)

        # Best threshold for F1, F0.5, F0.25
        thresholds = np.around(np.arange(0, 1.01, 0.01), 2)
        f1s = []
        f05s = []
        f025s = []
        for t in thresholds:
            curr_pred = (train_proba >= t).astype(int)
            f1s.append(fbeta_score(train_true, curr_pred, beta=1))
            f05s.append(fbeta_score(train_true, curr_pred, beta=0.5))
            f025s.append(fbeta_score(train_true, curr_pred, beta=0.25))
        train_best_f1 = np.max(f1s)
        train_best_f05 = np.max(f05s)
        train_best_f025 = np.max(f025s)

        test_true = []
        test_pred = []
        test_proba = []
        test_proba_full = []
        test_epoch_loss = 0
        for idx, batch_ in enumerate(test_dataloader):
            spectra = batch_['spectra'].to(device)
            labels = batch_['label'].to(device)

            with torch.no_grad():
                pred = model(spectra)
                loss = criterion(pred, labels)
                test_epoch_loss += loss.cpu().item()

                proba = nn.functional.softmax(pred, dim=1)
                predcls = torch.argmax(proba, dim=1).cpu().tolist()

                labels_cpu = labels.cpu().tolist()
                proba_cpu = proba.cpu().tolist()

                test_true += labels_cpu
                test_pred += predcls
                test_proba += [p[1] for p in proba_cpu]
                test_proba_full += proba_cpu

        test_epoch_loss /= len(test_dataloader)

        test_true = np.array(test_true) 
        test_pred = np.array(test_pred)
        test_proba = np.array(test_proba)

        # Accuracy
        test_acc = accuracy_score(test_true, test_pred)

        # Precision, Recall, F1
        test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(test_true, test_pred, pos_label=1, average='binary')
        test_pr_curve = wandb.plot.pr_curve(test_true, test_proba_full, labels=['clear', 'cloud'], classes_to_plot=[1])
        test_ap = average_precision_score(test_true, test_proba)

        # TPR, FPR, ROC
        test_roc = wandb.plot.roc_curve(test_true, test_proba_full, labels=['clear', 'cloud'], classes_to_plot=[1])
        test_auc = roc_auc_score(test_true, test_proba)


        # True Positive Rate
        test_tpr = np.sum((test_true == 1) & (test_pred == 1)) / np.sum(test_true == 1)
        # False Positive Rate
        test_fpr = np.sum((test_true == 0) & (test_pred == 1)) / np.sum(test_true == 0)

        # Best threshold for F1, F0.5, F0.25
        thresholds = np.around(np.arange(0, 1.01, 0.01), 2)
        f1s = []
        f05s = []
        f025s = []
        for t in thresholds:
            curr_pred = (test_proba >= t).astype(int)
            f1s.append(fbeta_score(test_true, curr_pred, beta=1))
            f05s.append(fbeta_score(test_true, curr_pred, beta=0.5))
            f025s.append(fbeta_score(test_true, curr_pred, beta=0.25))
        test_best_f1 = np.max(f1s)
        test_best_f05 = np.max(f05s)
        test_best_f025 = np.max(f025s)

        run.log({
            "train/loss": train_epoch_loss,
            "train/accuracy": train_acc,
            "train/precision": train_prec,
            "train/recall": train_rec,
            "train/f1": train_f1,
            "train/pr_curve": train_pr_curve,
            "train/ap": train_ap,
            "train/roc_curve": train_roc,
            "train/roc_auc": train_auc,
            "train/tpr": train_tpr,
            "train/fpr": train_fpr,
            "train/best_f100": train_best_f1,
            "train/best_f050": train_best_f05,
            "train/best_f025": train_best_f025,
            "test/loss": test_epoch_loss,
            "test/accuracy": test_acc,
            "test/precision": test_prec,
            "test/recall": test_rec,
            "test/f1": test_f1,
            "test/pr_curve": test_pr_curve,
            "test/ap": test_ap,
            "test/roc_curve": test_roc,
            "test/roc_auc": test_auc,
            "test/tpr": test_tpr,
            "test/fpr": test_fpr,
            "test/best_f100": test_best_f1,
            "test/best_f050": test_best_f05,
            "test/best_f025": test_best_f025,
            "epoch": epoch
        })
        if save_every_epoch:
            torch.save(model.state_dict(), os.path.join(outdir, f"spectf_cloud_{timestamp}_{epoch}.pt"))
    if not save_every_epoch:
        torch.save(model.state_dict(), os.path.join(outdir, f"spectf_cloud_{timestamp}.pt"))
    run.finish()

if __name__ == "__main__":
    print(MAIN_CALL_ERR_MSG % "finetune")