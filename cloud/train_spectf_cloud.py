import os
from datetime import datetime
import argparse
import h5py
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import schedulefree

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from sklearn.metrics import fbeta_score, roc_auc_score
import wandb

from dataset import SpectraDataset
from model import SimpleSeqClassifier

os.environ["WANDB__SERVICE_WAIT"] = "300"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classification model on tiled methane data.")

    parser.add_argument('dataset', nargs='+', type=str, help="Filepaths of the hdf5 datasets.")
    parser.add_argument('--train-csv', type=str, required=True, help="Filepath to train FID csv.")
    parser.add_argument('--test-csv', type=str, required=True, help="Filepath to test FID csv.")
    parser.add_argument('--outdir', type=str, default='./outdir', help="Output directory for models. Defaults to ./outdir.")
    parser.add_argument('--wandb-entity', type=str, default="", help="WandB project to be logged to.")
    parser.add_argument('--wandb-project', type=str, default="", help="WandB project to be logged to.")
    parser.add_argument('--wandb-name', type=str, default="", help="Project name to be appended to timestamp for wandb name.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs for training. Default is 50.")
    parser.add_argument('--batch', type=int, default=256, help="Batch size for training. Default is 256.")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate for training. Default is 0.0001.")
    parser.add_argument('--gpu', type=int, default=-1, help="GPU device to use. Default is -1 (cpu).")
    parser.add_argument('--arch-ff', type=int, default=64, help="Feed-forward dimensions. Default is 64.")
    parser.add_argument('--arch-heads', type=int, default=8, help="Number of heads for multihead attention. Default is 8.")
    parser.add_argument('--arch-dropout', type=float, default=0.1, help="Dropout percentage for overfit prevention. Default is 0.1.")
    parser.add_argument('--arch-agg', type=str, choices=['mean', 'max', 'flat'], default='max', help="Aggregate method prior to classification. Default is 'max'.")
    parser.add_argument('--arch-proj-dim', type=int, default=64, help="Projection dimensions. Default is 64.")

    args = parser.parse_args()

    # Importing the dataset
    for i, dataset in enumerate(args.dataset):
        if i == 0:
            f = h5py.File(dataset, 'r')
            labels = f['labels'][:]
            fids = f['fids'][:]
            spectra = f['spectra'][:]
            bands = f.attrs['bands']
        else:
            f = h5py.File(dataset, 'r')
            labels = np.concatenate((labels, f['labels'][:]))
            fids = np.concatenate((fids, f['fids'][:]))
            spectra = np.concatenate((spectra, f['spectra'][:]))

    print("Loaded dataset with shape:", spectra.shape)

    # Output directory
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using specified train/test splits.")
    # Open train csv
    with open(args.train_csv, 'r') as f:
        train_fids = [line.strip() for line in f.readlines()]
        train_fids = set(train_fids)
    # Only keep indices of fids that are in the train set
    train_i = [i for i, f in enumerate(fids) if f.decode('utf-8') in train_fids]

    # Open test csv
    with open(args.test_csv, 'r') as f:
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
    model = SimpleSeqClassifier(banddef,
                                num_classes=n_cls,
                                num_heads=args.arch_heads,
                                dim_proj=args.arch_proj_dim,
                                dim_ff=args.arch_ff,
                                dropout=args.arch_dropout,
                                agg=args.arch_agg).to(device)

    optimizer = schedulefree.AdamWScheduleFree((p for p in model.parameters() if p.requires_grad), lr=args.lr, warmup_steps=1000)

    # Define datasets - set device to CPU if model cannot fit on GPU
    train_dataset = SpectraDataset(train_X, train_y, transform=None, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_dataset = SpectraDataset(test_X, test_y, transform=None, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    # Define wandb
    timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S_%f_{args.wandb_name}")
    run = wandb.init(
        project = args.wandb_project,
        entity = 'jpl-cmml',
        name = timestamp,
        dir = './',
        config = {
            'dataset_path': args.dataset,
            'lr': args.lr,
            'epochs': args.epochs,
            'batch': args.batch,
            'arch_ff': args.arch_ff,
            'arch_heads': args.arch_heads,
            'arch_dropout': args.arch_dropout,
            'arch_proj_dim': args.arch_proj_dim,
            'arch_agg': args.arch_agg
        },
        settings=wandb.Settings(_service_wait=300)
    )

    # Training loop
    for epoch in range(args.epochs):
        # Set model and optimizer to train mode
        model.train()
        optimizer.train()

        train_epoch_loss = 0
        for idx, batch in enumerate(train_dataloader):
            spectra = batch['spectra'].to(device)
            labels = batch['label'].to(device)

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
        for idx, batch in enumerate(train_dataloader):
            spectra = batch['spectra'].to(device)
            labels = batch['label'].to(device)

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
        for idx, batch in enumerate(test_dataloader):
            spectra = batch['spectra'].to(device)
            labels = batch['label'].to(device)

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

        torch.save(model.state_dict(), os.path.join(args.outdir, f"{timestamp}_{epoch}.pt"))

run.finish()
