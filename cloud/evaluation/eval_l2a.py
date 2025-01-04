from datetime import datetime
import argparse
import h5py

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score, roc_auc_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classification model on tiled methane data.")

    parser.add_argument('dataset',          help="Filepaths of the hdf5 datasets",
                                            nargs='+',
                                            type=str)
    parser.add_argument('--test-csv',       help="Filepath to test FID csv.",
                                            type=str,
                                            required=True)

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

    # Generate a train/test split that prevents FID leakage
    if args.test_csv:
        # Open test csv
        with open(args.test_csv, 'r') as f:
            test_fids = [line.strip() for line in f.readlines()]
            test_fids = set(test_fids)
        # Only keep indices of fids that are in the test set
        test_i = [i for i, f in enumerate(fids) if f.decode('utf-8') in test_fids]
    else:
        raise ValueError("Must specify test csv.")

    # Define train/test split
    test_X = spectra[test_i]
    test_y = labels[test_i]
    banddef = np.array(bands).astype(float)
    
    b450 = test_X[:,np.argmin(np.abs(banddef - 450))]
    b1250 = test_X[:,np.argmin(np.abs(banddef - 1250))]
    b1380 = test_X[:,np.argmin(np.abs(banddef - 1380))]
    b1650 = test_X[:,np.argmin(np.abs(banddef - 1650))]

    cloud = (b450 > 0.28) & (b1250 > 0.46) & (b1650 > 0.22)
    cloud = cloud.astype(int)
    cirrus = (b1380 > 0.1)

    test_true = np.array(test_y)
    test_true[test_true==2] = 0

    test_pred = cloud | cirrus
    test_pred = test_pred.astype(int).flatten()

    # Accuracy
    test_acc = accuracy_score(test_true, test_pred)

    # True Positive Rate
    test_tpr = np.sum((test_true == 1) & (test_pred == 1)) / np.sum(test_true == 1)
    # False Positive Rate
    test_fpr = np.sum((test_true == 0) & (test_pred == 1)) / np.sum(test_true == 0)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test TPR: {test_tpr:.4f}")
    print(f"Test FPR: {test_fpr:.4f}")

    test_best_f1 = fbeta_score(test_true, test_pred, beta=1)
    test_best_f05 = fbeta_score(test_true, test_pred, beta=0.5)
    test_best_f025 = fbeta_score(test_true, test_pred, beta=0.25)
    test_best_f01 = fbeta_score(test_true, test_pred, beta=0.1)

    print(f"Test F1: {test_best_f1:.4f}")
    print(f"Test F0.5: {test_best_f05:.4f}")
    print(f"Test F0.25: {test_best_f025:.4f}")
    print(f"Test F0.1: {test_best_f01:.4f}")

    test_auc = roc_auc_score(test_true, test_pred)
    print(f"Test AUC: {test_auc:.4f}")