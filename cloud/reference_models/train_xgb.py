import xgboost as xgb
import argparse
import h5py
import numpy as np
import yaml
import logging
import datetime
import os
import time
import wandb
from training_utils import utils
from typing import Optional
from sklearn.metrics import fbeta_score, log_loss, confusion_matrix

def save_f_beta_scores(model:xgb.XGBClassifier, X:np.ndarray, Y:np.ndarray, log_dir:str, beta_values=[0.1, 0.25, 0.5, 1, 2]):
    if Y.ndim == 2: 
        Y = np.argmax(Y, axis=1)
        
    probabilities = model.predict_proba(X)[:, 1]
    bce_loss = log_loss(Y, probabilities, labels=[0, 1])
    
    # get the best F1 threshold
    thresholds = np.around(np.arange(0, 1.01, 0.01), 2)
    f1s = []
    for t in thresholds:
        curr_pred = (probabilities[:, 1] >= t).detach().numpy().astype(int)
        f1s.append(fbeta_score(Y, curr_pred, beta=1))
    test_best_f1 = thresholds[np.argmax(f1s)]

    f_beta_scores = {}
    for beta in beta_values:
        f_beta = fbeta_score(Y, probabilities[:, 1] > test_best_f1, beta=beta)
        f_beta_scores[f'F-{beta}'] = f_beta

    wandb.log({
        **{f"test_{k}": v for k, v in f_beta_scores.items()}
    })
    
    best_pred = (probabilities[:, 1] >= test_best_f1).to(dtype=int)
    tn, fp, fn, tp = confusion_matrix(Y, best_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    f_beta_scores.update({'Log Loss':bce_loss})
    f_beta_scores.update({'TPR':tpr})
    f_beta_scores.update({'FPR':fpr})
    f_beta_scores.update({'Best F1 Thresh':test_best_f1})
    
    with h5py.File(os.path.join(log_dir, "xgb_f_beta_score_tracking.hdf5"), 'w') as f:
        for k, v in f_beta_scores.items():
            dataset_name = f"{k}"
            f.create_dataset(dataset_name, data=[v], maxshape=(None,))
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str, help='Reflectance data path', required=True)
    parser.add_argument('-model_yaml', type=str, help='Hyperparameters for XGBoost file path', required=True)
    parser.add_argument('-train_split', type=str, help='The path to a text file containing the desired FID train split', required=True)
    parser.add_argument('-test_split', type=str, help='The path to a text file containing the desired FID test split', required=True)
    parser.add_argument('-log_dir', type=str, help='Directory to save log', required=False, default='.')
    parser.add_argument('-wandb_name', type=str, help='W&B run name', required=False)
    args = parser.parse_args()

    wandb.init()
    if args.wandb_name:
        custom_name = args.wandb_name
    else:
        custom_name = f"XGBoost-{wandb.run.id}-{args.data_path.split('/')[-1].split('.')[0]}"
    wandb.run.name = custom_name
    wandb.run.save()

    run(args.data_path, args.model_yaml, args.train_split, args.test_split, args.log_dir, args.wandb_name)

def run(
        data_path:str, 
        model_yaml:str, 
        train_split:str, 
        test_split:str, 
        log_dir:str='.', 
        wandb_name:Optional[str]=None, 
        ):

    ## create logging ##############################################################
    today_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    prefix = wandb_name if wandb_name is not None else "XGBoost"
    log_filename = os.path.join(log_dir, prefix+f"_{today_date}.log")
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
                        )   
    logging.info("Starting process..")
    ################################################################################

    with open(model_yaml, 'r') as f:
        model_hyperparameters = yaml.safe_load(f)
    
    ## load in data and create train/test split ####################################
    hdf5 = h5py.File(data_path, 'r')    
    spectra = hdf5['spectra'][:]
    labels = hdf5['labels'][:]
    all_fids = hdf5['fids'][:]

    ## Preprocess the data
    logging.info('Preprocessing data...')
    start = time.time()

    train_indices, test_indices, train_fids, test_fids = utils.gen_train_test_split(fids=all_fids, 
                                                                              train_split=train_split, 
                                                                              test_split=test_split, 
                                                                              return_fids=True)

    X_train, Y_train = spectra[train_indices], labels[train_indices]
    X_test, Y_test = spectra[test_indices], labels[test_indices]

    logging.info('Training fids:')
    logging.info(list(train_fids))
    logging.info('Testing fids:')
    logging.info(list(test_fids))

    end = time.time()
    logging.info(f'Preprocessing complete: {(end-start):.2f}')
    
    ################################################################################


    ## create model ################################################################
    model = xgb.XGBClassifier(
        **model_hyperparameters['architecture']
    )
    ################################################################################

    ## training loop ###############################################################
    logging.info('Starting training...')
    start = time.time()

    model.fit(X_train, Y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    val_acc = 100. * np.mean(y_pred == Y_test)
    fbeta_scores = utils.calculate_fbeta_scores(Y_test, y_pred)

    wandb.log({
        "test_acc": val_acc,
        **{f"test_{k}": v for k, v in fbeta_scores.items()}
    })

    logging.info(f"Test Acc: {val_acc:.2f}%")
    logging.info(f"Test F-beta scores (Positive Class):")
    for k, v in fbeta_scores.items():
        logging.info(f"\tTest {k}: {v:.4f}")

    end = time.time()
    logging.info(f'Training complete: {(end-start):.2f}')
    ################################################################################

    ## save model ##################################################################
    model.save_model(os.path.join(log_dir, prefix+f"_{today_date}.json"))
    logging.info(f"Saved XGBoost model to {prefix+f'_{today_date}.json'}")
    ################################################################################

    ## save F Scores & metrics #####################################################
    save_f_beta_scores(model, X_test, Y_test, log_dir)
    ################################################################################

