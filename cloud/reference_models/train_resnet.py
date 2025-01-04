
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from schedulefree import AdamWScheduleFree
import argparse
import h5py
import numpy as np
import yaml
from tqdm import tqdm
import logging
import datetime
import os
import time
import wandb
from ResNet import ResNet
from training_utils import utils
from typing import Optional
from sklearn.metrics import fbeta_score, log_loss, confusion_matrix

DEVICE = utils.get_device()

def save_f_beta_scores(model:torch.nn.Module, X:torch.Tensor, Y:np.ndarray, log_dir:str, prefix:str, beta_values=[0.1, 0.25, 0.5, 1, 2], chunk_logits=False):
    if Y.ndim == 2: 
        Y = np.argmax(Y, axis=1)

    model.eval()
    with torch.no_grad():
        if chunk_logits:
            logits_list = []
            chunk_size = 1_000
            for i in range(0, X.size(0), chunk_size):
                X_chunk = X[i:i + chunk_size].to(DEVICE)
                logits_chunk = model(X_chunk)
                logits_list.append(logits_chunk.cpu())
            logits = torch.cat(logits_list, dim=0)
        else:
            logits = model(X.to(DEVICE))

        probabilities = torch.softmax(logits, dim=1).cpu().detach().numpy()
        bce_loss = log_loss(Y, probabilities, labels=[0, 1])
        
        # Get the best F1 score threshold
        thresholds = np.around(np.arange(0, 1.01, 0.01), 2)
        f1s = []
        for t in thresholds:
            curr_pred = (probabilities[:, 1] >= t).astype(int)
            f1s.append(fbeta_score(Y, curr_pred, beta=1))
        test_best_f1 = thresholds[np.argmax(f1s)]

        f_beta_scores = {}
        for beta in beta_values:
            f_beta = fbeta_score(Y, probabilities[:, 1] > test_best_f1, beta=beta)
            f_beta_scores[f'F-{beta}'] = f_beta

    wandb.log({
        **{f"test_{k}": v for k, v in f_beta_scores.items()}
    })
    
    best_pred = (probabilities[:, 1] >= test_best_f1).astype(int)
    tn, fp, fn, tp = confusion_matrix(Y, best_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    f_beta_scores.update({'Log Loss':bce_loss})
    f_beta_scores.update({'TPR':tpr})
    f_beta_scores.update({'FPR':fpr})
    f_beta_scores.update({'Best F1 Thresh':test_best_f1})
    
    with h5py.File(os.path.join(log_dir, prefix+"_f_beta_score_tracking.hdf5"), 'w') as f:
        for k, v in f_beta_scores.items():
            dataset_name = f"{k}"
            f.create_dataset(dataset_name, data=[v], maxshape=(None,))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str, help='Reflectance data path', required=True)
    parser.add_argument('-model_yaml', type=str, help='Architecture for model file path', required=True)
    parser.add_argument('-train_split', type=str, help='The path to a text file containing the desired FID train split', required=True)
    parser.add_argument('-test_split', type=str, help='The path to a text file containing the desired FID test split', required=True)
    parser.add_argument('-log_dir', type=str, help='Directory to save log', required=False, default='.')
    parser.add_argument('-wandb_name', type=str, help='W&B run name', required=False)
    args = parser.parse_args()

    wandb.init()
    if args.wandb_name:
        custom_name = args.wandb_name
    else:
        custom_name = f"ResNet-{wandb.run.id}-{args.data_path.split('/')[-1].split('.')[0]}"
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
    prefix = wandb_name if wandb_name is not None else "ResNet"
    log_filename = os.path.join(log_dir, prefix+f"_{today_date}.log")
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
                        )   
    logging.info("Starting process...")
    ################################################################################

    with open(model_yaml, 'r') as f:
        model_meta_data = yaml.safe_load(f)
    
    ## find device #################################################################
    multiple_gpus = False if torch.cuda.device_count() < 2 else True
    logging.info(f'Using device: {str(DEVICE)} (num gpu: {torch.cuda.device_count()})')
    ################################################################################

    ## load in data and create dataloaders ########################################
    ## set the seeds
    utils.set_manual_seed()

    def cast_numpy(data):
        return torch.from_numpy(data).float().to(DEVICE)
    
    hps:dict = model_meta_data['training_hyperparameters']
    hdf5 = h5py.File(data_path, 'r')    
    spectra = hdf5['spectra'][:]
    labels = hdf5['labels'][:]
    all_fids = hdf5['fids'][:]

    ## Preprocess the data
    logging.info('Preprocessing data...')

    ## one-hot encode the labels if they aren't already
    num_classes = len(np.unique(labels))
    if labels.ndim == 1 or labels.shape[1] == 1:
        labels = np.eye(num_classes)[labels]

    start = time.time()

    train_indices, test_indices, train_fids, test_fids = utils.gen_train_test_split(fids=all_fids, 
                                                                              train_split=train_split, 
                                                                              test_split=test_split, 
                                                                              return_fids=True)

    X_test, Y_test = cast_numpy(spectra[test_indices]), cast_numpy(labels[test_indices])
    X_train, Y_train = cast_numpy(spectra[train_indices]), cast_numpy(labels[train_indices])

    logging.info('Training fids:')
    logging.info(list(train_fids))
    logging.info('Testing fids:')
    logging.info(list(test_fids))

    end = time.time()
    logging.info(f'Preprocessing complete: {(end-start):.2f}')
    
    batch_size = int(hps['bsz'])
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=True)
    ################################################################################


    ## create model ################################################################
    input_dim = spectra.shape[-1]

    model = ResNet.make_model(model_yaml, input_dim=input_dim, num_classes=num_classes, arch_subkeys=["architecture"])
    model = model.to(DEVICE)
    if multiple_gpus:
        model = nn.DataParallel(model)

    ## re-set the seeds. if one model samples more random numbers (more params) than another, then it could potentially go out of wack
    utils.set_manual_seed()
    ################################################################################


    ## create optimizer, scheduler, and loss fn ####################################
    criterion = torch.nn.BCEWithLogitsLoss().to(DEVICE)
    optimizer = AdamWScheduleFree(
            model.parameters(),
            lr=float(hps['optimizer']['lr']),
            betas=(float(hps['optimizer']['beta_1']), float(hps['optimizer']['beta_2'])),
    )
    ################################################################################


    ## training loop ###############################################################
    logging.info('Starting training...')
    start = time.time()
    for epoch in range(int(hps['epochs'])):
        model.train()
        running_loss = 0.0

        # Train loop
        for batch_X, batch_Y in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(batch_X)

            loss = criterion(outputs, batch_Y.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            y_true = torch.argmax(batch_Y, dim=1)
            correct = 100. * predicted.eq(y_true).sum().item() / batch_Y.size(0)
            wandb.log({"train_acc": correct, "train_loss": loss.item()})

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0.0
        total = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y.float())
                
                val_loss += loss.item()
                total += batch_Y.size(0)
                
                predicted = torch.argmax(outputs, dim=1)
                y_true = torch.argmax(batch_Y, dim=1)
                correct += predicted.eq(y_true).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_true.cpu().numpy())

        val_acc = 100. * correct / total
        fbeta_scores = utils.calculate_fbeta_scores(all_labels, all_preds)
        
        wandb.log({
            "test_acc": val_acc,
            "test_loss": val_loss / len(test_loader),
            **{f"test_{k}": v for k, v in fbeta_scores.items()}
        })
        
        logging.info(f"Epoch [{epoch+1}/{int(hps['epochs'])}]")
        logging.info(f"Train Loss: {running_loss/len(train_loader):.4f}")
        logging.info(f"Test Loss: {val_loss/len(test_loader):.4f}")
        logging.info(f"Test Acc: {val_acc:.2f}%")
        logging.info(f"Test F-beta scores (Positive Class):")
        for k, v in fbeta_scores.items():
            logging.info(f"\tTest {k}: {v:.4f}")

    end = time.time()
    logging.info(f'Training complete: {(end-start):.2f}')
    ################################################################################


    ## save model ##################################################################
    weights_path = os.path.join(log_dir, prefix+f"_{today_date}.pt")
    torch.save(model.state_dict(), weights_path)
    logging.info(f"Saved model weights to {weights_path}")
    ################################################################################

    ## save F Scores & metrics #####################################################
    save_f_beta_scores(model, X_test, Y_test.cpu().numpy(), log_dir, prefix, chunk_logits=False)
    ################################################################################
