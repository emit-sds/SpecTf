import xgboost as xgb
import rich_click as click
import h5py
import numpy as np
import yaml
import logging
import datetime
import os
import time
import wandb
from sklearn.metrics import fbeta_score, log_loss, confusion_matrix

from spectf_cloud.comparison_models import train_comparison, MAIN_CALL_ERR_MSG
from spectf_cloud.comparison_models.training_utils import utils

ENV_VAR_PREFIX = "XGBOOST_TRAIN_"

def save_f_beta_scores(model:xgb.XGBClassifier, X:np.ndarray, Y:np.ndarray, outdir:str, beta_values=[0.1, 0.25, 0.5, 1, 2]):
    if Y.ndim == 2: 
        Y = np.argmax(Y, axis=1)
        
    probabilities = model.predict_proba(X)[:, 1]
    bce_loss = log_loss(Y, probabilities, labels=[0, 1])
    
    # get the best F1 threshold
    thresholds = np.around(np.arange(0, 1.01, 0.01), 2)
    f1s = []
    for t in thresholds:
        curr_pred = (probabilities >= t).astype(np.int32)
        f1s.append(fbeta_score(Y, curr_pred, beta=1))
    test_best_f1 = thresholds[np.argmax(f1s)]

    f_beta_scores = {}
    for beta in beta_values:
        f_beta = fbeta_score(Y, probabilities > test_best_f1, beta=beta)
        f_beta_scores[f'F-{beta}'] = f_beta

    wandb.log({
        **{f"test_{k}": v for k, v in f_beta_scores.items()}
    })
    
    best_pred = (probabilities >= test_best_f1).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(Y, best_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    f_beta_scores.update({'Log Loss':bce_loss})
    f_beta_scores.update({'TPR':tpr})
    f_beta_scores.update({'FPR':fpr})
    f_beta_scores.update({'Best F1 Thresh':test_best_f1})
    
    with h5py.File(os.path.join(outdir, "xgb_f_beta_score_tracking.hdf5"), 'w') as f:
        for k, v in f_beta_scores.items():
            dataset_name = f"{k}"
            f.create_dataset(dataset_name, data=[v], maxshape=(None,))
    

@click.argument(
    "dataset",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    envvar=f'{ENV_VAR_PREFIX}DATASET'
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
    "--arch-yaml",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Filepath to the model's architecture yaml file. Needs to have 'architecture' key.",
    envvar=f'{ENV_VAR_PREFIX}ARCH_YAML'
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
@train_comparison.command(
    add_help_option=True,
    help="Train a XGBoost model."
)
def xgboost(
        dataset: list,
        train_csv: str,
        test_csv: str,
        arch_yaml: str,
        outdir: str,
        wandb_entity: str,
        wandb_project: str,
        wandb_name: str,
    ):

    ## create logging ##############################################################
    today_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    prefix = wandb_name if wandb_name is not None else "XGBoost"
    log_filename = os.path.join(outdir, prefix+f"_{today_date}.log")
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
                        )   
    logging.info("Starting process..")

    run = wandb.init(
        project = wandb_project,
        entity = wandb_entity,
        name = wandb_name,
        dir = './',
        config = {
            'dataset_path': dataset,
            'arch_file': arch_yaml,
        },
        settings=wandb.Settings(_service_wait=300)
    )
    if not wandb_name:
        run.name = f"XGBoost-{run.id}"
        run.save()
    ################################################################################

    with open(arch_yaml, 'r') as f:
        model_hyperparameters = yaml.safe_load(f)
        assert "architecture" in list(model_hyperparameters.keys()), "'architecture' key not found in yaml file."
    
    ## load in data and create train/test split ####################################
    for i, ds in enumerate(dataset):
        if i == 0:
            f = h5py.File(ds, 'r')
            labels = f['labels'][:]
            all_fids = f['fids'][:]
            spectra = f['spectra'][:]
        else:
            f = h5py.File(ds, 'r')
            labels = np.concatenate((labels, f['labels'][:]))
            all_fids = np.concatenate((all_fids, f['fids'][:]))
            spectra = np.concatenate((spectra, f['spectra'][:]))

    ## Preprocess the data
    logging.info('Preprocessing data...')
    start = time.time()

    train_indices, test_indices, train_fids, test_fids = utils.gen_train_test_split(fids=all_fids, 
                                                                              train_split=train_csv, 
                                                                              test_split=test_csv, 
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

    run.log({
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
    model.save_model(os.path.join(outdir, prefix+f"_{today_date}.json"))
    logging.info(f"Saved XGBoost model to {prefix+f'_{today_date}.json'}")
    ################################################################################

    ## save F Scores & metrics #####################################################
    save_f_beta_scores(model, X_test, Y_test, outdir)
    ################################################################################

if __name__ == "__main__":
    print(MAIN_CALL_ERR_MSG % "train-comparison xgboost")