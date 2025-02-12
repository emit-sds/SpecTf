import numpy as np
import torch
from sklearn.metrics import fbeta_score

def gen_train_test_split(fids, train_split:str, test_split:str, return_fids=False):
    """The 'train_split' and 'test_split' parameters need to be file paths to a text file of fids"""

    def read_split_file(file_path):
        with open(file_path, 'r') as file:
            file_fids = [i.encode('utf-8') for i in set(file.read().replace(' ', '').split('\n')) if len(i)]
            return file_fids
        
    train_fids = read_split_file(train_split)
    test_fids = read_split_file(test_split)

    train_indices = np.isin(fids, train_fids)
    test_indices = np.isin(fids, test_fids)

    if return_fids:
        return train_indices, test_indices, train_fids, test_fids
    return train_indices, test_indices


def calculate_fbeta_scores(true_labels, predictions, beta_values=[0.1, 0.25, 0.5, 1, 2], average='binary'):
    fbeta_scores = {}
    for beta in beta_values:
        fbeta_scores[f'F{beta}'] = fbeta_score(true_labels, predictions, average=average, beta=beta)
    return fbeta_scores
