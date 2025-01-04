import numpy as np
import torch
import random
from sklearn.metrics import fbeta_score

BANDEFF_FILE = 'banddef.npy'
BANDEFF = np.load(BANDEFF_FILE)
BAD_BAND_WL=[381.0055927, 388.4092083, 395.8158144, 403.2254112, 1275.339, 1282.794, 1290.25, 1297.705, 1305.16, 1312.614, 1320.068, 2455.9944995, 2463.3816762, 2470.7679025, 2478.1531786, 2485.5385046, 2492.9238809]

def get_x_ticks():
    return range(len(BANDEFF))[::50], np.round(BANDEFF[::50])

def get_good_bands():
    banddef = BANDEFF
    dropbands = [np.argmin(np.abs(banddef - wl)) for wl in BAD_BAND_WL]
    return torch.tensor(np.delete(banddef, dropbands))

def get_good_bands_data(data):
    banddef = BANDEFF
    dropbands = [np.argmin(np.abs(banddef - wl)) for wl in BAD_BAND_WL]
    return torch.tensor(np.delete(data, dropbands, axis=-1))

def insert_bad_bands_data(modified_data):
    banddef = BANDEFF
    insert_indices = [np.argmin(np.abs(banddef - wl)) for wl in BAD_BAND_WL]
    full_data = np.full(len(banddef), np.nan)
    good_data_indices = [i for i in range(len(banddef)) if i not in insert_indices]
    full_data[..., good_data_indices] = modified_data
    return full_data

def flatten_geotiff_data(data):
    if len(data) == 3:
        return data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    return data

def get_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps") # Apple silicon
    else:
        return torch.device("cpu")

# I know the answer of the universe is 42, but https://arxiv.org/pdf/2109.08203
# "Are there black swans, i.e., seeds that produce radically different results? 
# Yes. On a scanning of 10^4 seeds, we obtained a difference between the maximum and minimum accuracy close to 2% 
# which is above the threshold commonly used by the computer vision community of what is considered significant."
def set_manual_seed(s:int=3407):
    random.seed(s)
    torch.manual_seed(s)

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
