import sys, os
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

class SpectraDataset(Dataset):
    """Pixelwise spectra dataset."""

    def __init__(self, spectra, labels, transform=None, device='cpu'):
        super().__init__()
        self.spectra = torch.tensor(spectra, dtype=torch.float32).to(device)

        self.labels = torch.tensor(labels).to(device)
        self.labels[self.labels==2] = 0 # shadow considered clear

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        out_spec = self.spectra[idx]
        if self.transform is not None:
            out_spec = self.transform(out_spec)
        
        return {
            'spectra': torch.unsqueeze(out_spec, -1),
            'label': self.labels[idx]
        }

class BandConcat(nn.Module):
    """Concatenate band wavelength to reflectance spectra."""

    def __init__(self, banddef):
        super().__init__()
        self.banddef = torch.unsqueeze(banddef, -1)
        self.banddef = torch.unsqueeze(self.banddef, 0)
        self.banddef = (self.banddef - 1440) / 600
        #self.banddef = (self.banddef - torch.mean(self.banddef)) / torch.std(self.banddef)

    def forward(self, spectra):
        """ 
            spectra: (b, s, 1)
            banddef: (s, 1)
        """
        encoded = torch.cat((spectra, self.banddef.expand_as(spectra)), dim=-1)
        return encoded

class SpectralEmbed(nn.Module):
    """Embed spectra and bands using Conv1D"""

    def __init__(self, n_filters: int = 128):
        super().__init__()
        self.linear = nn.Linear(2, n_filters)
    
    def forward(self, x):
        x = self.linear(x)
        x = torch.tanh(x)
        return x

class SimpleSeqClassifier(nn.Module):
    def __init__(self, 
                 banddef,
                 num_classes: int = 2,
                 num_heads: int = 8,
                 dim_proj: int = 32,
                 dim_ff: int = 128,
                 dropout: float = 0.1,
                 agg: str = 'max'):
        super().__init__()

        # Embedding
        self.band_concat = BandConcat(banddef)
        self.spectral_embed = SpectralEmbed(n_filters=dim_proj)

        # Attention
        self.self_attn = nn.MultiheadAttention(dim_proj, num_heads, dropout=dropout, bias=True, batch_first=True)

        # Feedforward
        self.linear1 = nn.Linear(dim_proj, dim_ff, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, dim_proj, bias=True)

        # Normalization
        self.norm1 = nn.LayerNorm(dim_proj, eps=1e-5, bias=True)
        self.norm2 = nn.LayerNorm(dim_proj, eps=1e-5, bias=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.gelu = torch.nn.functional.gelu

        # Classification
        self.aggregate = agg
        self.classifier = nn.Linear(dim_proj, num_classes)
        self.initialize_weights()


    def forward(self, x):
        x = self.band_concat(x)
        x = self.spectral_embed(x)

        # Transformer without skip connections
        x = self._sa_block(self.norm1(x))
        x = self._ff_block(self.norm2(x))

        if self.aggregate == 'mean':
            x = torch.mean(x, dim=1)
        elif self.aggregate == 'max':
            x,_ = torch.max(x, dim=1)
        x = self.classifier(x)
        return x

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _sa_block(self, x):
        x = self.self_attn(x, x, x)[0]
        return self.dropout1(x)
    
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.gelu(self.linear1(x))))
        return self.dropout2(x)

def drop_banddef(
        banddef,
        wls=[381.0055, 388.4092, 395.8158, 403.2254,
            1275.339, 1282.794, 1290.25, 1297.705, 1305.16, 1312.614, 1320.068,
            2455.994, 2463.381, 2470.767, 2478.153, 2485.538, 2492.923]):

    dropbands = []
    for wl in wls:
        deltas = np.abs(banddef - wl)
        dropbands.append(np.argmin(deltas))

    banddef = np.delete(banddef, dropbands, axis=0)

    return banddef

def drop_bands(
        spectra,
        banddef,
        wls=[381.0055, 388.4092, 395.8158, 403.2254,
            1275.339, 1282.794, 1290.25, 1297.705, 1305.16, 1312.614, 1320.068,
            2455.994, 2463.381, 2470.767, 2478.153, 2485.538, 2492.923],
        nan=True):
    dropbands = []
    for wl in wls:
        deltas = np.abs(banddef - wl)
        dropbands.append(np.argmin(deltas))

    if nan:
        if len(spectra.shape) == 2:
            spectra[:,dropbands] = np.nan
        else:
            spectra[dropbands] = np.nan
    else:
        if len(spectra.shape) == 2:
            spectra = np.delete(spectra, dropbands, axis=1)
        else:
            spectra = np.delete(spectra, dropbands, axis=0)
        banddef = np.delete(banddef, dropbands, axis=0)

    return spectra, banddef



# Importing the dataset
datasets = ['/scratch/cmml/emit-cloud/datasets/lbox_1e4_v6.hdf5',
            '/scratch/cmml/emit-cloud/datasets/mmgis_1e4_v6.hdf5']
for i, dataset in enumerate(datasets):
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

device = torch.device("cuda:0")

model = SimpleSeqClassifier(banddef = torch.tensor(bands).float().to(device),
                            num_classes=2,
                            num_heads=8,
                            dim_proj=64,
                            dim_ff=64,
                            dropout=0,
                            agg='max').to('cpu', dtype=torch.float)
model.load_state_dict(torch.load("/home/jakelee/emit-cloud/model_out/20240917_172836_825354_v6_1_1_m3_29.pt", map_location='cuda:0'))
model.to(device)
model.eval()

dataset = SpectraDataset(spectra, labels, device=device, transform=None)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)

attention = np.zeros_like(spectra)

for i, batch in enumerate(tqdm(dataloader)):
    x = batch['spectra'].to(device)
    y = batch['label'].to(device)

    with torch.no_grad():
        x = model.band_concat(x)
        x = model.spectral_embed(x)
        x = model.norm1(x)
        attn_out, attn = model.self_attn(x, x, x,need_weights=True)
    
        attn_matrix = attn.cpu().numpy().transpose(1, 2, 0)
        attn_sum = np.sum(attn_matrix, axis=0).T

        attention[1024 * i:1024 * (i+1), :] = attn_sum

np.save('20240917_attn.npy', attention)
np.save('20240917_lab.npy', labels)
np.save('20240917_mean_attn.npy', np.mean(attention, axis=0))

print(attention.shape)

print("Average for all")
mean_attn = np.mean(attention, axis=0)
print(mean_attn)
print("Band importance ranked")
print(sorted(zip(mean_attn, bands))[::-1])

print("Average for cloud")
cloud_attn = attention[labels==1]
mean_cloud_attn = np.mean(cloud_attn, axis=0)
print(mean_cloud_attn)
print("Band importance ranked")
print(sorted(zip(mean_cloud_attn, bands))[::-1])