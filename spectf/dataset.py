""" Defines PyTorch dataset classes for loading raster or HDF5 data.

Copyright 2025 California Institute of Technology
Apache License, Version 2.0

Author: Jake Lee, jake.h.lee@jpl.nasa.gov
"""

import os
from collections.abc import Callable
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from spectf.toa import l1b_to_toa_arr
from spectf.utils import drop_bands, envi_header


class RasterDatasetTOA(Dataset):
    """A PyTorch dataset class for pixelwise access of top-of-atmosphere (TOA)
    reflectance data derived from L1b rdn.

    Attributes:
        shape (tuple): Shape of the L1b rdn raster.
        toa_arr (ndarray): TOA reflectance data reshaped as a list of pixels.
        banddef (np.array): Band wavelengths corresponding to `toa_arr` indices.
        metadata (dict): Metadata of the original raster image.
        transform (callable, optional): Transformations for each pixel spectra.

    Relies on the `l1b_to_toa_arr` function to process input data files and generate
    TOA reflectance data.
    """

    def __init__(
            self, 
            rdnfp: str, 
            obsfp: str, 
            irrfp: str,
            rm_bands: List[List[int]]=None,
            transform: Callable = None, 
            keep_bands: bool = False, 
            dtype: torch.dtype = torch.float,
            device: torch.device = None,
        ):
        """ Initialize the RasterDatasetTOA Dataset object.
        Args:
            rdnfp (str): File path to the radiance data (L1b product).
            obsfp (str): File path to the observation data (L1b product).
            irrfp (str): File path to the irradiance data.
            transform (callable): Optional transform to be applied to
                                  each spectral data point.
            keep_bands (bool): True to keep all bands on non-EMIT data.
        """
        super().__init__()

        # Raster files
        self.rdnhdr = envi_header(rdnfp)
        self.obshdr = envi_header(obsfp)

        assert os.path.exists(self.rdnhdr), f"Header file {self.rdnhdr} does not exist."
        assert os.path.exists(self.obshdr), f"Header file {self.obshdr} does not exist."
        assert os.path.exists(irrfp), f"Irradiance file {irrfp} does not exist."

        self.toa_arr, self.banddef, self.metadata = l1b_to_toa_arr(self.rdnhdr, self.obshdr, irrfp)
        self.shape = self.toa_arr.shape
        self.toa_arr = self.toa_arr.reshape((self.shape[0] * self.shape[1],
                                             self.shape[2]))
        if not keep_bands:
            self.toa_arr, self.banddef = drop_bands(self.toa_arr,
                                                    self.banddef, 
                                                    rm_bands,
                                                    nan=False)
        self.transform = transform

        self.toa_arr = torch.tensor(self.toa_arr, dtype=dtype)
        self.toa_arr = torch.unsqueeze(self.toa_arr, -1)
        if device is not None:
            self.toa_arr = self.toa_arr.to(device)

    def __len__(self):
        return len(self.toa_arr)

    def __getitem__(self, idx):
        out_spec = self.toa_arr[idx]
        if self.transform is not None:
            out_spec = self.transform(out_spec)

        return out_spec


class SpectraDataset(Dataset):
    """A PyTorch dataset class for access of ML-ready HDF5 spectral data.

    This dataset class is designed specifically for the dataset provided in:
    https://zenodo.org/records/14614218
    
    Attributes:
        spectra (ndarray): The spectral data.
        labels (ndarray): The corresponding labels for the spectral data.
        transform (callable): Transformations or normalizations 
                              for each spectral data point.
        device (str): The device to load the data onto (e.g., 'cpu', 'cuda:0').
    """

    def __init__(self, spectra: np.ndarray, labels: np.ndarray,
                 transform: bool = None, device: str = 'cpu'):
        """ Initialize the SpectraDataset object.
        
        Args:
            spectra (np.ndarray): The spectral data.
            labels (np.ndarray): The corresponding labels for the spectral data.
            transform (callable): Optional transform to be applied to
                                  each spectral data point. Default None.
            device (str): The device to load the data onto. Default 'cpu'.
        """
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
