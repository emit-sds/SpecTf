""" Utility functions for the SpecTf package

Copyright 2025 California Institute of Technology
Apache License, Version 2.0

Author: Jake Lee, jake.h.lee@jpl.nasa.gov
"""

import sys
import os.path as op
import logging
import re
import random

import numpy as np
import torch

def seed(i=42):
    """ Seed all random number generators """
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)

def verify_dir(dirpath):
    """ Verify that the directory path exists, throw if not """
    if not op.isdir(dirpath):
        logging.error("%s does not exist.", dirpath)
        sys.exit(1)

def verify_file(filepath):
    """ Verify that the file path exists, throw if not """
    if not op.isfile(filepath):
        logging.error("%s does not exist.", filepath)
        sys.exit(1)

def name_to_nm(bandname):
    """ Convert wavelength text to float """
    return float(re.search(r'(\d+\.\d+)', bandname).group(1))

def get_date(fid):
    """ Get the date string from full FID """
    return fid.split('_')[0].split('t')[1]

def drop_bands(spectra: np.ndarray, banddef: np.ndarray, wls: list = None,
               nan: bool = True):
    """
    Removes bands/wavelengths of high uncertainty from a single spectra
    or an array of spectras.

    By default, wavelengths <405nm and >2450nm are removed, as well as the
    high-uncertainty bands from 1275nm to 1320nm as defined by isofit.

    Args:
        spectra (np.ndarray): The input spectra (1D) or array of spectra (2D).
        banddef (np.ndarray): The original band wavelengths of the spectra.
        wls (list, optional): The list of wavelengths to be removed.
                              Default is defined for EMIT.
        nan (bool, optional): If True, replace removed bands with NaN.
                              Default True.

    Returns:
        np.ndarray: The modified spectra with bands removed.
        np.ndarray: The modified band definitions.
    """

    dropbands = []

    if wls is None:
        wls = [381.0055, 388.4092, 395.8158, 403.2254,
            1275.339, 1282.794, 1290.25, 1297.705, 1305.16, 1312.614, 1320.068,
            2455.994, 2463.381, 2470.767, 2478.153, 2485.538, 2492.923]

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

def drop_banddef(banddef: np.ndarray, wls: list = None):
    """Removes bands/wavelengths of high uncertainty only from a band
    definition array.

    By default, wavelengths <405nm and >2450nm are removed, as well as the
    high-uncertainty bands from 1275nm to 1320nm as defined by isofit.

    Args:
        banddef (np.ndarray): The original band definitions.
        wls (list, optional): The list of wavelengths to be removed.
                              Default is defined for EMIT.
    
    Returns:
        np.ndarray: The modified band definitions.
    """

    if wls is None:
        wls = [381.0055, 388.4092, 395.8158, 403.2254,
            1275.339, 1282.794, 1290.25, 1297.705, 1305.16, 1312.614, 1320.068,
            2455.994, 2463.381, 2470.767, 2478.153, 2485.538, 2492.923]

    dropbands = []
    for wl in wls:
        deltas = np.abs(banddef - wl)
        dropbands.append(np.argmin(deltas))

    banddef = np.delete(banddef, dropbands, axis=0)

    return banddef

def get_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps") # Apple silicon
    else:
        return torch.device("cpu")
