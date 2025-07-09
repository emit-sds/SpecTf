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
from typing import List, Optional, Tuple

import numpy as np
import torch

def seed(i:int = 42) -> None:
    """ Seed all random number generators """
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)

def verify_dir(dirpath:str) -> None:
    """ Verify that the directory path exists, throw if not """
    if not op.isdir(dirpath):
        logging.error("%s does not exist.", dirpath)
        sys.exit(1)

def verify_file(filepath:str) -> None:
    """ Verify that the file path exists, throw if not """
    if not op.isfile(filepath):
        logging.error("%s does not exist.", filepath)
        sys.exit(1)

def name_to_nm(bandname:str) -> float:
    """ Convert wavelength text to float """
    return float(re.search(r'(\d+\.\d+)', bandname).group(1))

def get_date(fid:str) -> str:
    """ Get the date string from full FID """
    return fid.split('_')[0].split('t')[1]

def envi_header(inputpath):
    """
    https://github.com/emit-sds/emit-utils/blob/develop/emit_utils/file_checks.py
    Convert a envi binary/header path to a header, handling extensions
    Args:
        inputpath: path to envi binary file
    Returns:
        str: the header file associated with the input reference.

    """
    if op.splitext(inputpath)[-1] == '.img' or op.splitext(inputpath)[-1] == '.dat' or op.splitext(inputpath)[-1] == '.raw':
        # headers could be at either filename.img.hdr or filename.hdr.  Check both, return the one that exists if it
        # does, if not return the latter (new file creation presumed).
        hdrfile = op.splitext(inputpath)[0] + '.hdr'
        if op.isfile(hdrfile):
            return hdrfile
        elif op.isfile(inputpath + '.hdr'):
            return inputpath + '.hdr'
        return hdrfile
    elif op.splitext(inputpath)[-1] == '.hdr':
        return inputpath
    else:
        return inputpath + '.hdr'

def drop_bands(
        spectra: np.ndarray, 
        banddef: np.ndarray, 
        drop_wl_ranges: Optional[List[List[int]]] = None,
        nan: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes bands/wavelengths of high uncertainty from a single spectra
    or an array of spectras.

    By default, wavelengths <405nm and >2450nm are removed, as well as the
    high-uncertainty bands from 1275nm to 1320nm as defined by isofit.

    Args:
        spectra (np.ndarray):                       The input spectra (1D) or array of spectra (2D).
        banddef (np.ndarray):                       The original band wavelengths of the spectra.
        drop_wl_ranges (List[List[int]], optional): The list of ranges (inclusive) of wavelengths to be removed.
        nan (bool, optional):                       If True, replace removed bands with NaN.
                                                    Default True.

    Returns:
        np.ndarray: The modified spectra with bands removed (either w/ or w/o NaNs).
        np.ndarray: The modified banddef
    """
    mask = np.ones_like(banddef, dtype=bool)
    for low, high in drop_wl_ranges:
        mask ^= (banddef >= low) & (banddef <= high)
    if nan:
        spectra[..., ~mask] = np.nan
    else:
        spectra = np.delete(spectra, ~mask, axis=-1)
        banddef = banddef[mask]

    return spectra, banddef

def drop_banddef(
        banddef: np.ndarray, 
        drop_wl_ranges: Optional[List[List[int]]] = None,
    ) -> np.ndarray:
    """
    Removes bands/wavelengths of high uncertainty only from a band
    definition array.

    Args:
        banddef (np.ndarray):                       The original band definitions.
        drop_wl_ranges (List[List[int]], optional): The list of ranges (inclusive) of wavelengths to be removed.
    
    Returns:
        np.ndarray: The modified band definitions.
    """
    mask = np.ones_like(banddef, dtype=bool)
    for low, high in drop_wl_ranges:
        mask ^= (banddef >= low) & (banddef <= high)
    banddef = banddef[mask]

    return banddef

def get_device(gpu:Optional[int]=None) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}" if gpu else "cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps") # Apple silicon
    else:
        return torch.device("cpu")
