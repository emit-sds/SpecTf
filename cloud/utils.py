import sys
import os.path as op
from glob import glob
import logging
import re

import numpy as np

def verify_dir(dirpath):
    """ Verify that the directory path exists, throw if not """
    if not op.isdir(dirpath):
        logging.error(f"{dirpath} does not exist.")
        sys.exit(1)

def verify_file(filepath):
    """ Verify that the file path exists, throw if not """
    if not op.isfile(filepath):
        logging.error(f"{filepath} does not exist.")
        sys.exit(1)

def name_to_nm(bandname):
    """ Convert wavelength text to float """
    return float(re.search(r'(\d+\.\d+)', bandname).group(1))

def get_date(fid):
    """ Get the date string from full FID """
    return fid.split('_')[0].split('t')[1]

def get_mask_img(root, fid):
    """ Retrieve l2a mask img from FID """
    matches = glob(op.join(root, get_date(fid), fid.split('_')[0], 'l2a', '*_l2a_mask_*.img'))
    if len(matches) == 0:
        return None
    else:
        return matches[0]

def get_rdn_hdr(root, fid):
    """ Retrieve l1b rdn hdr from FID """
    matches = glob(op.join(root, get_date(fid), fid.split('_')[0], 'l1b', '*_l1b_rdn_*.hdr'))
    if len(matches) == 0:
        return None
    else:
        return matches[0]

def get_rdn_img(root, fid):
    """ Retrieve l1b rdn img from FID """
    matches = glob(op.join(root, get_date(fid), fid.split('_')[0], 'l1b', '*_l1b_rdn_*.img'))
    if len(matches) == 0:
        return None
    else:
        return matches[0]

def get_rdn_png(root, fid):
    """ Retrieve l1b rdn png from FID """
    matches = glob(op.join(root, get_date(fid), fid.split('_')[0], 'l1b', '*_l1b_rdn_*v01.png'))
    if len(matches) == 0:
        return None
    else:
        return matches[0]

def get_obs_hdr(root, fid):
    """ Retrieve l1b obs hdr from FID """
    matches = glob(op.join(root, get_date(fid), fid.split('_')[0], 'l1b', '*_l1b_obs_*.hdr'))
    if len(matches) == 0:
        return None
    else:
        return matches[0]

def get_loc_hdr(root, fid):
    """ Retrieve l1b loc hdr from FID """
    matches = glob(op.join(root, get_date(fid), fid.split('_')[0], 'l1b', '*_l1b_loc_*.hdr'))
    if len(matches) == 0:
        return None
    else:
        return matches[0]

def get_overlay_png(root, fid):
    """ Retrieve Labelbox cloud overlay png from FID"""
    pattern = f"{fid.split('_')[0]}_*_cloud_overlay.png"
    matches = sorted(glob(op.join(root, pattern)))
    if len(matches) == 0:
        return None
    else:
        return matches[0]

def get_label_img(root, fid):
    """ Retrieve label img from FID """
    matches = glob(op.join(root, f"{fid}_*_cloudlabel_*.img"))
    if len(matches) == 0:
        return None
    else:
        return matches[0]

def get_label_hdr(root, fid):
    """ Retrieve label img from FID """
    matches = glob(op.join(root, f"{fid}_*_cloudlabel_*.hdr"))
    if len(matches) == 0:
        return None
    else:
        return matches[0]

def drop_bands(
        spectra, 
        banddef,
        wls=[381.0055, 388.4092, 395.8158, 403.2254, 
            1275.339, 1282.794, 1290.25, 1297.705, 1305.16, 1312.614, 1320.068,
            2455.994, 2463.381, 2470.767, 2478.153, 2485.538, 2492.923],
        nan=True):
    """
    Removes bands/wavelengths of high uncertainty from a single spectra or an array of spectras.

    Parameters:
    - spectra (np.array): The input spectra or array of spectra. 
                          A single spectra is 1D, and an array of spectra is 2D.
    - banddef (np.array): The band definitions or wavelengths corresponding to the `spectra` 
                          indices. It is assumed to be 1D and of same length as the spectra 
                          or, in the case of 2D spectra, its columns.
    - wls (list, optional): The list of wavelengths to be removed. By default, it includes 
                            [1275.339, 1282.794, 1290.25, 1297.705, 1305.16, 1312.614, 1320.068]
                            as defined by isofit at:
                            https://github.com/isofit/isofit/blob/main/data/emit_osf_uncertainty.txt
    - nan (bool, optional): If set to True, the bands to be removed will be replaced with NaN 
                            values. If set to False, the bands will be physically removed from 
                            the spectra and band definitions. Default is True.

    Returns:
    - tuple: A tuple containing the modified spectra and the modified band definitions. 
             If `nan` is True, then the shape of the spectra remains unchanged, but with 
             NaN values in the dropped bands. If `nan` is False, the dropped bands are removed 
             from both the spectra and banddef arrays.

    Example:
    ```
    spectra = np.array([[1,2,3,4,5], [5,6,7,8,9]])
    banddef = np.array([1275, 1280, 1290, 1300, 1310])
    new_spectra, new_banddef = drop_bands(spectra, banddef, wls=[1290, 1300], nan=False)
    print(new_spectra) # [[1 2 5] [5 6 9]]
    print(new_banddef) # [1275 1280 1310]
    ```

    """

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

def drop_banddef(
        banddef,
        wls=[381.0055, 388.4092, 395.8158, 403.2254, 
            1275.339, 1282.794, 1290.25, 1297.705, 1305.16, 1312.614, 1320.068,
            2455.994, 2463.381, 2470.767, 2478.153, 2485.538, 2492.923]):
    """
    Update banddef to remove specified wavelengths

    Parameters:
    - banddef (np.array): The band definitions or wavelengths corresponding to the `spectra` 
                          indices. It is assumed to be 1D and of same length as the spectra 
                          or, in the case of 2D spectra, its columns.
    - wls (list, optional): The list of wavelengths to be removed. By default, it includes 
                            [1275.339, 1282.794, 1290.25, 1297.705, 1305.16, 1312.614, 1320.068]
                            as defined by isofit at:
                            https://github.com/isofit/isofit/blob/main/data/emit_osf_uncertainty.txt

    Returns:
    - numpy.arr: modified band definitions. 
    """

    dropbands = []
    for wl in wls:
        deltas = np.abs(banddef - wl)
        dropbands.append(np.argmin(deltas))
    
    banddef = np.delete(banddef, dropbands, axis=0)
    
    return banddef
