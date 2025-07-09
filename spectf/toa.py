""" Functions for calculating top-of-atmosphere reflectance from Level 1b data.

This module will eventually be replaced by a different spectral util package.

Copyright 2025 California Institute of Technology
Apache License, Version 2.0

Author: Jake Lee, jake.h.lee@jpl.nasa.gov
"""

import numpy as np

from spectral.io import envi
from isofit.core.common import resample_spectrum

from spectf.utils import name_to_nm

def l1b_to_toa_arr(rdnfp: str, obsfp: str, irrfp: str):
    """
    Converts Level 1b radiance data to top-of-atmosphere (TOA) reflectance.

    Args:
        rdnfp (str): File path to the radiance data (L1b product .hdr).
        obsfp (str): File path to the observation data (L1b product .hdr).
        irrfp (str): File path to the irradiance data (.npy).

    Returns:
        toa_refl (np.ndarray): The calculated top-of-atmosphere reflectance.
        banddef (np.ndarray): Array of band center wavelengths from metadata.
        metadata (dict): Metadata from the radiance data header.
    """

    # open radiance
    rad_header = envi.open(rdnfp)
    rad = rad_header.open_memmap(interleave='bip')
    banddef = [name_to_nm(name) for name in rad_header.metadata['wavelength']]
    banddef = np.array(banddef, dtype=float)

    # To-sun zenith (0 to 90 degrees from zenith)
    obs = envi.open(obsfp).open_memmap(interleave='bip')
    zen = obs[:,:,4]
    zen = np.deg2rad(zen)

    # Earth-sun distance (AU)
    es_distance = obs[:,:,10]
    es_distance = np.average(es_distance)

    # calculate irr
    # wavelengths
    wl_arr = np.array(rad_header.metadata['wavelength'], dtype=float)
    # full width at half maximum
    fwhm_arr = np.array(rad_header.metadata['fwhm'], dtype=float)

    irr_arr = np.load(irrfp)
    irr_resamp = resample_spectrum(irr_arr[:,1], irr_arr[:,0], wl_arr, fwhm_arr)
    irr_resamp = np.array(irr_resamp, dtype=float)
    irr = irr_resamp / (es_distance ** 2)

    # Top of Atmosphere Reflectance
    toa_refl = (np.pi / np.cos(zen[:, :, np.newaxis])) * (rad / irr[np.newaxis, np.newaxis, :])
    
    # Handle NODATA values
    nodata_value = -9999.0
    toa_refl[rad == nodata_value] = np.nan

    return toa_refl, banddef, rad_header.metadata
