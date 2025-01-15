# import re
# import numpy as np

# from spectral.io import envi
# from isofit.core.common import resample_spectrum

# from utils import name_to_nm

def l1b_to_toa_arr(rdnfp, obsfp, irrfp): ...
#     """
#     Converts Level 1b radiance data to top-of-atmosphere (TOA) reflectance.

#     Parameters:
#     - rdnfp (str): File path to the radiance data (L1b product).
#     - obsfp (str): File path to the observation data (L1b product).
#     - irrfp (str): File path to the irradiance data.

#     Returns:
#     - toa_refl (numpy.ndarray): The calculated top-of-atmosphere reflectance.
#     - banddef (numpy.ndarray): Array of band definitions derived from the radiance data.
#     - metadata (dict): Metadata from the radiance data header.
#     """

#     # open radiance
#     rad_header = envi.open(rdnfp)
#     rad = rad_header.open_memmap(interleave='bip')
#     banddef = np.array([name_to_nm(name) for name in rad_header.metadata['wavelength']], dtype=float)

#     # To-sun zenith (0 to 90 degrees from zenith)
#     obs = envi.open(obsfp).open_memmap(interleave='bip')
#     zen = obs[:,:,4]
#     # zen = np.deg2rad(np.average(zen))
#     zen = np.deg2rad(zen)

#     # Earth-sun distance (AU)
#     es_distance = obs[:,:,10]
#     es_distance = np.average(es_distance)

#     # calculate irr
#     # wavelengths
#     wl_arr = np.array(rad_header.metadata['wavelength'], dtype=float)
#     # full width at half maximum
#     fwhm_arr = np.array(rad_header.metadata['fwhm'], dtype=float)

#     irr_arr = np.load(irrfp)
#     irr_resamp = resample_spectrum(irr_arr[:,1], irr_arr[:,0], wl_arr, fwhm_arr)
#     irr_resamp = np.array(irr_resamp, dtype=float)
#     irr = irr_resamp / (es_distance ** 2)

#     # Top of Atmosphere Reflectance
#     toa_refl = (np.pi / np.cos(zen[:, :, np.newaxis])) * (rad / irr[np.newaxis, np.newaxis, :])

#     return toa_refl, banddef, rad_header.metadata