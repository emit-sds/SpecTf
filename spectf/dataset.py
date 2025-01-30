import torch
from torch.utils.data import Dataset

from spectf.toa import l1b_to_toa_arr
from spectf.utils import drop_bands

class RasterDatasetTOA(Dataset):
    """
    A PyTorch dataset class for pixelwise access of top-of-atmosphere (TOA) reflectance data derived from L1b rdn.

    Attributes:
    - shape (tuple): The shape of the L1b rdn raster.
    - toa_arr (ndarray): The top-of-atmosphere reflectance data, reshaped as a list of pixels.
    - banddef (np.array): The band wavelengths corresponding to the `toa_arr` indices.
    - metadata (dict): Metadata of the original raster image.
    - transform (callable, optional): Transformations or normalizations for each pixel spectra.

    Parameters:
    - rdnfp (str): File path to the radiance data (L1b product).
    - obsfp (str): File path to the observation data (L1b product).
    - irrfp (str): File path to the irradiance data.
    - transform (callable, optional): Optional transform to be applied to each spectral data point.

    The class relies on the `l1b_to_toa_arr` function to process the input data files and generate the TOA reflectance data.
    """

    def __init__(self, rdnfp, obsfp, irrfp, transform=None, keep_bands=False):
        super().__init__()

        self.toa_arr, self.banddef, self.metadata = l1b_to_toa_arr(rdnfp, obsfp, irrfp)
        self.shape = self.toa_arr.shape
        self.toa_arr = self.toa_arr.reshape((self.shape[0]*self.shape[1],self.shape[2]))
        if not keep_bands:
            self.toa_arr, self.banddef = drop_bands(self.toa_arr, self.banddef, nan=False)
        self.transform = transform

    def __len__(self):
        return len(self.toa_arr)

    def __getitem__(self, idx):
        out_spec = torch.tensor(self.toa_arr[idx], dtype=torch.float)
        if self.transform is not None:
            out_spec = self.transform(out_spec)
        
        return torch.unsqueeze(out_spec, -1)

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