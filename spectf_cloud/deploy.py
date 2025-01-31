import rich_click as click
import logging
import yaml
import time

import numpy as np
import spectral.io.envi as envi

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spectf.model import SpecTfEncoder
from spectf.dataset import RasterDatasetTOA
from spectf_cloud.cli import spectf_cloud

ENV_VAR_PREFIX = 'SPECTF_DEPLOY_'

# TODO: Refactor this into the CLI 
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[
        # Uncomment to also log to a file
        #logging.FileHandler(op.join('out.log')),
        logging.StreamHandler()
    ]
)

@click.argument(
    "rdnfp",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    envvar=f"{ENV_VAR_PREFIX}RDNFP",
)
@click.argument(
    "obsfp",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    envvar=f"{ENV_VAR_PREFIX}OBSFP",
)
@click.argument(
    "outfp",
    type=click.Path(),
    required=True,
    envvar=f"{ENV_VAR_PREFIX}OUTFP",
)
@click.option(
    "--keep-bands",
    is_flag=True,
    default=False,
    help="Keep all bands in the spectra (use for non-EMIT data).",
    envvar=f"{ENV_VAR_PREFIX}KEEP_BANDS",
)
@click.option(
    "--proba",
    is_flag=True,
    default=False,
    help="Output probability map instead of binary cloud mask.",
    envvar=f"{ENV_VAR_PREFIX}PROBA",
)
@click.option(
    "--weights",
    default="weights.pt",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
    help="Filepath to trained model weights.",
    envvar=f"{ENV_VAR_PREFIX}WEIGHTS",
)
@click.option(
    "--irradiance",
    default="irr.npy",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
    help="Filepath to irradiance numpy file.",
    envvar=f"{ENV_VAR_PREFIX}IRRADIANCE",
)
@click.option(
    "--arch-spec",
    default="arch.yml",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
    help="Filepath to model architecture YAML specification.",
    envvar=f"{ENV_VAR_PREFIX}ARCH_SPEC",
)
@click.option(
    "--device",
    default=-1,
    type=int,
    show_default=True,
    help="Device specification for PyTorch (-1 for CPU, 0+ for GPU, MPS if available).",
    envvar=f"{ENV_VAR_PREFIX}DEVICE",
)
@click.option(
    "--threshold",
    default=0.52,
    type=float,
    show_default=True,
    help="Threshold for cloud classification.",
    envvar=f"{ENV_VAR_PREFIX}THRESHOLD",
)
@spectf_cloud.command(
    add_help_option=True,
    help="Produce a SpecTf transformer-generated cloud mask."
)
def deploy(
    rdnfp,
    obsfp,
    outfp,
    keep_bands,
    proba,
    weights,
    irradiance,
    arch_spec,
    device,
    threshold
):
    print("Threads:", torch.get_num_threads())

    # Open model architecture specification from YAML file
    with open(arch_spec, 'r') as f:
        spec = yaml.safe_load(f)
    
    arch = spec['arch']
    inference = spec['inference']

    # Setup PyTorch device
    if torch.cuda.is_available() and device != -1:
        device_ = torch.device(f"cuda:{device}")
        logging.info(f"Device is cuda:{device}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device_ = torch.device("mps") # Apple silicon
        logging.info(f"Device is Apple MPS acceleration")
    else:
        device_ = torch.device("cpu")
        logging.info(f"Device is CPU")
    
    # Initialize dataset and dataloader
    dataset = RasterDatasetTOA(rdnfp, obsfp, irradiance, transform=None, keep_bands=keep_bands)
    dataloader = DataLoader(dataset, batch_size=inference['batch'], shuffle=False, num_workers=inference['workers'])

    # Define and initialize the model
    banddef = torch.tensor(dataset.banddef, dtype=torch.float, device=device_)
    model = SpecTfEncoder(banddef=banddef,
                          num_classes=2,
                          num_heads=arch['n_heads'],
                          dim_proj=arch['dim_proj'],
                          dim_ff=arch['dim_ff'],
                          dropout=0,
                          agg=arch['agg'],
                          use_residual=False,
                          num_layers=1).to(device_, dtype=torch.float)
    state_dict = torch.load(weights, map_location=device_)
    model.load_state_dict(state_dict)
    model.eval()

    # Inference

    logging.info("Starting inference.")
    if proba:
        cloud_mask = np.zeros((dataset.shape[0]*dataset.shape[1],)).astype(np.float32)
    else:
        cloud_mask = np.zeros((dataset.shape[0]*dataset.shape[1],)).astype(np.uint8)
    total_len = len(dataloader)
    with torch.inference_mode():
        curr = 0
        start = time.time()
        for i, batch in enumerate(dataloader):
            batch = batch.to(device_, dtype=torch.float)
            pred = model(batch)
            proba_ = nn.functional.softmax(pred, dim=1)
            proba_ = proba_.cpu().detach().numpy()[:,1]

            nxt = curr+batch.size()[0]
            if proba:
                cloud_mask[curr:nxt] = proba_
            else:
                cloud_mask[curr:nxt] = (proba_ >= threshold).astype(np.uint8)

            # Handle NODATA pixels by setting cloud probability to 0 if any band is below -1
            min_values, _ = torch.min(batch[:,:,0], dim=1)
            cloud_mask[curr:nxt] = np.where(min_values.cpu().detach().numpy() < -1, 0, cloud_mask[curr:nxt])

            curr = nxt
            if (i+1) % 100 == 0:
                end = time.time()
                logging.info(f"Iter {i}: {(((end-start)/100)*(total_len-i-1))/60:.2f} min remain.")
                start = time.time()

    logging.info("Inference complete.")
    
    # Reshape into input shape
    cloud_mask = cloud_mask.reshape(dataset.shape[0], dataset.shape[1], 1)

    # Prepare metadata for output
    metadata = dataset.metadata
    del metadata['wavelength']
    del metadata['wavelength units']
    metadata['description'] = 'SpecTf Cloud Mask'
    metadata['bands'] = 1
    if proba:
        metadata['data type'] = 4
        metadata['data_type'] = 4
        metadata['band names'] = ['Cloud Probability']
    else:
        metadata['data type'] = 1
        metadata['data_type'] = 1
        metadata['band names'] = ['Cloud Mask']

    # Save cloud mask
    if proba:
        envi.save_image(outfp, cloud_mask, dtype=np.float32, metadata=metadata, force=True)
    else:
        envi.save_image(outfp, cloud_mask, dtype=np.uint8, metadata=metadata, force=True)

    logging.info(f"Cloud mask saved to {outfp}")
