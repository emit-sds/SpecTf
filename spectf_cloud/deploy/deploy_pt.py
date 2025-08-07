""" Applies the SpecTf cloud screening model to an EMIT scene.

Copyright 2025 California Institute of Technology
Apache License, Version 2.0

Author: Jake Lee, jake.h.lee@jpl.nasa.gov

This version of the dployment script quantizes the model, and runs it with PyTorch JiT
"""

import logging
import time

import yaml
import rich_click as click
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from spectf.model import SpecTfEncoder
from spectf.dataset import RasterDatasetTOA
from spectf_cloud.deploy.gen_geotiff import make_geotiff
from spectf_cloud.cli import spectf_cloud, MAIN_CALL_ERR_MSG, DEFAULT_DIR

PRECISION = torch.bfloat16
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
    help="Output probability map with the binary cloud mask.",
    envvar=f"{ENV_VAR_PREFIX}PROBA",
)
@click.option(
    "--weights",
    default=DEFAULT_DIR/"weights/current.pt",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
    help="Filepath to latest trained model weights.",
    envvar=f"{ENV_VAR_PREFIX}WEIGHTS",
)
@click.option(
    "--irradiance",
    default=DEFAULT_DIR/"irr.npy",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
    help="Filepath to irradiance numpy file.",
    envvar=f"{ENV_VAR_PREFIX}IRRADIANCE",
)
@click.option(
    "--arch-spec",
    default=DEFAULT_DIR/"spectf_cloud_config.yml",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
    help="Filepath to model architecture YAML specification. This file also needs to contain the bands to remove",
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
    default=0.51,
    type=float,
    show_default=True,
    help="Threshold for cloud classification.",
    envvar=f"{ENV_VAR_PREFIX}THRESHOLD",
)
@spectf_cloud.command(
    add_help_option=True,
    help="Produce a SpecTf transformer-generated cloud mask using PyTorch runtime."
)
def deploy_pt(
    rdnfp,
    obsfp,
    outfp,
    keep_bands,
    proba,
    weights,
    irradiance,
    arch_spec,
    device,
    threshold,
):
    """Applies the SpecTf cloud screening model to an EMIT scene."""

    # Open model architecture specification from YAML file
    with open(arch_spec, 'r', encoding='utf-8') as f:
        spec = yaml.safe_load(f)

    arch = spec['architecture']
    inference = spec['inference']

    # Setup PyTorch device
    if torch.cuda.is_available() and device != -1:
        device_ = torch.device(f"cuda:{device}")
        logging.info("Device is cuda:%s", device)
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device_ = torch.device("mps") # Apple silicon
        logging.info("Device is Apple MPS acceleration")
    else:
        device_ = torch.device("cpu")
        logging.info("Device is CPU")

    # Initialize dataset and dataloader
    dataset = RasterDatasetTOA(rdnfp, 
                               obsfp, 
                               irradiance, 
                               rm_bands=spec['spectra']['drop_band_ranges'],
                               transform=None, 
                               keep_bands=keep_bands, 
                               dtype=PRECISION, 
                               device=device_)
    dataloader = DataLoader(dataset,
                            batch_size=inference['batch'],
                            shuffle=False,
                            num_workers=inference['workers'])

    # Define and initialize the model
    banddef = torch.tensor(dataset.banddef, dtype=PRECISION, device=device_)
    model = SpecTfEncoder(banddef=banddef,
                          dim_output=2,
                          num_heads=arch['num_heads'],
                          dim_proj=arch['dim_proj'],
                          dim_ff=arch['dim_ff'],
                          agg=arch['agg'],
                          use_residual=False,
                          num_layers=1).to(device_, dtype=PRECISION)
    state_dict = torch.load(weights, map_location=device_)
    model.load_state_dict(state_dict)
    model.eval()

    # Optimize for jit
    model = torch.jit.optimize_for_inference(torch.jit.script(model))

    # Inference

    logging.info("Starting inference.")
    cloud_mask = np.zeros((dataset.shape[0]*dataset.shape[1],)).astype(np.float32)
    total_len = len(dataloader)
    with torch.inference_mode():
        curr = 0
        start = time.time()
        for i, batch in enumerate(dataloader):
            pred = model(batch)
            proba_ = nn.functional.softmax(pred, dim=1)
            proba_ = proba_.to(dtype=torch.float32).cpu().detach().numpy()[:,1]

            nxt = curr+batch.size()[0]
            cloud_mask[curr:nxt] = proba_

            curr = nxt
            if (i+1) % 100 == 0:
                end = time.time()
                logging.info("Iter %d: %.2f min remain.", i, (((end-start)/100)*(total_len-i-1))/60)
                start = time.time()

    logging.info("Inference complete.")

    make_geotiff(cloud_mask, dataset.shape, outfp, proba, threshold)

if __name__ == "__main__":
    print(MAIN_CALL_ERR_MSG % "deploy-pt")