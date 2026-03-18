""" Applies the SpecTf cloud screening model to an EMIT scene.

Copyright 2025 California Institute of Technology
Apache License, Version 2.0

Author: Jake Lee, jake.h.lee@jpl.nasa.gov

This version of the dployment script quantizes the model, and runs it with PyTorch JiT
"""

import logging
import time

import rich_click as click
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from spectf_cloud.deploy.infra_setup import open_model_arch_spec
from spectf.model import SpecTfEncoder
from spectf.dataset import RasterDatasetTOA, ArrayDatasetTOA, ToaDataset
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
    default=None,
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
    help="""Produce a SpecTf transformer-generated cloud mask using PyTorch runtime.
    
    OUTFP is where the output file will be written (GeoTIFF .tif)
    RDNFP is the filepath of the radiance data (ENVI .img)
    OBSFP is the filepath of the observation data (ENVI .img)
    """
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
) -> np.ndarray:
    """Applies the SpecTf cloud screening model to an EMIT scene."""

    # Open model architecture specification from YAML file and Setup PyTorch device
    if device !=-1:
        device_specification = f"cuda:{device}"
    else:
        device_specification = "cpu"

    spec, device_ = open_model_arch_spec(arch_spec, device_specification=device_specification)

    # Initialize dataset
    dataset = RasterDatasetTOA(rdnfp, 
                               obsfp, 
                               irradiance, 
                               rm_bands=spec['spectra']['drop_band_ranges'],
                               transform=None, 
                               keep_bands=keep_bands, 
                               dtype=PRECISION, 
                               device=device_)

    # Initialize and run inference
    cloud_mask = run_pt_inference_model(dataset, spec, weights, device_)

    if outfp is not None:
        make_geotiff(cloud_mask, dataset.shape, outfp, proba, threshold)

    # reshape back to original image shape (rows, cols) or (cols, rows)
    cloud_mask_reshaped = cloud_mask.reshape(dataset.shape[0], dataset.shape[1])

    return cloud_mask_reshaped

def deploy_pt_from_toa(
    toa_dataset: np.ndarray,
    banddef: np.ndarray,
    weights: str,
    arch_spec: str,
    proba: bool = False,
    device: int = -1,
    threshold: float = 0.51,
    outfp: str | None = None,
) -> np.ndarray:
    """
    Applies the SpecTf cloud screening model to a top of atmosphere dataset.
    Args:
        toa_dataset: np.ndarray containing TOA reflectance data shape (rows, cols, bands) or (cols, rows, bands).
        banddef: np.ndarray containing band wavelengths corresponding to the bands in toa_dataset.
        proba: bool: Output probability map with the binary cloud mask, default False.
        weights: str: Filepath to latest trained model weights.
        arch_spec: str: Filepath to model architecture YAML specification.
               This file also needs to contain the bands to remove (if desired)
        device: int: Device specification for PyTorch (-1 for CPU, 0+ for GPU, MPS if available), default -1 (CPU).
        threshold: float: Threshold for cloud classification, default 0.51.
        outfp: str | None: Output filepath for the cloud mask GeoTIFF. if no path is provided - no file will be saved.

    Returns:
        cloud_mask: np.ndarray: The generated cloud mask (probability values).

    """
    # Open model architecture specification from YAML file and Setup PyTorch device
    if device != -1:
        device_specification = f"cuda:{device}"
    else:
        device_specification = "cpu"

    spec, device_ = open_model_arch_spec(
        arch_spec, device_specification=device_specification
    )

    # Initialize dataset
    dataset = ArrayDatasetTOA(
        toa_dataset,
        banddef,
        rm_bands=spec['spectra']['drop_band_ranges'],
        dtype=PRECISION,
        device=device_,
    )

    # Initialize and run inference
    cloud_mask = run_pt_inference_model(dataset, spec, weights, device_)

    # Save output GeoTIFF if specified
    if outfp is not None:
        make_geotiff(cloud_mask, dataset.shape, outfp, proba, threshold)

    # reshape back to original image shape (rows, cols) or (cols, rows)
    cloud_mask_reshaped = cloud_mask.reshape(dataset.shape[0], dataset.shape[1])
    return cloud_mask_reshaped


def initialize_pt_model(
    arch: dict, dataset: ToaDataset, weights: str, device_: torch.device
) -> nn.Module:
    """
    Initializes a PyTorch SpecTf model with the given architecture and weights.
    Args:
        arch: dict: Model architecture specifications.
        dataset: ToaDataset: Dataset object containing toa data.
        weights: str: Filepath to the model weights.
        device_: torch.device: Device to load the model onto.

    Returns:
        model: nn.Module: The initialized PyTorch model.

    """
    # Define and initialize the model
    banddef = torch.tensor(dataset.banddef, dtype=PRECISION, device=device_)
    model = SpecTfEncoder(
        banddef=banddef,
        dim_output=2,
        num_heads=arch["num_heads"],
        dim_proj=arch["dim_proj"],
        dim_ff=arch["dim_ff"],
        agg=arch["agg"],
        use_residual=False,
        num_layers=1,
    ).to(device_, dtype=PRECISION)
    state_dict = torch.load(weights, map_location=device_)
    model.load_state_dict(state_dict)
    model.eval()

    # Optimize for jit
    model = torch.jit.optimize_for_inference(torch.jit.script(model))
    return model

def run_pt_inference_model(dataset: ToaDataset, spec: dict, weights: str, device_: torch.device) -> np.ndarray:
    """
    Initializes a pytorch model and runs inference on the provided dataset using the specified model architecture and
    weights.
    Args:
        dataset: ToaDataset: Dataset object containing toa data.
        spec: dict: Model architecture specifications.
        weights: str: Filepath to the model weights.
        device_: torch.device: Device to run inference on.

    Returns:
        cloud_mask: np.ndarray: The generated cloud mask (probability values).

    """
    # Initialize dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=spec["inference"]["batch"],
        shuffle=False,
        num_workers=spec["inference"]["workers"],
    )

    # Initialize model
    model = initialize_pt_model(spec["architecture"], dataset, weights, device_)

    # Run Inference
    dataset_shape = (dataset.shape[0] * dataset.shape[1],)

    logging.info("Starting inference.")
    cloud_mask = np.zeros(dataset_shape).astype(np.float32)
    total_len = len(dataloader)
    with torch.inference_mode():
        curr = 0
        start = time.time()
        for i, batch in enumerate(dataloader):
            pred = model(batch)
            proba_ = nn.functional.softmax(pred, dim=1)
            proba_ = proba_.to(dtype=torch.float32).cpu().detach().numpy()[:, 1]

            nxt = curr + batch.size()[0]
            cloud_mask[curr:nxt] = proba_

            curr = nxt
            if i % 100 == 0:
                end = time.time()
                logging.info(
                    "Iter %d: %.2f min remain.",
                    i,
                    (((end - start) / 100) * (total_len - i - 1)) / 60,
                )
                start = time.time()

    logging.info("Inference complete.")

    # Return cloud mask probabilities
    return cloud_mask

if __name__ == "__main__":
    print(MAIN_CALL_ERR_MSG % "deploy-pt")