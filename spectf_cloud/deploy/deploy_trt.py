""" Applies the SpecTf cloud screening model to an EMIT scene.

Copyright 2025 California Institute of Technology
Apache License, Version 2.0

Author: Jake Lee, jake.h.lee@jpl.nasa.gov

This version of the dployment script quantizes the model, and runs it with Nvidia TensorRT
"""

import logging
import time

import yaml
import rich_click as click
import numpy as np
from osgeo import gdal
gdal.UseExceptions()

import torch
from torch.utils.data import DataLoader

from spectf.model import BandConcat
from spectf.dataset import RasterDatasetTOA
from spectf_cloud.deploy.gen_geotiff import make_geotiff
from spectf_cloud.cli import spectf_cloud, MAIN_CALL_ERR_MSG, DEFAULT_DIR

from spectf_cloud.deploy.tensor_rt_model import load_model_network_engine
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

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
    help="Output probability map instead of binary cloud mask.",
    envvar=f"{ENV_VAR_PREFIX}PROBA",
)
@click.option(
    "--engine",
    default=DEFAULT_DIR/"weights/current.engine",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
    help="Filepath to TensoRT model engine.",
    envvar=f"{ENV_VAR_PREFIX}ENGINE",
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
    "--threshold",
    default=0.51,
    type=float,
    show_default=True,
    help="Threshold for cloud classification.",
    envvar=f"{ENV_VAR_PREFIX}THRESHOLD",
)
@spectf_cloud.command(
    add_help_option=True,
    help="""Produce a SpecTf transformer-generated cloud mask using the TensorRT engine.
    
    OUTFP is where the output file will be written (GeoTIFF .tif)
    RDNFP is the filepath of the radiance data (ENVI .img)
    OBSFP is the filepath of the observation data (ENVI .img)
    """
)
def deploy_trt(
    rdnfp,
    obsfp,
    outfp,
    keep_bands,
    proba,
    engine,
    irradiance,
    arch_spec,
    threshold,
):
    """Applies the SpecTf cloud screening model to an EMIT scene."""
    if not torch.cuda.is_available():
        raise RuntimeError("Cannot run the TensorRT runt time engine without a CUDA supported GPU.")

    # Open model architecture specification from YAML file
    with open(arch_spec, 'r', encoding='utf-8') as f:
        spec = yaml.safe_load(f)
    inference = spec['inference']
    
    device_ = torch.device("cuda")

    # Initialize dataset and dataloader
    dataset = RasterDatasetTOA(rdnfp, 
                               obsfp, 
                               irradiance, 
                               rm_bands=spec['spectra']['drop_band_ranges'],
                               transform=None, 
                               keep_bands=keep_bands, 
                               dtype=PRECISION, 
                               device=device_)
    banddef = torch.tensor(dataset.banddef, dtype=PRECISION).to(device_)
    bc = BandConcat(banddef)
    dataset.toa_arr = bc(dataset.toa_arr)
    dataloader = DataLoader(dataset,
                            batch_size=inference['batch'],
                            shuffle=False,
                            num_workers=inference['workers'])

    # Define and initialize the model
    engine = load_model_network_engine(engine)
    context = engine.create_execution_context()

    ## Allocate buffers
    input_name = None
    expected_bsz = -1
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        size = trt.volume(engine.get_tensor_shape(tensor_name))
        
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            input_name = tensor_name
            expected_bsz = engine.get_tensor_shape(tensor_name)[0]
        else:
            host_ouput_buffer = cuda.pagelocked_empty(size, dtype=np.float16)
            device_output_buffer = cuda.mem_alloc(host_ouput_buffer.nbytes)

            context.set_tensor_address(tensor_name, int(device_output_buffer))
    stream = cuda.Stream()

    # Inference
    
    logging.info("Starting inference.")
    cloud_mask = np.zeros((dataset.shape[0]*dataset.shape[1],)).astype(np.float32)
    total_len = len(dataloader)
    with torch.inference_mode():
        curr = 0
        start = time.time()
        for i, batch in enumerate(dataloader):
            # If the batch size is smaller than needed (happens for last batch), we neeed to pad it
            original_pad_shape = -1
            if inference['batch'] != batch.size(0):
                original_pad_shape = batch.size(0)
                batch = pad_batch(batch, inference['batch'])
            if expected_bsz != -1:
                assert batch.size(0) == expected_bsz, f"Got unsupported batch size. Got: {batch.shape(0)} | Need: {expected_bsz}"

            # Create an input buffer
            batch = batch.contiguous() # should be of shape: (bsz, n dims, 2 - for the spectra and index)
            context.set_tensor_address(input_name, int(batch.data_ptr()))

            # Execute inference
            context.execute_async_v3(stream.handle)

            out_gpu = torch.empty((batch.shape[0], 2), dtype=PRECISION, device=device_)
            # Device->Device copy from device_output buffer into tensor
            cuda.memcpy_dtod_async(
                dest=out_gpu.data_ptr(),
                src=device_output_buffer,
                size=out_gpu.numel() * out_gpu.element_size(),
                stream=stream
            )
            stream.synchronize()

            # Perform softmax on the GPU - putting this here versus fusing with the trt network had no benefits
            proba_gpu = torch.nn.functional.softmax(out_gpu.float(), dim=1)

            # Bring the result back to CPU
            proba_ = proba_gpu.to(dtype=torch.float32).cpu().detach().numpy()[:,1]

            if original_pad_shape != -1:
                nxt = curr+original_pad_shape
                proba_ = proba_[:original_pad_shape]
            else:
                nxt = curr+batch.size(0)
            
            cloud_mask[curr:nxt] = proba_

            curr = nxt
            if (i+1) % 100 == 0:
                end = time.time()
                logging.info("Iter %d: %.2f min remain.", i, (((end-start)/100)*(total_len-i-1))/60)
                start = time.time()

    logging.info("Inference complete.")

    make_geotiff(cloud_mask, dataset.shape, outfp, proba, threshold)

def pad_batch(b: torch.Tensor, target_bsz:int):    
    # Pad w/ zeros
    padded_shape = (target_bsz,) + b.shape[1:]
    padded_batch = torch.zeros(
        padded_shape,
        dtype=b.dtype,
        device=b.device
    )

    padded_batch[:b.size(0)] = b
    return padded_batch


if __name__ == "__main__":
    print(MAIN_CALL_ERR_MSG % "deploy-trt")