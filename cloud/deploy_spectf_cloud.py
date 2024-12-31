import argparse
import logging
import yaml
import time

import numpy as np
import spectral.io.envi as envi

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import SimpleSeqClassifier
from utils import verify_file
from dataset import RasterDatasetTOA

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

if __name__ == "__main__":

    print("Threads:", torch.get_num_threads())

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Produce a transformer-generated cloud mask.")
    parser.add_argument('rdnfp', type=str, help='Filepath to L1b radiance .hdr file')
    parser.add_argument('obsfp', type=str, help='Filepath to L1b observation .hdr file')
    parser.add_argument('outfp', type=str, help='Filepath to output .hdr file')
    parser.add_argument('--keep-bands', action='store_true', help='Keep all bands in the spectra (use for non-EMIT data)')
    parser.add_argument('--proba', action='store_true', help='Output probability map instead of binary cloud mask')
    parser.add_argument('--weights', type=str, default='v6_m4.pt', help='Filepath to trained model weights (default: v6_m4.pt)')
    parser.add_argument('--irradiance', type=str, default='irr.npy', help='Filepath to irradiance numpy file (default: irr.npy)')
    parser.add_argument('--arch-spec', type=str, default='arch.yml', help='Filepath to model architecture YAML specification (default: arch.yml)')
    parser.add_argument('--device', type=int, default=-1, help='Device specification for PyTorch (-1 for CPU, 0+ for GPU, default: -1, MPS if available)')
    parser.add_argument('--threshold', type=float, default=0.52, help='Threshold for cloud classification (default: 0.52)')

    args = parser.parse_args()
    
    # Verify that files exist
    verify_file(args.rdnfp)
    verify_file(args.obsfp)
    verify_file(args.weights)
    verify_file(args.irradiance)
    verify_file(args.arch_spec)

    # Open model architecture specification from YAML file
    with open(args.arch_spec, 'r') as f:
        spec = yaml.safe_load(f)
    
    arch = spec['arch']
    inference = spec['inference']

    # Setup PyTorch device
    if torch.cuda.is_available() and args.device != -1:
        device = torch.device(f"cuda:{args.device}")
        logging.info(f"Device is cuda:{args.device}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps") # Apple silicon
        logging.info(f"Device is Apple MPS acceleration")
    else:
        device = torch.device("cpu")
        logging.info(f"Device is CPU")
    
    # Initialize dataset and dataloader
    dataset = RasterDatasetTOA(args.rdnfp, args.obsfp, args.irradiance, transform=None, keep_bands=args.keep_bands)
    dataloader = DataLoader(dataset, batch_size=inference['batch'], shuffle=False, num_workers=inference['workers'])

    # Define and initialize the model
    model = SimpleSeqClassifier(banddef = torch.tensor(dataset.banddef, dtype=torch.float, device=device),
                                num_classes=2,
                                num_heads=arch['n_heads'],
                                dim_proj=arch['dim_proj'],
                                dim_ff=arch['dim_ff'],
                                dropout=0,
                                agg=arch['agg']).to(device, dtype=torch.float)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Inference

    logging.info("Starting inference.")
    if args.proba:
        cloud_mask = np.zeros((dataset.shape[0]*dataset.shape[1],)).astype(np.float32)
    else:
        cloud_mask = np.zeros((dataset.shape[0]*dataset.shape[1],)).astype(np.uint8)
    total_len = len(dataloader)
    with torch.inference_mode():
        curr = 0
        start = time.time()
        for i, batch in enumerate(dataloader):
            batch = batch.to(device, dtype=torch.float)
            pred = model(batch)
            proba = nn.functional.softmax(pred, dim=1)
            proba = proba.cpu().detach().numpy()[:,1]

            nxt = curr+batch.size()[0]
            if args.proba:
                cloud_mask[curr:nxt] = proba
            else:
                cloud_mask[curr:nxt] = (proba >= args.threshold).astype(np.uint8)

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
    if args.proba:
        metadata['data type'] = 4
        metadata['data_type'] = 4
        metadata['band names'] = ['Cloud Probability']
    else:
        metadata['data type'] = 1
        metadata['data_type'] = 1
        metadata['band names'] = ['Cloud Mask']

    # Save cloud mask
    if args.proba:
        envi.save_image(args.outfp, cloud_mask, dtype=np.float32, metadata=metadata, force=True)
    else:
        envi.save_image(args.outfp, cloud_mask, dtype=np.uint8, metadata=metadata, force=True)

    logging.info(f"Cloud mask saved to {args.outfp}")
