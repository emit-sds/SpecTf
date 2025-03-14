from tensor_rt_model import SpecTfEncoderTensorRT
import torch
import yaml
import rich_click as click
from rich.traceback import install
import h5py
import os

# NOTE: THE BATCH SIZE, DEVICE, AND DTYPE ARE ALL SUPER IMPORTANT HERE AS THEY DICTATE HOW TensorRT WILL BUILD THE NETWORK
PRECISION = torch.bfloat16
DEVICE = torch.device("cuda")
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

def get_abs_fp(f:str)->str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f)

install()

@click.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=True)
@click.argument(
    "output_filepath",
    type=click.Path(),
    required=True,
)
@click.option(
    "--batch-size",
    default=2**12,
    type=int,
)
@click.option(
    "--weights",
    type=click.Path(exists=True, dir_okay=False),
    default=get_abs_fp("weights.pt"),
)
@click.option(
    "--arch-file",
    type=click.Path(exists=True, dir_okay=False),
    default=get_abs_fp("spectf_cloud_config.yml"),
)
def onnx(
    output_filepath:str,
    batch_size:int,
    weights:str,
    arch_file:str,
    ):

    with open(arch_file, 'r', encoding='utf-8') as f:
        spec = yaml.safe_load(f)

        arch = spec['arch']

        model = SpecTfEncoderTensorRT(
            dim_proj=arch['dim_proj'],
            dim_ff=arch['dim_ff'],
            agg=arch['agg'],
        ).to(DEVICE, dtype=PRECISION)

        model.load_state_dict(weights)
        model.eval()

        dummy_input = torch.randn((batch_size, 268), device=DEVICE).to(PRECISION)
        torch.onnx.export(model, dummy_input, output_filepath)

if __name__ == '__main__':
    onnx()