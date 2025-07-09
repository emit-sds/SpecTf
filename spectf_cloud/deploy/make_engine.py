import tensorrt as trt
import rich_click as click
from rich.traceback import install
import os

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

def get_abs_fp(f:str)->str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f)

install()

@click.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=True)
@click.argument(
    "output_filepath",
    type=click.Path(),
    required=True,
)
@click.option(
    "--onnx-fp",
    default=get_abs_fp("weights/current.onnx"),
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
def build_engine(
        output_filepath: str,
        onnx_fp: str,
    ):
    ######### Builder
    builder = trt.Builder(TRT_LOGGER)

    ########## Network
    ##### This is the network defintion
    network = builder.create_network()

    ########## Parser
    ##### network definition must be populated from the ONNX representation
    parser = trt.OnnxParser(network, TRT_LOGGER)
    success = parser.parse_from_file(onnx_fp)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    if not success:
        raise Exception("parser error")
    
    ########## Engine
    ##### The engine is a config on how to optimize the model
    # By default, the workspace is set to the total global memory size of the given device. 
    #   This is fine since we are only using 1 layer (pretty much). If the network has more layers, we'll need to explicity state the allowed mem size
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.BF16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    serialized_engine = builder.build_serialized_network(network, config)
    with open(output_filepath, "wb") as f:
        f.write(serialized_engine)

if __name__ == '__main__':
    build_engine()