import logging
import yaml
import torch


def open_model_arch_spec(arch_spec: str, device_specification: str) -> tuple[dict, torch.device]:
    """
    Opens the model architecture specification from a YAML file.

    Args:
        arch_spec (str): Filepath to the model architecture YAML specification.
        device_specification: str: Device specification string (e.g., "cuda:0", "cpu").

    Returns:
        spec (dict): Dictionary containing the model architecture and inference specifications.
        device_ (torch.device): PyTorch device object for computation.
    """
    # Open model architecture specification from YAML file
    with open(arch_spec, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    # Setup PyTorch device
    if device_specification.startswith("cuda") and torch.cuda.is_available():
        device_ = torch.device(device_specification)
        logging.info(f"Device is {device_specification}")
    elif device_specification.startswith("cpu") or device_specification.startswith("mps"):
        # first try MPS if available
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device_ = torch.device("mps")  # Apple silicon
            logging.info("Device is Apple MPS acceleration")
        # otherwise use CPU
        else:
            device_ = torch.device("cpu")
            logging.info("Device is CPU")
    else:
        raise ValueError(f"Unsupported device specification: {device_specification}")

    return spec, device_
