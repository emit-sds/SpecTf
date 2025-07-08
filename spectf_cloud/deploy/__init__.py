__all__ = [
    'deploy_pt',
]
from spectf_cloud.deploy import deploy_pt

# Only load deploy_trt if TensorRT is available
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    from spectf_cloud.deploy import deploy_trt
    __all__.append('deploy_trt')
except (ModuleNotFoundError, ImportError):
    pass