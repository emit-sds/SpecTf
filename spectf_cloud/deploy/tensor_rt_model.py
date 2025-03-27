import torch
from torch import nn
from spectf.model import SpectralEmbed, EncoderLayer
from spectf_cloud.deploy import __SUPPORTS_TRT__

class SpecTfEncoderTensorRT(nn.Module):
    """This model just takes out the band concatenation step for the TensorRT runtime network"""
    def __init__(self,
                 dim_output: int = 2,
                 num_heads: int = 8,
                 dim_proj: int = 64,
                 dim_ff: int = 64,
                 dropout: float = 0.1,
                 agg: str = 'max',
                 use_residual: bool = False,
                 num_layers: int = 1):
        super().__init__()

        # Embedding
        self.spectral_embed = SpectralEmbed(n_filters=dim_proj)

        # Attention
        self.layers = nn.ModuleList([
            EncoderLayer(dim_proj, num_heads, dim_ff, dropout, use_residual)
            for _ in range(num_layers)
        ])

        # Head
        self.head = nn.Linear(dim_proj, dim_output)
        if agg == 'mean':
            self.aggregate = lambda x: torch.mean(x, dim=1)
        elif agg == 'max':
            self.aggregate = lambda x: torch.max(x, dim=1)[0]
        else:
            raise ValueError(f'Aggregation method {agg} is not implemented.')

        self.initialize_weights()

    def forward(self, x: torch.Tensor):
        x = self.spectral_embed(x)

        for layer in self.layers:
            x = layer(x)

        x = self.aggregate(x)
        x = self.head(x)

        return x

    def initialize_weights(self):
        """Initialize weights for the model."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            own_state[name].copy_(param)

if __SUPPORTS_TRT__:
    import tensorrt as trt

    def load_model_network_engine(enine_fp: str) -> trt.ICudaEngine:
        trt_logger = trt.Logger(trt.Logger.ERROR)
        with open(enine_fp, "rb") as f:
            runtime = trt.Runtime(trt_logger)
            engine = runtime.deserialize_cuda_engine(f.read())
            return engine