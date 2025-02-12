import torch
from torch import nn
from torch import tensor 
import yaml
from typing import List
from spectf.utils import get_device

class RMSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(num_features))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.GELU(), dropout_rate=0.1, norm_type='batch'):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.norm1 = self._get_normalization_layer(norm_type, out_dim)
        
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.norm2 = self._get_normalization_layer(norm_type, out_dim)
        
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        
        # If the input dim and output dim don't alaign, need to project it into output dimension space 
        # If it's the same, then it's just the identity matrix
        self.project = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        if in_dim != out_dim:
            self.norm_project = self._get_normalization_layer(norm_type, out_dim)

    def _get_normalization_layer(self, norm_type, num_features):
        norm_type = norm_type.lower()
        if norm_type == 'batch':
            return nn.BatchNorm1d(num_features)
        elif norm_type == 'layer':
            return nn.LayerNorm(num_features)
        elif norm_type == 'rms':
            return RMSNorm(num_features)
        else:
            raise ValueError("Invalid normalization type. Choose 'batch', 'layer', or 'rms'.")
    
    def forward(self, x):
        identity = self.project(x)
        if hasattr(self, 'norm_project'):
            identity = self._apply_norm(self.norm_project, identity) # identity = self.norm_project(identity)
        
        out = self.linear1(x)
        out = self._apply_norm(self.norm1, out) # out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self._apply_norm(self.norm2, out) # out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out += identity
        out = self.activation(out)
        
        return out
    
    ######################## NEW ########################
    def _apply_norm(self, norm_layer, x):
        if isinstance(norm_layer, nn.BatchNorm1d):
            # If the input is 3D [batch_size, channels, length], BatchNorm1d should be applied along the channel dimension
            return norm_layer(x.transpose(1, 2)).transpose(1, 2)
        return norm_layer(x)

class ResNet(nn.Module):
    def __init__(self, in_dim:int, layer_dims:list, num_classes:int, dropout_rate=0.1, norm_type='batch'):
        super(ResNet, self).__init__()
        
        layer_dims = [in_dim]+layer_dims
        self.residual_blocks = nn.Sequential()
        for i in range(len(layer_dims)-1):
            self.residual_blocks.add_module(
                f"block_{i}", 
                ResidualBlock(layer_dims[i], layer_dims[i+1], dropout_rate=dropout_rate, norm_type=norm_type)
                )

        self.mlp_classification_head = nn.Linear(layer_dims[-1], num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    
    def forward(self, x:tensor):
        if len(x.shape) == 2:
            x = x.unsqueeze(1) # add in a dummy dim for the spectral dims if just passing in 2D batch
        x = self.residual_blocks(x)
        x = self.mlp_classification_head(x)
        x = x.squeeze(1)
        return x
    

def make_model(arch_yml:str, input_dim:int, num_classes:int, arch_subkeys:List[str]=[], weight_file:str=None):
    with open(arch_yml, 'r') as f:
        arch = yaml.safe_load(f)
        if arch_subkeys:
            for key in arch_subkeys:
                arch = arch[key]

    layers = []
    for key in arch['layers'].keys():
        layers += [arch['layers'][key]['dim'] for _ in range(arch['layers'][key]['count'])]
    dropout = arch['dropout']
    norm_type = arch['norm_type']

    model = ResNet(input_dim, layers, num_classes, dropout, norm_type)
    if weight_file:
        model.load_state_dict(torch.load(weight_file, map_location=get_device()))
        model.eval()
    return model
