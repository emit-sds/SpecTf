"""  
MLP backbones for ensembling and DER.

Copyright 2025 California Institute of Technology
Apache License, Version 2.0

Author: William Keely, william.r.keely@jpl.nasa.gov

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, List


""" NLL MLP for ensembling. """
class GaussianMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_hidden: Sequence[int],
        activation: str = "LeakyReLU",
        weight_init: str = "kaiming",
        eps: float = 1e-6,                # numerical floor for variance
    ) -> None:
        super().__init__()

        try:
            act_cls = getattr(nn, activation)
        except AttributeError as e:
            raise ValueError(f"Unknown activation '{activation}'") from e

        layers: List[nn.Module] = []
        prev = input_dim
        for h in n_hidden:
            layers += [nn.Linear(prev, h), act_cls()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))          # mean + raw‑var
        self.net = nn.Sequential(*layers)

        self.output_dim = output_dim
        self.eps = eps
        self._init_weights(weight_init)

    def _init_weights(self, scheme: str) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if scheme == "kaiming":
                    nn.init.kaiming_normal_(m.weight)
                elif scheme == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif scheme == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                else:
                    raise ValueError(f"Unknown weight_init '{scheme}'")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # flatten everything except the batch dim
        if x.ndim > 2:
            x = x.view(x.size(0), -1)

        out = self.net(x)                              # (b, 2*output_dim)
        mu_raw, var_raw = out[:,0].view(-1, 1), out[:, 1].view(-1, 1)      # split along feature axis

        mu  = mu_raw
        var = F.softplus(var_raw) + self.eps           # ensure strictly positive

        return mu, var


""" DER MLP backbone. """ 
class EvidentialMLP(nn.Module):
    def __init__(
        self,
        n_hidden: Sequence[int],
        input_dim: int,
        output_dim: int = 4,
        activation: str = "LeakyReLU",
        weight_init: str = "kaiming",
        eps: float = 1e-6,                # numerical floor for variance
    ) -> None:
    
        super().__init__()

        try:
            act_cls = getattr(nn, activation)
        except AttributeError as e:
            raise ValueError(f"Unknown activation '{activation}'") from e

        layers: List[nn.Module] = []
        prev = input_dim
        for h in n_hidden:
            layers += [nn.Linear(prev, h), act_cls()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))          # mean + raw‑var
        self.net = nn.Sequential(*layers)

        self.output_dim = output_dim
        self.eps = eps
        self._init_weights(weight_init)

    def _init_weights(self, scheme: str) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if scheme == "kaiming":
                    nn.init.kaiming_normal_(m.weight)
                elif scheme == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif scheme == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                else:
                    raise ValueError(f"Unknown weight_init '{scheme}'")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        # (batch, 4)
        return self.net(x)