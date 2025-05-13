import torch
import torch.nn as nn
from typing import List, Sequence

""" NLL MLP for ensembling. """
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_hidden: Sequence[int],
        activation: str = "LeakyReLU",
        weight_init: str = "kaiming",
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
        layers.append(nn.Linear(prev, 2 * output_dim))          # mean + log‑var
        self.net = nn.Sequential(*layers)

        self.output_dim = output_dim
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
        # ensure shape (batch, input_dim)
        if x.ndim > 2:
            x = x.view(x.size(0), -1)          # collapse all but batch dim
        out = self.net(x)
        mu, var = out[:,0].view(-1, 1), out[:, 1].view(-1, 1)
        
        return mu, var
