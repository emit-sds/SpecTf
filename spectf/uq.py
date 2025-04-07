""" Evidential Head and utils for UQ from SpecTf

Copyright 2025 California Institute of Technology
Apache License, Version 2.0

Author: William Keely, william.r.keely@jpl.nasa.gov
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialRegressionHead(nn.Module):
    """
        Map the 4 logit channels [gamma, nu, alpha, beta] to valid ranges,
        returning shape (b,4). 
    """

    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
          X: shape (b,4)
        Returns:
          shape (b,4) with [gamma, nu, alpha, beta] in valid range.
        """
        gamma = X[:, 0:1]                          # any real
        nu = F.softplus(X[:, 1:2])                 # > 0
        alpha = F.softplus(X[:, 2:3]) + 1.0         # > 1
        beta = F.softplus(X[:, 3:4])               # > 0
        return torch.cat((gamma, nu, alpha, beta), dim=1)
    


def compute_aleatoric_uct(beta: torch.Tensor, alpha: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """
        sqrt( beta * (1+nu) / (alpha * nu) )
    """
    numerator = beta * (1.0 + nu)
    denominator = alpha * nu + 1e-9
    return torch.sqrt(torch.clamp(numerator / denominator, min=1e-9))


def compute_epistemic_uct(nu: torch.Tensor) -> torch.Tensor:
    """
        1 / sqrt(nu)
    """
    return 1.0 / torch.sqrt(torch.clamp(nu, min=1e-9))


def compute_evidential_predictions(logits: torch.Tensor) -> dict:
    """
        Output dict.
    """
    gamma = logits[:, 0:1]
    nu    = logits[:, 1:2]
    alpha = logits[:, 2:3]
    beta  = logits[:, 3:4]

    aleatoric_component = compute_aleatoric_uct(beta, alpha, nu)
    epistemic_component = compute_epistemic_uct(nu)
    total_uq = aleatoric_component + epistemic_component

    return {
        "pred"          : gamma,         # shape (b,1)
        "pred_uq"       : total_uq,      # shape (b,1)
        "aleatoric_component"  : aleatoric_component,
        "epistemic_component"  : epistemic_component,
        "logits"    : logits,            # shape (b,4)
    }