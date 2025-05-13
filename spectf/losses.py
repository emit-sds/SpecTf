""" Losses for Evidential-SpecTf

Copyright 2025 California Institute of Technology
Apache License, Version 2.0

Author: William Keely, william.r.keely@jpl.nasa.gov
"""


import math

import torch
import torch.nn as nn


class EvidentialNLL(nn.Module):
    """
    Example from:
      https://arxiv.org/abs/2205.10060
    """

    def __init__(self, coeff: float = 0.01):
        super().__init__()
        self.coeff = coeff

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor):
        """
            logits: shape (b,4) interpreted as the logits [gamma, nu, alpha, beta]
            y_true: shape (b,1)
        """
        gamma = logits[:, 0:1]
        nu    = logits[:, 1:2]
        alpha = logits[:, 2:3]  
        beta  = logits[:, 3:4]

        error = gamma - y_true
        var   = beta / (nu + 1e-9)  

        #   log(var) + (1.0 + coeff*nu) * error^2 / var 

        loss = torch.log(var) + (1.0 + self.coeff * nu) * error.pow(2) / var
        return torch.mean(loss)
    

class EvidentialHuberNLL(nn.Module):

    def __init__(self, delta: float = 1.0, coeff: float = 0.01):
        super().__init__()
        self.delta = delta
        self.eps = 1e-9  # to avoid division by zero
        self.coeff = coeff

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor):
        """
            logits: shape (b,4) interpreted as [gamma, nu, alpha, beta]
            y_true: shape (b,1)
        """
        gamma = logits[:, 0:1]
        nu    = logits[:, 1:2]
        alpha = logits[:, 2:3]  
        beta  = logits[:, 3:4]

        error = gamma - y_true
        var   = beta / (nu + self.eps)  

        # Huber loss
        quadratic = torch.minimum(torch.abs(error), self.delta)
        linear = torch.abs(error) - quadratic
        huber_component = (0.5 * quadratic.pow(2) + self.delta * linear) / var
        loss = torch.log(var) + (1.0 + self.coeff * nu) * huber_component
        return torch.mean(loss)
    

class GaussianNLLLoss(nn.Module):
    def __init__(self, reduction: str | None = "mean", eps: float = 1e-6):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self._c = 0.5 * math.log(2.0 * math.pi)

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp(min=self.eps)
        nll = -( -torch.log(sigma) - self._c - 0.5 * (y - mu).pow(2) / sigma )
        if self.reduction == "mean":
            return nll.mean()
        if self.reduction == "batch":
            return nll.mean(dim=tuple(range(1, nll.ndim)))
        if self.reduction is None:
            return nll
        raise ValueError("reduction must be 'mean', 'batch', or None")


class HuberNLLLoss(nn.Module):
    def __init__(self, delta: float = 0.5, reduction: str | None = "mean", eps: float = 1e-6):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self.eps = eps
        self._c = 0.5 * math.log(2.0 * math.pi)

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        huber = torch.nn.functional.huber_loss(mu, y, reduction="none", delta=self.delta)
        sigma = sigma.clamp(min=self.eps)
        nll = -( -torch.log(sigma) - self._c - 0.5 * huber / sigma )
        if self.reduction == "mean":
            return nll.mean()
        if self.reduction == "batch":
            return nll.mean(dim=tuple(range(1, nll.ndim)))
        if self.reduction is None:
            return nll
        raise ValueError("reduction must be 'mean', 'batch', or None")