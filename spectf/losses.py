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
      Eq. 12 implementation
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
        _alpha = logits[:, 2:3] # nu + 1, but not used
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
        _alpha = logits[:, 2:3]  
        beta  = logits[:, 3:4]

        huber = torch.nn.functional.huber_loss(gamma,
                y_true,
                reduction='none',
                delta=self.delta,
                weight=None
        )
        var   = beta / (nu + self.eps)  

        # Huber loss
        loss = torch.log(var) + (1.0 + self.coeff * nu) * huber / var
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
    
class EvidentialTweedieNLL(nn.Module):

    def __init__(self, p: float = 1.5, coeff: float = 0.01):
        super().__init__()
        self.p = p  # Tweedie power parameter (guaranteed 1 < p < 2)
        self.eps = 1e-9  # to avoid division by zero
        self.coeff = coeff

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor):
        """
            logits: shape (b,4) interpreted as [gamma, nu, alpha, beta]
            y_true: shape (b,1)
        """
        gamma = logits[:, 0:1]
        nu    = logits[:, 1:2]
        _alpha = logits[:, 2:3]  
        beta  = logits[:, 3:4]

        # Tweedie loss: -y * (mu^(1-p) / (1-p)) + (mu^(2-p) / (2-p))
        term1 = -y_true * (torch.pow(gamma, 1 - self.p) / (1 - self.p))
        term2 = torch.pow(gamma, 2 - self.p) / (2 - self.p)
        tweedie = term1 + term2

        var = beta / (nu + self.eps)  

        # Tweedie loss with evidential weighting
        loss = torch.log(var) + (1.0 + self.coeff * nu) * tweedie / var
        return torch.mean(loss)





""" Regularized DER loss. """
def NormalInverseGamma_Reg(y, gamma, v, alpha, beta, reduce=True):
    error = torch.abs(y-gamma)
    evidential = 2.0 * v * (alpha)
    reg = error * evidential
    return reg.mean() if reduce else reg

def NormalInverseGamma_NLL(y, gamma, v, alpha, beta, reduce=True):
    two_blambda = 2.0 * beta * (1.0 + v)                    
    nll = 0.5*torch.log(torch.pi/v)  \
        - alpha*torch.log(two_blambda)  \
        + (alpha+0.5) * torch.log(v*(y-gamma)**2 + two_blambda)  \
        + torch.lgamma(alpha)  \
        - torch.lgamma(alpha+0.5)
    return nll.mean() if reduce else nll

class DER_Loss(nn.Module):
    def __init__(self, coeff: float = 1.0, 
                 reduce: bool = True, ):
        super().__init__()
        self.coeff = coeff
        self.reduce = reduce

    def forward(self, y, gamma, v, alpha, beta):
        loss_nll = NormalInverseGamma_NLL(y, gamma, v, alpha, beta)
        loss_reg = NormalInverseGamma_Reg(y, gamma, v, alpha, beta, self.reduce,)
        return loss_nll + self.coeff * loss_reg
