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





""" Regularized KL DER loss. """
class NormalInverseGamma_NLL(nn.Module):
    def __init__(self, reduce: bool = True, eps: float = 1e-6):
        super().__init__()
        self.reduce = reduce
        self.eps = eps

    def forward(self, y, gamma, v, alpha, beta):
        v = v.clamp_min(self.eps)
        alpha = alpha.clamp_min(self.eps)
        beta = beta.clamp_min(self.eps)

        two_blambda = 2.0 * beta * (1.0 + v)                    
        nll = (0.5 * torch.log(math.pi / v)
               - alpha * torch.log(two_blambda)
               + (alpha + 0.5) * torch.log(v * (y - gamma).pow(2) + two_blambda)
               + torch.lgamma(alpha)
               - torch.lgamma(alpha + 0.5))
        return nll.mean() if self.reduce else nll


class NormalInverseGamma_KL(nn.Module):
    def forward(self, mu1, v1, a1, b1, mu2, v2, a2, b2):
        term1 = 0.5 * (a1 - 1) / b1 * (v2 * (mu2 - mu1).pow(2))
        term2 = 0.5 * v2 / v1
        term3 = -0.5 * torch.log(torch.abs(v2) / torch.abs(v1))
        term4 = -0.5
        term5 = a2 * torch.log(b1 / b2)
        term6 = -(torch.lgamma(a1) - torch.lgamma(a2))
        term7 = (a1 - a2) * torch.digamma(a1)
        term8 = -(b1 - b2) * a1 / b1
        return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8


class NormalInverseGamma_Reg(nn.Module):
    def __init__(self, omega: float = 0.01, reduce: bool = True, kl: bool = False):
        super().__init__()
        self.omega = omega
        self.reduce = reduce
        self.kl_flag = kl
        self.kl_module = NormalInverseGamma_KL()

    def forward(self, y, gamma, v, alpha, beta):
        error = torch.abs(y - gamma)

        if self.kl_flag:
            kl = self.kl_module(
                gamma, v, alpha, beta,
                gamma, self.omega, 1.0 + self.omega, beta
            )
            reg = error * kl
        else:
            evidential = 2.0 * v + alpha
            reg = error * evidential

        return reg.mean() if self.reduce else reg


class DERLossRegularized(nn.Module):
    def __init__(self, coeff: float = 1.0, omega: float = 0.01,
                 reduce: bool = True, kl: bool = False):
        super().__init__()
        self.nll = NormalInverseGamma_NLL(reduce=reduce)
        self.reg = NormalInverseGamma_Reg(omega=omega, reduce=reduce, kl=kl)
        self.coeff = coeff

    def forward(self, y, gamma, v, alpha, beta):
        loss_nll = self.nll(y, gamma, v, alpha, beta)
        loss_reg = self.reg(y, gamma, v, alpha, beta)
        return loss_nll + self.coeff * loss_reg
