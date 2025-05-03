""" Losses for Evidential-SpecTf

Copyright 2025 California Institute of Technology
Apache License, Version 2.0

Author: William Keely, william.r.keely@jpl.nasa.gov
"""



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

        # Huber loss replaces error**2
        huber = torch.nn.functional.huber_loss(gamma,
                y_true,
                reduction='none',
                delta=delta,
                weight=None
        )
        var   = beta / (nu + self.eps)  

        # Huber loss
        loss = torch.log(var) + (1.0 + self.coeff * nu) * huber / var
        return torch.mean(loss)
