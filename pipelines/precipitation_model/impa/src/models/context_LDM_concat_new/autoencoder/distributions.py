# -*- coding: utf-8 -*-
# flake8: noqa: E501

import numpy as np
import torch


def kl_from_standard_normal(mean, log_var):
    kl = 0.5 * (log_var.exp() + mean.square() - 1.0 - log_var)
    return kl.mean()


def sample_from_standard_normal(mean, log_var, num=None):
    std = (0.5 * log_var).exp()
    shape = mean.shape
    if num is not None:
        # expand channel 1 to create several samples
        shape = shape[:1] + (num,) + shape[1:]
        mean = mean[:, None, ...]
        std = std[:, None, ...]
    return mean + std * torch.randn(shape, device=mean.device)


def ensemble_nll_normal(ensemble, sample, epsilon=1e-5):
    mean = ensemble.mean(dim=1)
    var = ensemble.var(dim=1, unbiased=True) + epsilon
    logvar = var.log()

    diff = sample[:, None, ...] - mean
    logtwopi = np.log(2 * np.pi)
    nll = (logtwopi + logvar + diff.square() / var).mean()
    return nll


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean, device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape, device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor) for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )
