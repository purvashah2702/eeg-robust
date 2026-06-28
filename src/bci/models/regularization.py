"""Regularization techniques: mixup (1D), label smoothing."""
import torch
import torch.nn as nn
import numpy as np


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """
    Mixup for 1D/time-series data: blends pairs of trials and their labels.
    Returns mixed_x, y_a, y_b, lam — loss should be computed as:
        lam * loss(pred, y_a) + (1 - lam) * loss(pred, y_b)
    """
    batch_size = x.size(0)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def cutmix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """
    CutMix for 1D time-series: replaces a contiguous time chunk from one
    trial with the same chunk from another trial.
    """
    batch_size, n_channels, n_times = x.shape
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    cut_len = int(n_times * (1 - lam))
    if cut_len <= 0:
        return x, y, y, 1.0

    start = np.random.randint(0, n_times - cut_len + 1)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = x.clone()
    mixed_x[:, :, start:start + cut_len] = x[index, :, start:start + cut_len]

    actual_lam = 1 - (cut_len / n_times)
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, actual_lam


def mixup_loss(loss_fn, pred, y_a, y_b, lam):
    """Combine the two label losses weighted by lam."""
    return lam * loss_fn(pred, y_a) + (1 - lam) * loss_fn(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing built in."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, pred, target):
        return self.loss_fn(pred, target)
