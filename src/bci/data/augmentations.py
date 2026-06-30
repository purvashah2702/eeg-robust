"""EEG-specific data augmentations for training robustness."""
import torch
import numpy as np


def time_jitter(x: torch.Tensor, max_shift: int = 10) -> torch.Tensor:
    """Randomly shift the signal along the time axis (circular roll)."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    return torch.roll(x, shifts=shift, dims=-1)


def gaussian_noise(x: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    """Add small Gaussian noise to simulate sensor/measurement noise."""
    return x + torch.randn_like(x) * std


def amplitude_scale(x: torch.Tensor, scale_range=(0.9, 1.1)) -> torch.Tensor:
    """Randomly scale amplitude — simulates electrode contact variability."""
    scale = np.random.uniform(*scale_range)
    return x * scale


def channel_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
    """Randomly zero out some channels — simulates bad/disconnected electrodes."""
    n_channels = x.shape[0]
    mask = (torch.rand(n_channels) > p).float().unsqueeze(-1)
    return x * mask


def spec_augment_time_mask(x: torch.Tensor, max_mask_len: int = 50) -> torch.Tensor:
    """Randomly mask out a contiguous chunk of time (SpecAugment-style)."""
    x = x.clone()
    n_times = x.shape[-1]
    mask_len = np.random.randint(0, max_mask_len)
    if mask_len == 0:
        return x
    start = np.random.randint(0, max(1, n_times - mask_len))
    x[:, start:start + mask_len] = 0
    return x


class EEGAugmentor:
    """Applies a random combination of augmentations to a single trial."""
    def __init__(self, noise_std=0.05, max_shift=10, channel_drop_p=0.1,
                 max_mask_len=50, scale_range=(0.9, 1.1), apply_prob=0.5):
        self.noise_std = noise_std
        self.max_shift = max_shift
        self.channel_drop_p = channel_drop_p
        self.max_mask_len = max_mask_len
        self.scale_range = scale_range
        self.apply_prob = apply_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.rand() < self.apply_prob:
            x = time_jitter(x, self.max_shift)
        if np.random.rand() < self.apply_prob:
            x = gaussian_noise(x, self.noise_std)
        if np.random.rand() < self.apply_prob:
            x = amplitude_scale(x, self.scale_range)
        if np.random.rand() < self.apply_prob:
            x = channel_dropout(x, self.channel_drop_p)
        if np.random.rand() < self.apply_prob:
            x = spec_augment_time_mask(x, self.max_mask_len)
        if np.random.rand() < self.apply_prob:
            x = channel_shuffle(x, p=1.0)
        return x


def channel_shuffle(x: torch.Tensor, p: float = 0.3) -> torch.Tensor:
    """Randomly permute a subset of channels — simulates electrode cap misplacement/swap."""
    n_channels = x.shape[0]
    if np.random.rand() > p:
        return x
    x = x.clone()
    perm = torch.randperm(n_channels)
    return x[perm]
