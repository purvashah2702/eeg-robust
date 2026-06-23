"""EEGNet — compact CNN for EEG via Braindecode."""
import torch
from braindecode.models import EEGNetv4


def build_eegnet(n_channels: int, n_classes: int, n_times: int) -> torch.nn.Module:
    """Build an EEGNet model matching your data's shape."""
    model = EEGNetv4(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
    )
    return model
