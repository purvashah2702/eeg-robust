"""ShallowConvNet — simple CNN baseline for EEG via Braindecode."""
import torch
from braindecode.models import ShallowFBCSPNet


def build_shallow_convnet(n_channels: int, n_classes: int, n_times: int) -> torch.nn.Module:
    """Build a ShallowConvNet model matching your data's shape."""
    model = ShallowFBCSPNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        final_conv_length="auto",
        add_log_softmax=False,
    )
    return model
