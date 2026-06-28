"""Simple CNN feature extractor for use with DANN (exposes intermediate features)."""
import torch
import torch.nn as nn


class EEGFeatureExtractor(nn.Module):
    """Temporal + spatial conv feature extractor, outputs a flat feature vector."""
    def __init__(self, n_channels: int, n_times: int, n_filters: int = 32):
        super().__init__()
        self.temporal_conv = nn.Conv2d(1, n_filters, kernel_size=(1, 25), padding=(0, 12))
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.spatial_conv = nn.Conv2d(n_filters, n_filters, kernel_size=(n_channels, 1))
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.pool = nn.AvgPool2d(kernel_size=(1, 50), stride=(1, 15))
        self.dropout = nn.Dropout(0.5)

        # Compute feature_dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self._forward_conv(dummy)
            self.feature_dim = out.shape[1]

    def _forward_conv(self, x):
        x = torch.relu(self.bn1(self.temporal_conv(x)))
        x = torch.relu(self.bn2(self.spatial_conv(x)))
        x = self.pool(x)
        x = self.dropout(x)
        return x.flatten(start_dim=1)

    def forward(self, x):
        # x: (batch, channels, time) -> add dim for conv2d: (batch, 1, channels, time)
        x = x.unsqueeze(1)
        return self._forward_conv(x)


def build_feature_extractor(n_channels: int, n_times: int):
    return EEGFeatureExtractor(n_channels, n_times)
