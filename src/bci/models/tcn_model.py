"""1D Temporal Convolutional Network for EEG classification."""
import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """One dilated causal conv block with residual connection."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.padding = padding

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x))[:, :, :-self.padding] if self.padding else self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out))[:, :, :-self.padding] if self.padding else self.bn2(self.conv2(out)))
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """Stack of temporal blocks -> global average pool -> classifier."""
    def __init__(self, n_channels, n_classes, n_times,
                 hidden_channels=(32, 32, 64), kernel_size=7, dropout=0.2):
        super().__init__()
        layers = []
        in_ch = n_channels
        for i, out_ch in enumerate(hidden_channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_ch, n_classes)

    def forward(self, x):
        # x: (batch, channels, time)
        out = self.tcn(x)
        out = self.pool(out).squeeze(-1)
        return self.classifier(out)


def build_tcn(n_channels: int, n_classes: int, n_times: int) -> torch.nn.Module:
    """Build a TCN model matching your data's shape."""
    return TCN(n_channels, n_classes, n_times)
