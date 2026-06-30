"""1D Temporal Convolutional Network for EEG classification, with stochastic depth."""
import torch
import torch.nn as nn


class StochasticDepth(nn.Module):
    """Randomly drops the residual branch's contribution during training."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        # Bernoulli mask per-sample in the batch
        shape = [x.shape[0]] + [1] * (x.dim() - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = random_tensor.floor()
        return x.div(keep_prob) * binary_mask


class TemporalBlock(nn.Module):
    """One dilated causal conv block with residual connection + stochastic depth."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2, stoch_depth_prob=0.0):
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
        self.stoch_depth = StochasticDepth(stoch_depth_prob)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x))[:, :, :-self.padding] if self.padding else self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out))[:, :, :-self.padding] if self.padding else self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = self.stoch_depth(out)  # randomly zero this block's contribution during training
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """Stack of temporal blocks -> global average pool -> classifier."""
    def __init__(self, n_channels, n_classes, n_times,
                 hidden_channels=(32, 32, 64), kernel_size=7, dropout=0.2,
                 stoch_depth_probs=None):
        super().__init__()
        if stoch_depth_probs is None:
            # linearly increase stochastic depth probability with layer depth
            stoch_depth_probs = [0.1 * i / max(1, len(hidden_channels) - 1)
                                  for i in range(len(hidden_channels))]

        layers = []
        in_ch = n_channels
        for i, out_ch in enumerate(hidden_channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation,
                                         dropout, stoch_depth_probs[i]))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_ch, n_classes)

    def forward(self, x):
        out = self.tcn(x)
        out = self.pool(out).squeeze(-1)
        return self.classifier(out)


def build_tcn(n_channels: int, n_classes: int, n_times: int) -> torch.nn.Module:
    """Build a TCN model matching your data's shape, with stochastic depth enabled."""
    return TCN(n_channels, n_classes, n_times)
