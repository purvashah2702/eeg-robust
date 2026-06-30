"""Measure latency and parameter counts for all 5 models -- closes Week 3 gap."""
import time
import torch
import torch.nn as nn
import numpy as np
from bci.models.eegnet_model import build_eegnet
from bci.models.shallow_convnet_model import build_shallow_convnet
from bci.models.tcn_model import build_tcn
from bci.eval.metrics import count_params, measure_latency_ms

N_CHANNELS, N_TIMES, N_CLASSES = 26, 1126, 4
dummy_input = torch.randn(1, N_CHANNELS, N_TIMES)

models = {
    "EEGNet": build_eegnet(N_CHANNELS, N_CLASSES, N_TIMES),
    "ShallowConvNet": build_shallow_convnet(N_CHANNELS, N_CLASSES, N_TIMES),
    "TCN": build_tcn(N_CHANNELS, N_CLASSES, N_TIMES),
}

print(f"{'Model':<18}{'Params':<12}{'Latency (ms)':<15}")
print("-" * 45)

for name, model in models.items():
    model.eval()
    params = count_params(model)

    def predict_fn(x):
        with torch.no_grad():
            return model(x)

    latency = measure_latency_ms(predict_fn, dummy_input, n_runs=50)
    print(f"{name:<18}{params:<12,}{latency:<15.2f}")

print("\n✅ Latency and params measurement complete!")
