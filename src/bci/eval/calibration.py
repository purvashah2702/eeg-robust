"""Calibration and uncertainty: temperature scaling, ECE, MC-dropout."""
import torch
import torch.nn as nn
import numpy as np


class TemperatureScaler(nn.Module):
    """
    Learns a single scalar temperature to calibrate a trained model's logits.
    Higher temperature -> softer (less overconfident) probabilities.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            lr: float = 0.01, max_iter: int = 100):
        """Fit temperature on a validation set by minimizing NLL loss."""
        logits = logits.detach()
        labels = labels.detach()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        loss_fn = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = loss_fn(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature.item()


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE: measures the gap between predicted confidence and actual accuracy.
    Lower is better (0 = perfectly calibrated).
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        bin_acc = accuracies[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        bin_weight = in_bin.sum() / len(confidences)
        ece += bin_weight * abs(bin_acc - bin_conf)

    return ece


def mc_dropout_predict(model, x: torch.Tensor, n_samples: int = 20):
    """
    Monte Carlo dropout: run the model n_samples times with dropout active,
    to estimate predictive uncertainty.
    Returns mean probabilities and their std (uncertainty) across samples.
    """
    model.train()  # keep dropout active
    all_probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs)
    model.eval()

    all_probs = torch.stack(all_probs, dim=0)  # (n_samples, batch, n_classes)
    mean_probs = all_probs.mean(dim=0)
    std_probs = all_probs.std(dim=0)

    return mean_probs, std_probs


def confidence_aware_reject(mean_probs: torch.Tensor, std_probs: torch.Tensor,
                              uncertainty_threshold: float = 0.15):
    """
    Reject (abstain from) predictions where uncertainty is too high.
    Returns predictions, with -1 for rejected/uncertain samples.
    """
    predictions = mean_probs.argmax(dim=1)
    max_class_std = std_probs.gather(1, predictions.unsqueeze(1)).squeeze(1)
    rejected = max_class_std > uncertainty_threshold
    predictions[rejected] = -1
    return predictions, rejected
