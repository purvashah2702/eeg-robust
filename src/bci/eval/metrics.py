"""Evaluation metrics for BCI classifiers: accuracy, F1, AUROC, ITR, latency, params."""
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_itr(accuracy: float, n_classes: int, trial_duration_sec: float) -> float:
    """
    Information Transfer Rate (bits/min) — Wolpaw's formula.
    accuracy: classification accuracy (0-1)
    n_classes: number of possible classes
    trial_duration_sec: time taken per trial/decision
    """
    if accuracy <= 0 or accuracy >= 1:
        accuracy = np.clip(accuracy, 1e-6, 1 - 1e-6)

    bits_per_trial = np.log2(n_classes) + accuracy * np.log2(accuracy) + \
        (1 - accuracy) * np.log2((1 - accuracy) / (n_classes - 1) + 1e-12)
    bits_per_trial = max(bits_per_trial, 0)

    trials_per_min = 60.0 / trial_duration_sec
    return bits_per_trial * trials_per_min


def measure_latency_ms(predict_fn, X_sample, n_runs: int = 50) -> float:
    """Measure average single-sample inference latency in milliseconds."""
    predict_fn(X_sample[:1])
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_fn(X_sample[:1])
        times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times))


def count_params(model) -> int:
    """Count trainable parameters (works for PyTorch models)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_classifier(y_true, y_pred, y_proba=None, n_classes=4, trial_duration_sec=4.0):
    """Compute the standard metric suite given predictions."""
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }

    if y_proba is not None:
        try:
            results["auroc"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except ValueError:
            results["auroc"] = None

    results["itr_bits_per_min"] = compute_itr(
        results["accuracy"], n_classes, trial_duration_sec
    )

    return results
