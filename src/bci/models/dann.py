"""DANN (Domain-Adversarial Neural Network) for cross-subject domain generalization."""
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Reverses gradient sign during backprop — the core trick behind DANN."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, alpha)


class DomainClassifier(nn.Module):
    """Small classifier that tries to predict which subject/domain a sample is from."""
    def __init__(self, feature_dim: int, n_domains: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_domains),
        )

    def forward(self, x):
        return self.net(x)


class DANNWrapper(nn.Module):
    """
    Wraps a feature extractor + label classifier with an adversarial domain
    classifier. The feature extractor is trained to make subjects'
    representations indistinguishable, while still being good at the task —
    encouraging features that generalize across subjects.
    """
    def __init__(self, feature_extractor: nn.Module, label_classifier: nn.Module,
                 feature_dim: int, n_domains: int, grl_alpha: float = 1.0):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.label_classifier = label_classifier
        self.domain_classifier = DomainClassifier(feature_dim, n_domains)
        self.grl_alpha = grl_alpha

    def forward(self, x, return_domain=True):
        features = self.feature_extractor(x)
        label_logits = self.label_classifier(features)

        if return_domain:
            reversed_features = grad_reverse(features, self.grl_alpha)
            domain_logits = self.domain_classifier(reversed_features)
            return label_logits, domain_logits

        return label_logits


def dann_loss(label_logits, label_targets, domain_logits, domain_targets,
              label_loss_fn=None, domain_weight: float = 0.5):
    """Combined DANN loss: task loss + weighted adversarial domain loss."""
    if label_loss_fn is None:
        label_loss_fn = nn.CrossEntropyLoss()
    domain_loss_fn = nn.CrossEntropyLoss()

    task_loss = label_loss_fn(label_logits, label_targets)
    domain_loss = domain_loss_fn(domain_logits, domain_targets)

    total_loss = task_loss + domain_weight * domain_loss
    return total_loss, task_loss, domain_loss
