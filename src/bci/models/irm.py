"""IRM (Invariant Risk Minimization) for domain generalization."""
import torch
import torch.nn as nn


def irm_penalty(logits: torch.Tensor, y: torch.Tensor, loss_fn=None) -> torch.Tensor:
    """
    Computes the IRM gradient penalty for one environment/domain's batch.
    Measures how much a dummy scale-1.0 classifier's gradient w.r.t. itself
    is non-zero -- a proxy for whether the learned features are causally
    predictive (invariant) vs spuriously correlated with this domain.
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    scale = torch.tensor(1.0, requires_grad=True, device=logits.device)
    loss = loss_fn(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    penalty = torch.sum(grad ** 2)
    return penalty


def irm_loss(domain_logits_list, domain_labels_list, loss_fn=None, penalty_weight: float = 1.0):
    """
    Combined IRM loss across multiple domains/environments.
    domain_logits_list: list of logit tensors, one per domain in the batch
    domain_labels_list: list of corresponding label tensors
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    total_task_loss = 0.0
    total_penalty = 0.0

    for logits, y in zip(domain_logits_list, domain_labels_list):
        total_task_loss = total_task_loss + loss_fn(logits, y)
        total_penalty = total_penalty + irm_penalty(logits, y, loss_fn)

    n_domains = len(domain_logits_list)
    avg_task_loss = total_task_loss / n_domains
    avg_penalty = total_penalty / n_domains

    total_loss = avg_task_loss + penalty_weight * avg_penalty
    return total_loss, avg_task_loss, avg_penalty
