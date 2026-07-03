from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_acc_loss(
    *,
    y_true: torch.Tensor,
    class_probs: torch.Tensor,
    n_classes: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Differentiable accuracy-style loss.

    We maximize the expected probability assigned to the true class:
        soft_acc = mean_i P(y_i = true_i)
    and minimize:
        loss_acc = 1 - soft_acc

    Args:
        y_true: (batch,) integer labels in {1..K}
        class_probs: (batch, K) probabilities
        n_classes: K
        eps: numerical stability epsilon
    Returns:
        loss: scalar tensor in [0, 1]
    """
    if class_probs.ndim != 2 or int(class_probs.shape[1]) != int(n_classes):
        raise ValueError("class_probs must have shape (batch, K)")

    y_true = y_true.to(torch.int64)
    y_onehot = F.one_hot(y_true - 1, num_classes=int(n_classes)).to(dtype=class_probs.dtype)

    P = class_probs.clamp_min(0.0)
    P = P / P.sum(dim=1, keepdim=True).clamp_min(eps)

    p_true = (P * y_onehot).sum(dim=1)
    return 1.0 - p_true.mean()


def soft_qwk_loss(
    *,
    y_true: torch.Tensor,
    class_probs: torch.Tensor,
    n_classes: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Differentiable quadratic weighted kappa (QWK) loss.

    We use a "soft" confusion matrix O = Y^T P, where:
      - Y is one-hot true labels
      - P is predicted class probabilities

    QWK = 1 - (sum(W * O) / sum(W * E)), where E is the expected matrix from marginals.
    We minimize (sum(W * O) / sum(W * E)), which is equivalent to minimizing (1 - QWK).

    Args:
        y_true: (batch,) integer labels in {1..K}
        class_probs: (batch, K) probabilities (will be clamped/renormalized to be safe)
        n_classes: K
        eps: numerical stability epsilon
    Returns:
        loss: scalar tensor
    """
    if class_probs.ndim != 2 or int(class_probs.shape[1]) != int(n_classes):
        raise ValueError("class_probs must have shape (batch, K)")

    y_true = y_true.to(torch.int64)
    y_onehot = F.one_hot(y_true - 1, num_classes=int(n_classes)).to(dtype=class_probs.dtype)  # (N, K)

    # Ensure valid probabilities even if tiny negatives appear from numerical subtraction.
    P = class_probs.clamp_min(0.0)
    P = P / P.sum(dim=1, keepdim=True).clamp_min(eps)

    # Soft observed matrix O and expected matrix E (both sum to N).
    O = y_onehot.t() @ P  # (K, K)
    a = y_onehot.sum(dim=0)  # (K,)
    b = P.sum(dim=0)  # (K,)
    N = float(P.shape[0])
    E = torch.outer(a, b) / max(1.0, N)  # (K, K)

    # Quadratic weight matrix.
    idx = torch.arange(int(n_classes), device=P.device, dtype=P.dtype)
    W = (idx[:, None] - idx[None, :]).pow(2) / float((int(n_classes) - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum().clamp_min(eps)
    return num / den
