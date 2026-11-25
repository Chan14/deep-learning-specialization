"""
losses.py — MSE loss with L2 regularization for the 3-layer MLP.

Design notes:
    - Pure function of (y_hat, y, params, lambda_reg). No mutation.
    - Returns a dict with separate 'data' and 'reg' components so the
      training loop can log both — useful for diagnosing whether
      regularization is well-tuned.
    - L2 applies to weights (W1, W2, W3) only, never biases. Biases have
      no regularization-relevant overfitting behavior and empirically
      hurt when penalized.
    - Gradient of the L2 term is NOT computed here. It lives at the
      optimizer step, where `grad_W += (lambda_reg / m) * W` is added
      to each weight gradient. See optimizer.py.

Tested in tests/gd_variants/test_losses.py.
"""

import numpy as np


def compute_data_loss(y_hat: np.ndarray, y: np.ndarray) -> float:
    """
    MSE data loss: (1/m) * sum((y_hat - y)^2).

    Args:
        y_hat: Predictions, shape (m, 1).
        y:     Targets, shape (m, 1).

    Returns:
        Scalar loss value (Python float).

    Invariant: matches the loss implied by model.backward's dL/dy_hat = (2/m)(y_hat - y).
    """
    m = y.shape[0]
    L = (1 / m) * np.sum((y_hat - y) ** 2)
    return float(
        L
    )  # defensive: forces scalar even if something else regresses. Note - np.sum without the axis is always a scalar.


def compute_reg_loss(params: dict, lambda_reg: float, m: int) -> float:
    """
    L2 regularization loss: (lambda_reg / (2 * m)) * sum over l of ||W_l||_F^2.

    Args:
        params:     Parameter dict. Iterates only over keys starting with 'W'.
        lambda_reg: Regularization strength (>= 0). Zero means no regularization.
        m:          Number of examples in the batch used for the data loss.
                    Must match the m used in compute_data_loss to keep the
                    data and reg terms on the same scale.

    Returns:
        Scalar reg loss value (Python float). Zero if lambda_reg is 0.

    Invariant: biases (keys starting with 'b') are never included.
    """
    if lambda_reg == 0:
        return 0.0

    l2_sum = sum(np.sum(W**2) for name, W in params.items() if name.startswith("W"))
    reg_loss = lambda_reg * l2_sum / (2 * m)
    return float(reg_loss)  # defensive: forces scalar even if something else regresses.


def compute_total_loss(
    y_hat: np.ndarray,
    y: np.ndarray,
    params: dict,
    lambda_reg: float,
) -> dict:
    """
    Total loss = data loss + L2 regularization loss.

    Args:
        y_hat:      Predictions, shape (m, 1).
        y:          Targets, shape (m, 1).
        params:     Parameter dict (weights penalized, biases not).
        lambda_reg: L2 strength.

    Returns:
        A dict with keys:
            'data':  scalar MSE data loss.
            'reg':   scalar L2 penalty (0.0 if lambda_reg == 0).
            'total': scalar sum of the above.

    The dict form is intentional: the training loop logs all three so we can
    diagnose over/under-regularization by watching their relative magnitudes.
    """
    m = y.shape[0]
    data_loss = compute_data_loss(y_hat, y)
    reg_loss = compute_reg_loss(params, lambda_reg, m)
    return {"data": data_loss, "reg": reg_loss, "total": data_loss + reg_loss}
