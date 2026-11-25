"""
optimizer.py — Vanilla gradient descent step with L2 regularization.

Design notes:
    - Mutates params in place. Returns None. Standard Python in-place idiom.
    - L2 gradient (lambda_reg / m) * W is added to WEIGHT gradients only.
      Biases never receive L2. This matches the loss-side invariant from
      losses.py.
    - The caller MUST pass the m that was used to compute `grads`. This is
      not optional — batch size mismatch between data gradient and L2
      gradient would silently make regularization strength depend on batch
      size, contaminating the Batch/SGD/Mini-batch comparison.
    - `lambda_reg` has no default. Making it required forces the caller to
      be intentional about regularization. A forgotten `lambda_reg` argument
      is a bug; defaulting to 0.0 would let that bug pass silently.
    - Vanilla GD only. No momentum, no Adam. Those come later.

Tested in tests/gd_variants/test_optimizer.py.
"""


def optimizer_step(
    params: dict,
    grads: dict,
    learning_rate: float,
    lambda_reg: float,
    m: int,
) -> None:
    """
    Apply one gradient descent update to the parameters, in place.

    Update rule:
        For weights W_l:
            W_l <- W_l - learning_rate * (grads[W_l] + (lambda_reg / m) * W_l)
        For biases b_l:
            b_l <- b_l - learning_rate * grads[b_l]

    Args:
        params:        Parameter dict. Mutated in place.
        grads:         Gradient dict from model.backward. Must have same keys
                       and shapes as params.
        learning_rate: Step size alpha. Must be > 0.
        lambda_reg:    L2 regularization strength. Zero means no regularization.
                       Pass 0.0 explicitly rather than relying on a default.
        m:             Batch size used to compute `grads`. Needed so the L2
                       gradient (lambda_reg / m) * W is on the same scale as
                       the data-loss gradient, which already has a 2/m factor
                       baked in from backward().

    Returns:
        None. Params are updated in place.

    Invariants:
        1. params.keys() is unchanged after the call.
        2. Each params[k].shape is unchanged.
        3. L2 term is added ONLY to weight gradients (keys starting with 'W').
        4. If lambda_reg == 0, behavior is identical to vanilla GD.
    """
    for name in params:
        grad = grads[name]
        if name.startswith("W"):
            # Use `+`, not `+=`: grads is the caller's data. The variance probe
            # reads these same arrays later and needs the unmodified data gradient.
            grad = grad + (lambda_reg / m) * params[name]
        params[name] -= learning_rate * grad
