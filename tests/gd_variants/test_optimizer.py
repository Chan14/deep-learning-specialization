import numpy as np
import pytest

from dlspec.gd_variants.model import backward, forward, init_parameters
from dlspec.gd_variants.optimizer import optimizer_step


@pytest.fixture
def batch():
    rng = np.random.default_rng(0)
    return rng.normal(size=(10, 8)), rng.normal(size=(10, 1))


def _grads(params, X, y):
    y_hat, cache = forward(X, params)
    return backward(y_hat, y, params, cache)


def test_step_moves_params_and_keeps_keys_and_shapes(batch):
    X, y = batch
    params = init_parameters([8, 32, 16, 1], seed=1)
    before = {k: v.copy() for k, v in params.items()}

    optimizer_step(
        params, _grads(params, X, y), learning_rate=0.01, lambda_reg=0.1, m=10
    )

    assert set(params) == {"W1", "b1", "W2", "b2", "W3", "b3"}
    for k in params:
        assert params[k].shape == before[k].shape
    assert not np.array_equal(params["W1"], before["W1"])
    assert not np.array_equal(params["b1"], before["b1"])


def test_step_does_not_mutate_grads(batch):
    X, y = batch
    params = init_parameters([8, 32, 16, 1], seed=1)
    grads = _grads(params, X, y)
    snapshot = {k: v.copy() for k, v in grads.items()}

    optimizer_step(params, grads, learning_rate=0.01, lambda_reg=0.1, m=10)

    for k in grads:
        assert np.array_equal(grads[k], snapshot[k]), (
            f"optimizer_step mutated grads['{k}']"
        )


def test_l2_changes_weight_updates_but_not_bias_updates(batch):
    X, y = batch
    params_a = init_parameters([8, 32, 16, 1], seed=1)
    params_b = init_parameters([8, 32, 16, 1], seed=1)
    grads_a = _grads(params_a, X, y)
    grads_b = _grads(params_b, X, y)

    optimizer_step(params_a, grads_a, learning_rate=0.01, lambda_reg=0.0, m=10)
    optimizer_step(params_b, grads_b, learning_rate=0.01, lambda_reg=0.1, m=10)

    for b in ("b1", "b2", "b3"):
        assert np.allclose(params_a[b], params_b[b]), f"L2 affected {b}"
    assert not np.allclose(params_a["W1"], params_b["W1"]), "L2 did not affect W1"
