import numpy as np

from dlspec.gd_variants.model import backward, forward, init_parameters


def test_forward_and_backward_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 8))
    y = rng.normal(size=(5, 1))
    params = init_parameters([8, 32, 16, 1])

    y_hat, cache = forward(X, params)
    grads = backward(y_hat, y, params, cache)

    assert y_hat.shape == (5, 1)
    # Invariant: grads[k].shape == params[k].shape for every k.
    assert grads.keys() == params.keys()
    for k in params:
        assert grads[k].shape == params[k].shape


def test_init_uses_he_for_hidden_and_xavier_for_output():
    # Wide layers so the sample variance is close to the target variance.
    params = init_parameters([2000, 1000, 1], seed=0)
    assert abs(np.var(params["W1"]) - 2 / 2000) < 0.05 * (2 / 2000)  # He
    assert abs(np.var(params["W2"]) - 1 / 1000) < 0.15 * (1 / 1000)  # Xavier
    assert not params["b1"].any() and not params["b2"].any()


def test_forward_does_not_mutate_params():
    params = init_parameters([8, 32, 16, 1], seed=1)
    before = {k: v.copy() for k, v in params.items()}
    forward(np.ones((3, 8)), params)
    for k in params:
        assert np.array_equal(params[k], before[k])
