import numpy as np

from dlspec.gd_variants.grad_check import relative_error, run_grad_check


def test_backward_matches_numeric_gradient(capsys):
    errors = run_grad_check()
    assert set(errors) == {"W1", "b1", "W2", "b2", "W3", "b3"}
    assert max(errors.values()) < 1e-5, errors


def test_relative_error_is_zero_for_equal_and_one_for_opposite():
    g = np.array([[1.0, -2.0], [3.0, 0.5]])
    assert relative_error(g, g) == 0.0
    assert np.isclose(relative_error(g, -g), 1.0)
