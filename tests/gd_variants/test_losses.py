import numpy as np
import pytest

from dlspec.gd_variants.losses import compute_total_loss
from dlspec.gd_variants.model import forward, init_parameters


@pytest.fixture
def predictions():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 8))
    y = rng.normal(size=(10, 1))
    params = init_parameters([8, 32, 16, 1], seed=1)
    y_hat, _ = forward(X, params)
    return y_hat, y, params


def test_zero_lambda_means_zero_reg(predictions):
    y_hat, y, params = predictions
    loss = compute_total_loss(y_hat, y, params, lambda_reg=0.0)
    assert loss["reg"] == 0.0
    assert loss["total"] == pytest.approx(loss["data"], abs=1e-12)


def test_positive_lambda_adds_reg_and_leaves_data_loss_alone(predictions):
    y_hat, y, params = predictions
    loss0 = compute_total_loss(y_hat, y, params, lambda_reg=0.0)
    loss1 = compute_total_loss(y_hat, y, params, lambda_reg=0.1)
    assert loss1["reg"] > 0
    assert loss1["total"] == pytest.approx(loss1["data"] + loss1["reg"], abs=1e-12)
    assert loss1["data"] == pytest.approx(loss0["data"], abs=1e-12)


def test_biases_are_never_penalized(predictions):
    y_hat, y, params = predictions
    big_biases = {k: (v + 100.0 if k.startswith("b") else v) for k, v in params.items()}
    reg = compute_total_loss(y_hat, y, params, lambda_reg=0.1)["reg"]
    reg_big_b = compute_total_loss(y_hat, y, big_biases, lambda_reg=0.1)["reg"]
    assert reg_big_b == pytest.approx(reg)
