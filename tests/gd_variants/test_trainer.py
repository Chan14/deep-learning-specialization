from functools import partial

import numpy as np
import pytest

from dlspec.gd_variants.losses import compute_data_loss
from dlspec.gd_variants.model import forward, init_parameters
from dlspec.gd_variants.trainer import (
    batch_gd_iterator,
    minibatch_iterator,
    train,
    true_sgd_iterator,
)

# ---------- Batch iterators ----------


@pytest.fixture
def Xy():
    rng = np.random.default_rng(0)
    return rng.normal(size=(100, 8)), rng.normal(size=(100, 1))


def _covers_every_example_once(batches, X):
    X_concat = np.concatenate([Xb for Xb, _ in batches], axis=0)
    return np.allclose(np.sort(X_concat[:, 0]), np.sort(X[:, 0]))


def test_batch_gd_yields_whole_dataset_once(Xy):
    X, y = Xy
    batches = list(batch_gd_iterator(X, y, np.random.default_rng(0)))
    assert len(batches) == 1
    Xb, yb = batches[0]
    assert Xb.shape == (100, 8) and yb.shape == (100, 1)


def test_true_sgd_yields_m_batches_with_leading_dim_1(Xy):
    X, y = Xy
    batches = list(true_sgd_iterator(X, y, np.random.default_rng(0)))
    assert len(batches) == 100
    for Xb, yb in batches:
        assert Xb.shape == (1, 8) and yb.shape == (1, 1)
    assert _covers_every_example_once(batches, X)


def test_minibatch_keeps_the_smaller_last_batch(Xy):
    X, y = Xy
    batches = list(minibatch_iterator(X, y, np.random.default_rng(0), batch_size=32))
    assert [Xb.shape[0] for Xb, _ in batches] == [32, 32, 32, 4]
    assert _covers_every_example_once(batches, X)


def test_each_call_reshuffles_and_same_seed_reproduces(Xy):
    X, y = Xy
    rng = np.random.default_rng(7)
    first = list(true_sgd_iterator(X, y, rng))[0][0]
    second = list(true_sgd_iterator(X, y, rng))[0][0]
    assert not np.array_equal(first, second), "rng not advancing"

    rep_a = list(true_sgd_iterator(X, y, np.random.default_rng(42)))[0][0]
    rep_b = list(true_sgd_iterator(X, y, np.random.default_rng(42)))[0][0]
    assert np.array_equal(rep_a, rep_b)


# ---------- Training dynamics ----------


@pytest.fixture(scope="module")
def synthetic():
    """A known-learnable problem, y = X @ w_true + noise. If the MLP can't fit
    this, the trainer is broken."""
    rng = np.random.default_rng(0)
    w_true = rng.normal(size=(8, 1))
    X_train = rng.normal(size=(500, 8))
    y_train = X_train @ w_true + 0.1 * rng.normal(size=(500, 1))
    X_dev = rng.normal(size=(100, 8))
    y_dev = X_dev @ w_true + 0.1 * rng.normal(size=(100, 1))
    return X_train, y_train, X_dev, y_dev


def _train(data, iterator, lr, **kwargs):
    X_train, y_train, X_dev, y_dev = data
    defaults = dict(
        lambda_reg=0.0, max_epochs=60, patience=100, rng=np.random.default_rng(42)
    )
    return train(
        params=init_parameters([8, 32, 16, 1], seed=1),
        X_train=X_train,
        y_train=y_train,
        X_dev=X_dev,
        y_dev=y_dev,
        batch_iterator=iterator,
        learning_rate=lr,
        **{**defaults, **kwargs},
    )


# SGD needs a smaller alpha, as in the real experiment.
METHODS = {
    "batch_gd": (batch_gd_iterator, 0.05),
    "minibatch": (partial(minibatch_iterator, batch_size=32), 0.02),
    "true_sgd": (true_sgd_iterator, 0.005),
}


@pytest.fixture(scope="module")
def results(synthetic):
    return {name: _train(synthetic, it, lr) for name, (it, lr) in METHODS.items()}


@pytest.mark.parametrize("name", METHODS)
def test_training_at_least_halves_the_loss(results, name):
    history = results[name]["history"]["train_data_loss"]
    assert history[-1] < history[0] / 2


@pytest.mark.parametrize("name", METHODS)
def test_best_params_snapshot_is_really_the_best(results, synthetic, name):
    _, _, X_dev, y_dev = synthetic
    r = results[name]
    y_hat_best, _ = forward(X_dev, r["best_params"])
    assert (
        compute_data_loss(y_hat_best, y_dev) <= r["history"]["dev_data_loss"][-1] + 1e-8
    )


def test_all_three_methods_reach_comparable_loss(results):
    # A method-specific bug (e.g. a wrong shape in the SGD path) would show up
    # as one method ending far worse than the others.
    losses = [r["best_dev_loss"] for r in results.values()]
    assert max(losses) / min(losses) < 5.0, dict(zip(results, losses, strict=True))


def test_early_stopping_fires(synthetic):
    r = _train(synthetic, batch_gd_iterator, 0.05, max_epochs=500, patience=5)
    assert r["stopped_early"]
    assert r["final_epoch"] < 400


def test_per_update_losses_logged_only_for_requested_epochs(synthetic):
    r = _train(
        synthetic,
        partial(minibatch_iterator, batch_size=32),
        0.02,
        max_epochs=10,
        log_per_update_epochs=(0, 5, 9),
    )
    assert set(r["per_update_history"]) == {0, 5, 9}
    # 500 examples, batch size 32 -> ceil(500 / 32) = 16 updates per epoch.
    assert all(len(v) == 16 for v in r["per_update_history"].values())


def test_same_seed_gives_identical_trajectories(synthetic):
    def run():
        return _train(
            synthetic,
            partial(minibatch_iterator, batch_size=32),
            0.02,
            lambda_reg=0.01,
            max_epochs=10,
        )

    assert run()["history"]["dev_data_loss"] == run()["history"]["dev_data_loss"]
