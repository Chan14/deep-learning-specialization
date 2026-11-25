import numpy as np
import pytest

from dlspec.gd_variants.data import load_and_prepare_data


@pytest.fixture(scope="module")
def data():
    # California Housing is downloaded once and cached in ~/scikit_learn_data.
    try:
        return load_and_prepare_data()
    except OSError as e:  # no network and no cached copy
        pytest.skip(f"California Housing not available: {e}")


def test_split_sizes_and_shapes(data):
    m = sum(data[k].shape[0] for k in ("X_train", "X_dev", "X_test"))
    assert m == 20640
    for split in ("train", "dev", "test"):
        assert data[f"X_{split}"].shape[1] == 8
        assert data[f"y_{split}"].ndim == 2 and data[f"y_{split}"].shape[1] == 1


def test_train_is_standardized_and_dev_test_are_close(data):
    assert abs(data["X_train"].mean()) < 0.01
    assert abs(data["X_train"].std() - 1.0) < 0.01
    # Dev/test use train statistics, so they are close to 0 mean but not exactly.
    assert abs(data["X_dev"].mean()) < 0.1
    assert abs(data["X_test"].mean()) < 0.1


def test_same_seed_gives_same_split(data):
    again = load_and_prepare_data()
    assert np.array_equal(again["X_dev"], data["X_dev"])
