import numpy as np

from paraseedbench.metrics import (
    mean_pairwise_cosine_distance,
    mean_pairwise_hamming,
    normalized_hamming,
    pairwise_state_jsd,
)


def test_hamming():
    assert normalized_hamming("101", "111") == 1 / 3
    assert mean_pairwise_hamming(["11", "10", "00"]) == 2 / 3


def test_cosine_distance():
    values = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert np.isclose(mean_pairwise_cosine_distance(values), 2 / 3)


def test_jsd_zero_for_same_distributions():
    assert np.isclose(pairwise_state_jsd([["10", "11"], ["10", "11"]]), 0.0)


def test_jsd_positive_for_different_distributions():
    assert pairwise_state_jsd([["00"] * 8, ["11"] * 8]) > 0.5
