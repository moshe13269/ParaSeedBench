from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
from scipy.spatial.distance import jensenshannon


def normalized_hamming(a: str, b: str) -> float:
    if len(a) != len(b):
        raise ValueError("Semantic states must have equal length")
    if not a:
        raise ValueError("Semantic states cannot be empty")
    return sum(x != y for x, y in zip(a, b)) / len(a)


def mean_pairwise_hamming(states: Iterable[str]) -> float:
    values = [str(state) for state in states]
    pairs = list(combinations(values, 2))
    return float(np.mean([normalized_hamming(a, b) for a, b in pairs])) if pairs else np.nan


def mean_pairwise_cosine_distance(embeddings: np.ndarray) -> float:
    if len(embeddings) < 2:
        return np.nan
    normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    similarity = normalized @ normalized.T
    upper = np.triu_indices(len(normalized), 1)
    return float(np.mean(1.0 - similarity[upper]))


def pairwise_state_jsd(groups: list[list[str]], alpha: float = 0.5) -> float:
    """Mean pairwise Jensen-Shannon divergence (base 2) over semantic states."""
    if len(groups) < 2:
        return np.nan
    support = sorted({state for group in groups for state in group})
    if not support:
        return np.nan
    probabilities = []
    for group in groups:
        counts = np.asarray([group.count(state) for state in support], dtype=float) + alpha
        probabilities.append(counts / counts.sum())
    values = [
        float(jensenshannon(probabilities[i], probabilities[j], base=2.0) ** 2)
        for i, j in combinations(range(len(probabilities)), 2)
    ]
    return float(np.mean(values))


def percentile_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    draws = values[indices].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

