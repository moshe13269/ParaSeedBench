"""Finite-grid descriptive estimands, not a random-effects variance estimator."""
from itertools import combinations, product
import numpy as np
from scipy.spatial.distance import jensenshannon


def decomposition(q):
    q = np.asarray(q, dtype=float)
    if q.ndim != 3 or min(q.shape[:2]) < 2 or q.shape[2] < 1 or not np.isin(q, [0, 1]).all():
        raise ValueError("Expected binary [wording, seed, atom] grid with J,S >= 2")
    j, s, _ = q.shape
    mu = q.mean(axis=(0, 1), keepdims=True)
    a = q.mean(axis=1, keepdims=True) - mu
    b = q.mean(axis=0, keepdims=True) - mu
    interaction = q - mu - a - b
    vw, vs, vi = float((a*a).mean()), float((b*b).mean()), float((interaction*interaction).mean())
    return {"variance_wording": vw, "variance_seed": vs, "variance_interaction": vi,
            "variance_total": float(((q - mu)**2).mean()),
            "wording_disagreement": 2*j/(j-1)*(vw+vi),
            "seed_disagreement": 2*s/(s-1)*(vs+vi)}


def crossfit_worst(a):
    """Split-symmetrized held-out selected-worst accuracy.

    ``a`` is a ``[wording, seed]`` correctness grid.  For every *unordered*
    balanced partition of the seeds, the function selects all wordings tied
    for the lowest accuracy on one half, averages their scores on the other
    half, reverses train/test, and finally averages all directed evaluations.

    Fixing seed index zero in the first member of each partition enumerates
    each unordered partition exactly once.  Thus the paper's J=4, S=8 design
    uses C(7,3)=35 partitions and 70 directed held-out evaluations.  Averaging
    ties makes the result invariant to wording order; enumerating every
    partition makes it invariant to seed order.  This remains a finite-grid
    sensitivity diagnostic, not an unbiased population minimum.

    The historical function name is retained for CSV/API compatibility.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 2:
        raise ValueError("Expected a [wording, seed] grid with J,S >= 2")
    if a.shape[1] % 2:
        raise ValueError("Held-out selected-worst accuracy requires an even seed count")
    if not np.isfinite(a).all():
        raise ValueError("Held-out selected-worst accuracy requires finite values")

    seed_count = a.shape[1]
    half = seed_count // 2
    all_seeds = np.arange(seed_count)
    held_out_scores = []
    # Every unordered balanced partition has exactly one half containing seed 0.
    for remainder in combinations(range(1, seed_count), half - 1):
        first = np.asarray((0, *remainder), dtype=int)
        second = np.setdiff1d(all_seeds, first, assume_unique=True)
        for train, test in ((first, second), (second, first)):
            training_accuracy = a[:, train].mean(axis=1)
            selected = np.flatnonzero(
                np.isclose(training_accuracy, training_accuracy.min(), rtol=0, atol=1e-12)
            )
            held_out_scores.append(float(a[np.ix_(selected, test)].mean()))
    return float(np.mean(held_out_scores))


def state_jsd(q, alpha=0.5):
    q = np.asarray(q, dtype=int)
    support = list(product((0, 1), repeat=q.shape[2]))
    if len(support) > 4096:
        raise ValueError("Too many atoms for enumerated semantic-state JSD")
    probabilities = []
    for wording in q:
        states = [tuple(row) for row in wording]
        counts = np.array([states.count(state) for state in support]) + alpha
        probabilities.append(counts / counts.sum())
    return float(np.mean([jensenshannon(a, b, base=2)**2 for a, b in combinations(probabilities, 2)]))


def stratified_ci(values, categories, rng, n_boot=2000):
    values, categories = np.asarray(values, float), np.asarray(categories)
    if n_boot < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    labels = sorted(set(categories))
    arrays = [values[categories == label] for label in labels]
    arrays = [a[np.isfinite(a)] for a in arrays]
    # Undefined metric in an entire category must not silently reweight the target.
    if any(len(a) == 0 for a in arrays) or not arrays:
        return np.nan, np.nan, np.nan
    estimate = float(np.mean([a.mean() for a in arrays]))
    draws = np.stack([a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1) for a in arrays]).mean(axis=0)
    return estimate, *np.percentile(draws, [2.5, 97.5]).tolist()
