import numpy as np
import pandas as pd

from paraseedbench.analyze import case_metrics, latex_table


def synthetic_grid():
    states = {
        ("p0", 1): "11",
        ("p1", 1): "10",
        ("p0", 2): "11",
        ("p1", 2): "11",
    }
    return pd.DataFrame([
        {
            "model": "toy_model",
            "case_id": "case_0",
            "category": "toy",
            "variant_id": variant,
            "seed": seed,
            "semantic_state": state,
            "all_correct": int(state == "11"),
            "image_path": f"{variant}_{seed}.png",
        }
        for (variant, seed), state in states.items()
    ])


def test_case_metrics_factorization():
    result = case_metrics(synthetic_grid(), {})
    assert result["mean_accuracy"] == 0.75
    assert result["robust_accuracy"] == 0.5
    assert result["wording_disagreement"] == 0.25
    assert result["seed_disagreement"] == 0.25
    assert result["semantic_jsd"] > 0
    assert np.isnan(result["valid_diversity"])


def test_latex_table_is_compilable_fragment():
    rows = []
    for metric in [
        "mean_accuracy", "robust_accuracy", "wording_disagreement",
        "seed_disagreement", "semantic_jsd", "valid_diversity",
    ]:
        rows.append({
            "model": "toy_model", "category": "all", "metric": metric,
            "estimate": 0.1, "ci_low": 0.05, "ci_high": 0.15,
        })
    table = latex_table(pd.DataFrame(rows))
    assert "toy\\_model" in table
    assert "\\begin{tabular}" in table
    assert "0.100 [0.050, 0.150]" in table
