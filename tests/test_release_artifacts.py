from pathlib import Path

from paraseedbench.verify_release import (
    _verify_automatic_results,
    _verify_human_results,
    _verify_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


def test_released_protocol_and_paper_values():
    _verify_protocol(ROOT)
    _verify_human_results(ROOT)
    _verify_automatic_results(ROOT)
