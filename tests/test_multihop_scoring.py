
import pytest

from src.evaluation.multihop_scoring import (
    extract_answer,
    score_multihop_answer,
)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("Yes. The evidence supports the claim.", "Yes"),
        ("No. The articles describe different events.", "No"),
        ("Answer: Sam Bankman-Fried", "Sam Bankman-Fried"),
        ("Answer: Google", "Google"),
        ("", ""),
    ],
)
def test_answer_extraction(output, expected):
    assert extract_answer(output) == expected


def test_abstention_is_not_scored_as_correct_yes():
    result = score_multihop_answer(
        "The evidence is insufficient to decide.",
        "Yes",
    )
    assert result["correct"] is False


@pytest.mark.parametrize(
    "output",
    [
        "Yes, but the evidence does not support the claim.",
        "Yes. However, the evidence is insufficient to determine this.",
    ],
)
def test_contradictory_yes_answer_is_not_accepted(output):
    result = score_multihop_answer(output, "Yes")
    assert result["correct"] is False
