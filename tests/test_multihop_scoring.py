
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


# --- Phase 0 gold-checked cases (v1.1 policy) ---
import pytest as _pytest
from src.evaluation.multihop_scoring import extract_answer as _extract, score_multihop_answer as _score


@_pytest.mark.parametrize("raw,gold,extracted,correct", [
    ("Answer: Yes", "Yes", "Yes", True),
    ("**Answer:** Yes", "Yes", "Yes", True),
    ("**Answer: No**", "No", "No", True),
    ("Reasoning.\nAnswer: yes.", "Yes", "Yes", True),
    ("Answer: No", "Yes", "No", False),
    ("Not yes", "Yes", "Not yes", False),
    ("The answer is yes.", "Yes", "The answer is yes.", False),
    ("Answer: Yes or No", "Yes", "", False),
    ("Answer: Insufficient evidence", "Yes", "", False),
    ("", "Yes", "", False),
    ("Answer: Sporting News", "Sporting News", "Sporting News", True),
    ("Answer: TechCrunch", "Sporting News", "TechCrunch", False),
])
def test_gold_checked_extraction_and_correctness(raw, gold, extracted, correct):
    r = _score(raw, gold)
    assert r["extracted_answer"] == extracted and r["correct"] is correct


def test_yes_no_uses_exact_mode_and_other_answers_keep_f1_rule():
    assert _score("Answer: Yes", "Yes")["scoring_mode"] == "multihop_yes_no_exact"
    assert _score("Answer: Sporting News and others", "Sporting News")["scoring_mode"] == "hotpotqa_em_f1"
