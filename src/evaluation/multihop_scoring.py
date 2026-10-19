
"""Conservative answer extraction for MultiHop-RAG pilot experiments."""

import re

from src.evaluation.scoring import AnswerScorer


CONTRADICTION_PATTERNS = (
    r"\bbut\b.{0,160}\b(?:does not|doesn't|did not|cannot|can't|unable to)\b",
    r"\bhowever\b.{0,160}\b(?:does not|doesn't|did not|cannot|can't|unable to)\b",
    r"\b(?:insufficient evidence|insufficient information|"
    r"unable to determine|not enough information|cannot determine)\b",
)


def extract_answer(output: str) -> str:
    """Extract a short answer, conservatively rejecting obvious contradictions."""
    if not isinstance(output, str):
        return ""

    # Drop markdown emphasis/backticks so "**Answer:** Yes" parses like "Answer: Yes".
    text = re.sub(r"[*`]+", "", output).strip()
    if not text:
        return ""

    # Keep the complete response for validity checks. A labelled final answer
    # must not hide a contradiction or abstention stated in the reasoning.
    full_text = text
    labelled = re.search(
        r"(?im)^\s*answer\s*:\s*(.+?)\s*$",
        full_text,
    )
    answer_text = labelled.group(1).strip() if labelled else full_text

    abstention_pattern = (
        r"\b(insufficient evidence|insufficient information|the evidence is insufficient|"
        r"unable to determine|not enough information|cannot determine|could not determine)\b"
    )
    # An explicit abstention anywhere in the response invalidates a confident
    # final answer; otherwise reasoning/final-answer contradictions can score as correct.
    if re.search(abstention_pattern, full_text, flags=re.IGNORECASE):
        return ""

    text = answer_text

    # For yes/no answers, reject explicit contradictions or abstentions.
    yes_no = re.match(
        r"^\s*(yes|no)(?=$|[\s.,:;!?—-])",
        text,
        flags=re.IGNORECASE,
    )
    if yes_no:
        remainder = text[yes_no.end():].strip()

        if re.match(
            r"^(or\b|and\s+(yes|no)\b)",
            remainder,
            flags=re.IGNORECASE,
        ):
            return ""

        has_contradiction = any(
            re.search(pattern, full_text, flags=re.IGNORECASE | re.DOTALL)
            for pattern in CONTRADICTION_PATTERNS
        )

        if has_contradiction:
            return ""

        return yes_no.group(1).capitalize()

    # Conservative fallback for entity and other non-yes/no answers.
    first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    return first_sentence.strip()


def score_multihop_answer(output: str, gold_answer: str) -> dict:
    """Extract an answer and score it with the unchanged canonical scorer."""
    extracted = extract_answer(output)
    gold_norm = AnswerScorer._normalise(gold_answer) if isinstance(gold_answer, str) else ""
    if gold_norm in ("yes", "no"):
        # Yes/no gold needs an exact yes/no: token-F1 >= 0.5 would accept "Not yes".
        exact = AnswerScorer._normalise(extracted) == gold_norm
        metrics = {"correct": exact, "exact_match": exact,
                   "token_f1": 1.0 if exact else 0.0, "scoring_mode": "multihop_yes_no_exact"}
    else:
        metrics = AnswerScorer.score_metrics(extracted, gold_answer)

    return {
        "raw_output": output,
        "extracted_answer": extracted,
        "gold_answer": gold_answer,
        **metrics,
    }
