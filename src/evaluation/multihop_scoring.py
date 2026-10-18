
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

    text = output.strip()
    if not text:
        return ""

    # Preserve the full response for contradiction and abstention checks.
    labelled = re.search(
        r"(?im)^\s*answer\s*:\s*(.+?)\s*$",
        text,
    )
    if labelled:
        text = labelled.group(1).strip()

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
            re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            for pattern in CONTRADICTION_PATTERNS
        )

        has_abstention = bool(
            re.search(
                r"\b("
                r"insufficient evidence|"
                r"insufficient information|"
                r"the evidence is insufficient|"
                r"unable to determine|"
                r"not enough information|"
                r"cannot determine|"
                r"could not determine"
                r")\b",
                text,
                flags=re.IGNORECASE,
            )
        )

        if has_contradiction or has_abstention:
            return ""

        return yes_no.group(1).capitalize()

    # Conservative fallback for entity and other non-yes/no answers.
    first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    return first_sentence.strip()


def score_multihop_answer(output: str, gold_answer: str) -> dict:
    """Extract an answer and score it with the unchanged canonical scorer."""
    extracted = extract_answer(output)
    metrics = AnswerScorer.score_metrics(extracted, gold_answer)

    return {
        "raw_output": output,
        "extracted_answer": extracted,
        "gold_answer": gold_answer,
        **metrics,
    }
