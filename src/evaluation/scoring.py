
import re


class AnswerScorer:
    """Canonical answer scorer for the current TOC research pilot."""

    TARGET = "trade oversight commission"
    ALIASES = ("trade oversight commission", "toc")
    WRONG_ENTITIES = (
        "department of treasury",
        "ministry of finance",
        "unknown monetary policy",
    )
    ABSTENTIONS = (
        "insufficient evidence",
        "unable to determine",
        "not enough information",
        "cannot determine",
    )

    @classmethod
    def score_answer(cls, verdict: str, gold_answer: str) -> bool:
        if not isinstance(verdict, str) or not isinstance(gold_answer, str):
            return False

        text = verdict.lower().strip()
        gold = gold_answer.lower().strip()

        if not text or not gold:
            return False

        if any(phrase in text for phrase in cls.ABSTENTIONS):
            return False

        # Reject explicit negation of the target authorization.
        negation_patterns = (
            r"\b(?:did not|didn't|never)\s+authorize\b",
            r"\bnot\s+authorized\s+by\b",
            r"\bfailed\s+to\s+authorize\b",
            r"\bincorrect\s+to\s+say\b",
        )
        if any(re.search(p, text) for p in negation_patterns):
            return False

        # Identify the target as a complete phrase/token, not a substring
        # inside an unrelated word.
        has_target = any(
            re.search(r"\b" + re.escape(alias) + r"\b", text)
            for alias in cls.ALIASES
        )
        if not has_target:
            return False

        # A wrong agency appearing alongside the target is ambiguous,
        # so reject it rather than silently counting it as correct.
        if any(entity in text for entity in cls.WRONG_ENTITIES):
            return False

        # This pilot has one canonical gold entity. Do not silently
        # treat arbitrary gold answers as equivalent to the TOC answer.
        gold_is_target = any(
            re.search(r"\b" + re.escape(alias) + r"\b", gold)
            for alias in cls.ALIASES
        )
        if not gold_is_target:
            raise ValueError(
                f"Unsupported gold answer for this scorer: {gold_answer!r}"
            )

        return True
