"""Single canonical scorer for the controlled fixture and HotpotQA.

Controlled fixture: all required claim match terms must be present and no
abstention/explicit denial may occur. HotpotQA: normalized token F1 >= 0.5
is the correctness threshold; exact match and F1 are also exposed separately.
"""
import re
import string
from collections import Counter
from typing import Any, Iterable, Optional


class AnswerScorer:
    ABSTENTION_PHRASES = (
        "insufficient evidence", "unable to determine", "not enough information",
        "cannot determine", "could not determine", "insufficient information",
    )
    NEGATION_PATTERNS = (
        r"did not authorize", r"not authorized by", r"failed to authorize",
        r"incorrect to say.*trade oversight commission",
        r"claim that.*is false", r"statement that.*is false",
    )
    HOTPOTQA_F1_THRESHOLD = 0.5

    @staticmethod
    def _normalise(text: str) -> str:
        text = (text or "").lower()
        text = "".join(ch for ch in text if ch not in set(string.punctuation))
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        return " ".join(text.split())

    @classmethod
    def _is_abstention_or_denial(cls, verdict: str) -> bool:
        clean = cls._normalise(verdict)
        if not clean or any(p in clean for p in cls.ABSTENTION_PHRASES):
            return True
        return any(re.search(pattern, (verdict or "").lower()) for pattern in cls.NEGATION_PATTERNS)

    @classmethod
    def score_metrics(
        cls, verdict: str, gold_answer: str,
        required_claims: Optional[Iterable[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Return a consistent set of metrics for every pipeline path."""
        if not isinstance(verdict, str) or not isinstance(gold_answer, str):
            return {"correct": False, "exact_match": False, "token_f1": 0.0, "scoring_mode": "invalid_input"}
        if cls._is_abstention_or_denial(verdict):
            return {"correct": False, "exact_match": False, "token_f1": 0.0,
                    "scoring_mode": "required_claims" if required_claims else "hotpotqa_em_f1"}

        if required_claims:
            clean = cls._normalise(verdict)
            passed = True
            for claim in required_claims:
                terms = claim.get("match_terms", [])
                if not terms or not all(cls._normalise(term) in clean for term in terms):
                    passed = False
                    break
            return {"correct": passed, "exact_match": passed, "token_f1": 1.0 if passed else 0.0,
                    "scoring_mode": "required_claims"}

        pred = cls._normalise(verdict)
        gold = cls._normalise(gold_answer)
        if not pred or not gold:
            return {"correct": False, "exact_match": False, "token_f1": 0.0, "scoring_mode": "hotpotqa_em_f1"}
        exact = pred == gold
        pred_tokens, gold_tokens = pred.split(), gold.split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        precision = overlap / len(pred_tokens) if pred_tokens else 0.0
        recall = overlap / len(gold_tokens) if gold_tokens else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        return {"correct": bool(exact or f1 >= cls.HOTPOTQA_F1_THRESHOLD),
                "exact_match": exact, "token_f1": f1, "scoring_mode": "hotpotqa_em_f1"}

    @classmethod
    def score_answer(cls, verdict: str, gold_answer: str,
                     required_claims: Optional[Iterable[dict[str, Any]]] = None) -> bool:
        return bool(cls.score_metrics(verdict, gold_answer, required_claims)["correct"])
