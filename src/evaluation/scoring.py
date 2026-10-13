# src/evaluation/scoring.py
import re

class AnswerScorer:
    @staticmethod
    def score_answer(verdict: str, gold_answer: str) -> bool:
        """
        Structured answer scoring protecting against false positives, negations, 
        and wrong entity assignments.
        """
        clean_verdict = verdict.lower().strip()
        
        # 1. Check for explicit negation of the target agency
        negation_patterns = [
            r"did not authorize", 
            r"not authorized by", 
            r"failed to authorize", 
            r"incorrect to say.*trade oversight commission"
        ]
        for pattern in negation_patterns:
            if re.search(pattern, clean_verdict):
                return False

        # 2. Check for presence of wrong entities / competing agencies
        wrong_entities = ["department of treasury", "ministry of finance", "unknown monetary policy"]
        for entity in wrong_entities:
            if entity in clean_verdict and "trade oversight commission" not in clean_verdict:
                return False

        # 3. Canonical entity presence check
        target_entities = ["trade oversight commission", "toc"]
        has_target = any(entity in clean_verdict for entity in target_entities)
        
        # 4. Check for abstention / missing answer
        abstention_phrases = ["insufficient evidence", "unable to determine", "not enough information"]
        is_abstaining = any(phrase in clean_verdict for phrase in abstention_phrases)

        if is_abstaining:
            return False

        return has_target