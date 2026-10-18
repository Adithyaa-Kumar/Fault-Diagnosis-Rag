import json
from typing import Any, Dict
from src.evaluation.scoring import AnswerScorer
from src.schema import FaultType


class ItemLevelLOO:
    """Leave one document out and score every counterfactual with AnswerScorer."""
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def evaluate_loo(self, sample_path: str, condition_key: str) -> Dict[str, Any]:
        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = data["conditions"][condition_key]
        baseline = self.pipeline.execute_pipeline(
            sample_path, condition_key, FaultType.NONE, write_log=False
        )
        baseline_metrics = AnswerScorer.score_metrics(
            baseline.final_verdict, baseline.gold_answer, baseline.required_claims or None
        )
        effects = {}
        for doc in docs:
            modified_docs = [d for d in docs if d["doc_id"] != doc["doc_id"]]
            trace = self.pipeline.execute_pipeline(
                sample_path, condition_key, FaultType.NONE,
                documents_override=modified_docs, write_log=False,
            )
            metrics = AnswerScorer.score_metrics(
                trace.final_verdict, trace.gold_answer, trace.required_claims or None
            )
            effects[doc["doc_id"]] = {
                "baseline_correct": baseline_metrics["correct"],
                "intervention_correct": metrics["correct"],
                "baseline_exact_match": baseline_metrics["exact_match"],
                "intervention_exact_match": metrics["exact_match"],
                "baseline_token_f1": baseline_metrics["token_f1"],
                "intervention_token_f1": metrics["token_f1"],
                "is_loo_correctness_effect": bool(baseline_metrics["correct"] and not metrics["correct"]),
                "is_repair": AnswerScorer.is_repair(baseline_metrics["correct"], metrics["correct"]),
                "final_verdict": trace.final_verdict,
            }
        repairs = sum(x["is_repair"] for x in effects.values())
        eligible = int(not baseline_metrics["correct"])
        return {
            "query_id": data["query_id"], "condition": condition_key,
            "baseline_correct": baseline_metrics["correct"],
            "baseline_exact_match": baseline_metrics["exact_match"],
            "baseline_token_f1": baseline_metrics["token_f1"],
            "loo_effects": effects,
            "loo_correctness_effect_count": sum(x["is_loo_correctness_effect"] for x in effects.values()),
            "loo_repair_count": repairs,
            "eligible_failed_baseline_trajectories": eligible,
            "loo_repair_rate": (int(repairs > 0) / eligible) if eligible else None,
            "interpretation_note": "LOO correctness effects are not by themselves a calibrated attribution-ambiguity metric.",
        }


if __name__ == "__main__":
    print(json.dumps(ItemLevelLOO.__name__))
