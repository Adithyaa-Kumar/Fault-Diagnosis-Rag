import json
from typing import Any, Dict
from src.evaluation.scoring import AnswerScorer
from src.schema import FaultType


def is_repair(baseline_success: bool, intervention_success: bool) -> bool:
    """A repair occurs only when an incorrect baseline becomes correct."""
    return AnswerScorer.is_repair(baseline_success, intervention_success)


class ClusterAblation:
    """Remove each source cluster and score counterfactual outputs consistently."""
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def evaluate_cluster_ablation(self, sample_path: str, condition_key: str) -> Dict[str, Any]:
        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = data["conditions"][condition_key]
        baseline = self.pipeline.execute_pipeline(
            sample_path, condition_key, FaultType.NONE, write_log=False
        )
        baseline_metrics = AnswerScorer.score_metrics(
            baseline.final_verdict, baseline.gold_answer, baseline.required_claims or None
        )
        clusters = sorted({d["source_cluster_id"] for d in docs})
        effects = {}
        for cluster_id in clusters:
            modified_docs = [d for d in docs if d["source_cluster_id"] != cluster_id]
            trace = self.pipeline.execute_pipeline(
                sample_path, condition_key, FaultType.NONE,
                documents_override=modified_docs, write_log=False,
            )
            metrics = AnswerScorer.score_metrics(
                trace.final_verdict, trace.gold_answer, trace.required_claims or None
            )
            effects[cluster_id] = {
                "baseline_correct": baseline_metrics["correct"],
                "intervention_correct": metrics["correct"],
                "baseline_exact_match": baseline_metrics["exact_match"],
                "intervention_exact_match": metrics["exact_match"],
                "baseline_token_f1": baseline_metrics["token_f1"],
                "intervention_token_f1": metrics["token_f1"],
                "is_cluster_ablation_correctness_effect": bool(baseline_metrics["correct"] and not metrics["correct"]),
                "is_repair": is_repair(baseline_metrics["correct"], metrics["correct"]),
                "final_verdict": trace.final_verdict,
            }
        eligible = int(not baseline_metrics["correct"])
        repair_count = int(any(x["is_repair"] for x in effects.values()))
        return {
            "query_id": data["query_id"], "condition": condition_key,
            "baseline_correct": baseline_metrics["correct"],
            "baseline_exact_match": baseline_metrics["exact_match"],
            "baseline_token_f1": baseline_metrics["token_f1"],
            "cluster_ablation_effects": effects,
            "repair_count": repair_count,
            "eligible_failed_baseline_trajectories": eligible,
            "cluster_ablation_repair_rate": (repair_count / eligible) if eligible else None,
            "interpretation_note": "Ablation is a repair only if a failed baseline becomes correct; rate is N/A when no failed baseline is eligible.",
        }


if __name__ == "__main__":
    print(json.dumps(ClusterAblation.__name__))
