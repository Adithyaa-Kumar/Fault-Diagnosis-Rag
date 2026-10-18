import json
from typing import Any, Dict
from src.evaluation.scoring import AnswerScorer
from src.pipeline.agent_runner import BaselinePipeline
from src.schema import FaultType


def is_repair(baseline_success: bool, intervention_success: bool) -> bool:
    """A repair occurs only when a failed baseline becomes correct."""
    return (not baseline_success) and intervention_success


class ClusterAblation:
    def __init__(self, pipeline: BaselinePipeline):
        self.pipeline = pipeline

    def evaluate_cluster_ablation(self, sample_path: str, condition_key: str) -> Dict[str, Any]:
        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = data["conditions"][condition_key]
        baseline = self.pipeline.execute_pipeline(sample_path, condition_key, FaultType.NONE)
        baseline_success = AnswerScorer.score_answer(
            baseline.final_verdict, baseline.gold_answer, baseline.required_claims
        )
        clusters = sorted({d["source_cluster_id"] for d in docs})
        effects = {}
        for cluster_id in clusters:
            modified_docs = [d for d in docs if d["source_cluster_id"] != cluster_id]
            trace = self.pipeline.execute_pipeline(
                sample_path, condition_key, FaultType.NONE,
                documents_override=modified_docs,
            )
            success = AnswerScorer.score_answer(trace.final_verdict, trace.gold_answer, trace.required_claims)
            effects[cluster_id] = {
                "baseline_success": baseline_success,
                "ablated_success": success,
                "is_cluster_ablation_correctness_effect": bool(baseline_success and not success),
                # Correct repair predicate; ordinary ablation is not automatically a repair.
                "is_repair": is_repair(baseline_success, success),
                "final_verdict": trace.final_verdict,
            }
        failed_baseline_denominator = 0 if baseline_success else 1
        repair_count = int(any(x["is_repair"] for x in effects.values()))
        # Count each baseline trajectory at most once, even if several cluster interventions repair it.
        repair_rate = (repair_count / failed_baseline_denominator) if failed_baseline_denominator else None
        return {
            "query_id": data["query_id"], "condition": condition_key,
            "baseline_success": baseline_success, "cluster_ablation_effects": effects,
            "repair_count": repair_count,
            "eligible_failed_baseline_trajectories": failed_baseline_denominator,
            "cluster_ablation_repair_rate": repair_rate,
            "interpretation_note": "Ablation correctness effects are not repairs. Repair rate is N/A when no baseline failures are eligible.",
        }


if __name__ == "__main__":
    print(json.dumps(ClusterAblation(BaselinePipeline()).evaluate_cluster_ablation("data/seed_sample.json", "A_lowC_lowP"), indent=2))
