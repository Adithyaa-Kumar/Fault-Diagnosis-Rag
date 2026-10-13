import json
from typing import Any, Dict

from src.pipeline.agent_runner import BaselinePipeline
from src.schema import FaultType
from src.evaluation.scoring import AnswerScorer


class ClusterAblation:
    def __init__(self, pipeline: BaselinePipeline):
        self.pipeline = pipeline

    def evaluate_cluster_ablation(
        self, sample_path: str, condition_key: str
    ) -> Dict[str, Any]:
        with open(sample_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        original_docs = data["conditions"][condition_key]

        baseline = self.pipeline.execute_pipeline(
            sample_path, condition_key, FaultType.NONE
        )
        baseline_success = AnswerScorer.score_answer(
            baseline.final_verdict, baseline.gold_answer
        )

        clusters = sorted({
            doc["source_cluster_id"] for doc in original_docs
        })

        results = {}

        for cluster_id in clusters:
            modified_docs = [
                doc for doc in original_docs
                if doc["source_cluster_id"] != cluster_id
            ]

            ablated = self.pipeline.execute_pipeline(
                sample_path,
                condition_key,
                FaultType.NONE,
                documents_override=modified_docs,
            )

            ablated_success = AnswerScorer.score_answer(
                ablated.final_verdict, ablated.gold_answer
            )

            results[cluster_id] = {
                "ablated_success": ablated_success,
                "cluster_ablation_correctness_effect": (
                    baseline_success and not ablated_success
                ),
                "final_verdict": ablated.final_verdict,
                "is_repair": (not baseline_success) and ablated_success,
            }

        eligible_repair_interventions = (
            len(results) if not baseline_success else 0
        )

        repair_count = sum(
            item["is_repair"] for item in results.values()
        )

        repair_rate = (
            repair_count / eligible_repair_interventions
            if eligible_repair_interventions > 0
            else None
        )

        return {
            "query_id": data["query_id"],
            "condition": condition_key,
            "baseline_success": baseline_success,
            "cluster_ablation_effects": results,
            "critical_cluster_count": sum(
                item["cluster_ablation_correctness_effect"]
                for item in results.values()
            ),
            "eligible_repair_interventions": eligible_repair_interventions,
            "repair_count": repair_count,
            "repair_rate": repair_rate,
            "repair_rate_note": (
                "Not measured by ordinary cluster ablation. "
                "A repair rate requires a separately defined intervention "
                "on eligible failed baseline trajectories."
            ),
        }


if __name__ == "__main__":
    pipeline = BaselinePipeline()
    runner = ClusterAblation(pipeline)
    print(json.dumps(
        runner.evaluate_cluster_ablation(
            "data/seed_sample.json", "A_lowC_lowP"
        ),
        indent=2,
    ))