
import json
from typing import Any, Dict

from src.pipeline.agent_runner import BaselinePipeline
from src.schema import FaultType
from src.evaluation.scoring import AnswerScorer


class ItemLevelLOO:
    def __init__(self, pipeline: BaselinePipeline):
        self.pipeline = pipeline

    def evaluate_loo(
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

        loo_results = {}

        for doc in original_docs:
            doc_id = doc["doc_id"]
            modified_docs = [
                item for item in original_docs
                if item["doc_id"] != doc_id
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

            loo_results[doc_id] = {
                "baseline_success": baseline_success,
                "ablated_success": ablated_success,
                "loo_correctness_effect": (
                    baseline_success and not ablated_success
                ),
                "final_verdict": ablated.final_verdict,
            }

        summary = {
            "query_id": data["query_id"],
            "condition": condition_key,
            "baseline_success": baseline_success,
            "loo_effects": loo_results,
            "critical_item_count": sum(
                result["loo_correctness_effect"]
                for result in loo_results.values()
            ),
        }

        return summary


if __name__ == "__main__":
    pipeline = BaselinePipeline()
    runner = ItemLevelLOO(pipeline)
    print(json.dumps(
        runner.evaluate_loo("data/seed_sample.json", "A_lowC_lowP"),
        indent=2,
    ))
