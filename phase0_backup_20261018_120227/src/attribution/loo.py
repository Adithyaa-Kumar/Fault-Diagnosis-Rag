import json
from typing import Any, Dict
from src.evaluation.scoring import AnswerScorer
from src.pipeline.agent_runner import BaselinePipeline
from src.schema import FaultType


class ItemLevelLOO:
    def __init__(self, pipeline: BaselinePipeline):
        self.pipeline = pipeline

    def evaluate_loo(self, sample_path: str, condition_key: str) -> Dict[str, Any]:
        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = data["conditions"][condition_key]
        baseline = self.pipeline.execute_pipeline(sample_path, condition_key, FaultType.NONE)
        baseline_success = AnswerScorer.score_answer(
            baseline.final_verdict, baseline.gold_answer, baseline.required_claims
        )
        effects = {}
        for doc in docs:
            modified_docs = [d for d in docs if d["doc_id"] != doc["doc_id"]]
            trace = self.pipeline.execute_pipeline(
                sample_path, condition_key, FaultType.NONE,
                documents_override=modified_docs,
            )
            success = AnswerScorer.score_answer(trace.final_verdict, trace.gold_answer, trace.required_claims)
            effects[doc["doc_id"]] = {
                "baseline_success": baseline_success,
                "ablated_success": success,
                "is_loo_correctness_effect": bool(baseline_success and not success),
                "final_verdict": trace.final_verdict,
            }
        return {
            "query_id": data["query_id"], "condition": condition_key,
            "baseline_success": baseline_success, "loo_effects": effects,
            "loo_correctness_effect_count": sum(x["is_loo_correctness_effect"] for x in effects.values()),
            "interpretation_note": "Correctness effects only; not a calibrated attribution-ambiguity metric.",
        }


if __name__ == "__main__":
    import json
    print(json.dumps(ItemLevelLOO(BaselinePipeline()).evaluate_loo("data/seed_sample.json", "A_lowC_lowP"), indent=2))
