# src/attribution/loo.py
import json
import os
from typing import Dict, Any
from src.pipeline.agent_runner import BaselinePipeline
from src.schema import ExecutionTrace, ProvenanceDocument, FaultType

class ItemLevelLOO:
    def __init__(self, pipeline: BaselinePipeline):
        self.pipeline = pipeline

    def evaluate_loo(self, sample_path: str, condition_key: str) -> Dict[str, Any]:
        """Runs Leave-One-Out attribution for each document in the sample."""
        with open(sample_path, "r") as f:
            data = json.load(f)

        original_docs = data["conditions"][condition_key]
        gold_answer = data["gold_answer"]
        
        # 1. Run baseline (full documents) - pass FaultType.NONE instead of model_name
        baseline_trace = self.pipeline.execute_pipeline(sample_path, condition_key, FaultType.NONE)
        baseline_success = gold_answer.lower() in baseline_trace.final_verdict.lower()

        loo_results = {}

        # 2. Iterate through each document, remove it, and evaluate effect
        for i, doc_to_remove in enumerate(original_docs):
            doc_id = doc_to_remove["doc_id"]
            
            # Create a modified copy without this document
            modified_docs = [d for d in original_docs if d["doc_id"] != doc_id]
            
            # Temporarily override condition docs in a temporary copy of the data file
            temp_data = data.copy()
            temp_data["conditions"][f"{condition_key}_loo_{doc_id}"] = modified_docs
            
            temp_path = f"data/temp_{doc_id}.json"
            with open(temp_path, "w") as tf:
                json.dump(temp_data, tf)

            # Run pipeline on ablated documents - pass FaultType.NONE
            ablated_trace = self.pipeline.execute_pipeline(temp_path, f"{condition_key}_loo_{doc_id}", FaultType.NONE)
            ablated_success = gold_answer.lower() in ablated_trace.final_verdict.lower()

            # Measure performance drop (LOO effect)
            is_critical = baseline_success and not ablated_success
            
            loo_results[doc_id] = {
                "ablated_success": ablated_success,
                "is_critical": is_critical,
                "final_verdict": ablated_trace.final_verdict
            }

            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        summary = {
            "query_id": data["query_id"],
            "condition": condition_key,
            "baseline_success": baseline_success,
            "loo_effects": loo_results
        }

        print(f"LOO Attribution completed for condition {condition_key}.")
        return summary

if __name__ == "__main__":
    pipeline = BaselinePipeline()
    loo_runner = ItemLevelLOO(pipeline)
    results = loo_runner.evaluate_loo("data/seed_sample.json", "A_lowC_lowP")
    print(json.dumps(results, indent=2))