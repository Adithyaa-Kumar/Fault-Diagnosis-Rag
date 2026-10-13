# src/attribution/cluster_ablation.py
import json
import os
from typing import Dict, Any
from src.pipeline.agent_runner import BaselinePipeline
from src.schema import FaultType

class ClusterAblation:
    def __init__(self, pipeline: BaselinePipeline):
        self.pipeline = pipeline

    def evaluate_cluster_ablation(self, sample_path: str, condition_key: str) -> Dict[str, Any]:
        """Runs cluster-level counterfactual ablation by removing entire provenance clusters."""
        with open(sample_path, "r") as f:
            data = json.load(f)

        original_docs = data["conditions"][condition_key]
        gold_answer = data["gold_answer"]
        
        # Identify unique clusters present in this condition
        clusters = list(set(d["source_cluster_id"] for d in original_docs))
        
        # 1. Run baseline
        baseline_trace = self.pipeline.execute_pipeline(sample_path, condition_key, FaultType.NONE)
        baseline_success = gold_answer.lower() in baseline_trace.final_verdict.lower()

        cluster_results = {}

        # 2. Iterate through each provenance cluster, remove all its docs, and evaluate effect
        for cluster_id in clusters:
            # Filter out all documents belonging to this cluster
            modified_docs = [d for d in original_docs if d["source_cluster_id"] != cluster_id]
            
            temp_data = data.copy()
            temp_data["conditions"][f"{condition_key}_cluster_{cluster_id}"] = modified_docs
            
            temp_path = f"data/temp_cluster_{cluster_id}.json"
            with open(temp_path, "w") as tf:
                json.dump(temp_data, tf)

            # Run pipeline on cluster-ablated documents
            ablated_trace = self.pipeline.execute_pipeline(temp_path, f"{condition_key}_cluster_{cluster_id}", FaultType.NONE)
            ablated_success = gold_answer.lower() in ablated_trace.final_verdict.lower()

            is_repairable_or_critical = baseline_success and not ablated_success
            
            cluster_results[cluster_id] = {
                "ablated_success": ablated_success,
                "is_critical_cluster": is_repairable_or_critical,
                "final_verdict": ablated_trace.final_verdict
            }

            if os.path.exists(temp_path):
                os.remove(temp_path)

        summary = {
            "query_id": data["query_id"],
            "condition": condition_key,
            "baseline_success": baseline_success,
            "cluster_ablation_effects": cluster_results
        }

        print(f"Cluster-Level Ablation completed for condition {condition_key}.")
        return summary

if __name__ == "__main__":
    pipeline = BaselinePipeline()
    cluster_runner = ClusterAblation(pipeline)
    results = cluster_runner.evaluate_cluster_ablation("data/seed_sample.json", "A_lowC_lowP")
    print(json.dumps(results, indent=2))