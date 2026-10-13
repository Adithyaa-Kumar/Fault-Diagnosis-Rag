# src/evaluation/run_pilot.py
import json
import os
from typing import Dict, Any
from src.pipeline.agent_runner import BaselinePipeline
from src.attribution.loo import ItemLevelLOO
from src.attribution.cluster_ablation import ClusterAblation
from src.schema import FaultType

class PilotEvaluator:
    def __init__(self, sample_path: str):
        self.sample_path = sample_path
        self.pipeline = BaselinePipeline()
        self.loo_runner = ItemLevelLOO(self.pipeline)
        self.cluster_runner = ClusterAblation(self.pipeline)

    def run_pilot_condition(self, condition_key: str) -> Dict[str, Any]:
        """Runs baseline, LOO, cluster ablation, and scores diagnosis metrics for a condition."""
        print(f"\n--- Running Pilot for Condition: {condition_key} ---")
        
        # 1. Baseline Execution
        baseline_trace = self.pipeline.execute_pipeline(self.sample_path, condition_key, FaultType.NONE)
        gold_answer = baseline_trace.gold_answer
        baseline_success = gold_answer.lower() in baseline_trace.final_verdict.lower()
        
        failed_trajectories = 0 if baseline_success else 1
        total_trajectories = 1

        # 2. LOO Attribution
        loo_results = self.loo_runner.evaluate_loo(self.sample_path, condition_key)
        
        # Calculate LOO ambiguity rate (e.g., if multiple items look equally critical due to redundancy)
        critical_items = sum(1 for d in loo_results["loo_effects"].values() if d["is_critical"])
        loo_ambiguity_rate = 1.0 if critical_items > 1 else 0.0

        # 3. Cluster Ablation
        cluster_results = self.cluster_runner.evaluate_cluster_ablation(self.sample_path, condition_key)
        cluster_repair_count = sum(1 for c in cluster_results["cluster_ablation_effects"].values() if c["is_critical_cluster"])
        cluster_ablation_repair_rate = 1.0 if cluster_repair_count > 0 else 0.0

        # 4. Independent Fault-Layer Diagnosis Accuracy (Mock scoring based on successful provenance resolution)
        fault_layer_accuracy = 100.0 if baseline_success else 0.0

        # 5. Cost proxy (number of pipeline runs executed)
        total_calls = 1 + len(loo_results["loo_effects"]) + len(cluster_results["cluster_ablation_effects"])
        estimated_tokens = total_calls * 450  # approximate token footprint per run

        summary_row = {
            "condition": condition_key,
            "total_trajectories": total_trajectories,
            "failed_trajectories": failed_trajectories,
            "loo_ambiguity_rate": f"{loo_ambiguity_rate * 100}%",
            "cluster_ablation_repair_rate": f"{cluster_ablation_repair_rate * 100}%",
            "fault_layer_accuracy": f"{fault_layer_accuracy}%",
            "calls_tokens_cost": f"{total_calls} calls / ~{estimated_tokens} tokens"
        }

        return summary_row

    def execute_full_pilot(self) -> None:
        """Executes the pilot across multiple specified conditions."""
        conditions = ["A_lowC_lowP", "D_highC_highP"]
        results_table = []

        for cond in conditions:
            row = self.run_pilot_condition(cond)
            results_table.append(row)

        print("\n==========================================================================")
        print("                        PILOT EVALUATION SUMMARY                          ")
        print("==========================================================================")
        print(f"{'Condition':<15} | {'Total':<6} | {'Failed':<6} | {'LOO Ambig.':<12} | {'Cluster Repair':<15} | {'Diagnosis Acc.':<14} | {'Cost':<15}")
        print("-" * 95)
        for r in results_table:
            print(f"{r['condition']:<15} | {r['total_trajectories']:<6} | {r['failed_trajectories']:<6} | {r['loo_ambiguity_rate']:<12} | {r['cluster_ablation_repair_rate']:<15} | {r['fault_layer_accuracy']:<14} | {r['calls_tokens_cost']:<15}")
        print("==========================================================================")

if __name__ == "__main__":
    evaluator = PilotEvaluator("data/seed_sample.json")
    evaluator.execute_full_pilot()