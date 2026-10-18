import json
from pathlib import Path
from typing import Any, Dict, List
from src.attribution.cluster_ablation import ClusterAblation
from src.attribution.loo import ItemLevelLOO
from src.evaluation.scoring import AnswerScorer
from src.pipeline.agent_runner import BaselinePipeline
from src.schema import FaultType

CONDITIONS = ["A_lowC_lowP", "B_highC_lowP", "C_lowC_highP", "D_highC_highP"]
FAULT_TYPES = [FaultType.NONE, FaultType.RETRIEVAL, FaultType.SPECIALIST, FaultType.ARBITER]


class PilotEvaluator:
    def __init__(self, sample_path: str):
        self.sample_path = sample_path
        self.pipeline = BaselinePipeline()
        self.loo_runner = ItemLevelLOO(self.pipeline)
        self.cluster_runner = ClusterAblation(self.pipeline)

    def run_factorial_pilot(self) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for condition in CONDITIONS:
            for fault_type in FAULT_TYPES:
                trace = self.pipeline.execute_pipeline(self.sample_path, condition, fault_type)
                success = AnswerScorer.score_answer(trace.final_verdict, trace.gold_answer, trace.required_claims)
                predicted = trace.predicted_fault_layer
                results.append({
                    "query_id": trace.query_id,
                    "condition": condition,
                    "injected_fault_layer": fault_type.value,
                    "predicted_fault_layer": predicted.value if predicted else None,
                    "diagnosis_correct": bool(predicted == fault_type),
                    "answer_correct": success,
                    "trajectory_failed": not success,
                    "final_verdict": trace.final_verdict,
                })

        # LOO and cluster ablation are run on the no-fault baseline for each condition.
        attribution = []
        for condition in CONDITIONS:
            loo = self.loo_runner.evaluate_loo(self.sample_path, condition)
            cluster = self.cluster_runner.evaluate_cluster_ablation(self.sample_path, condition)
            attribution.append({
                "condition": condition,
                "loo_correctness_effect_count": loo["loo_correctness_effect_count"],
                "loo_effects": loo["loo_effects"],
                "cluster_ablation_correctness_effect_count": sum(
                    x["is_cluster_ablation_correctness_effect"]
                    for x in cluster["cluster_ablation_effects"].values()
                ),
                "cluster_ablation_repair_rate": cluster["cluster_ablation_repair_rate"],
                "eligible_failed_baseline_trajectories": cluster["eligible_failed_baseline_trajectories"],
                "cluster_ablation_effects": cluster["cluster_ablation_effects"],
            })

        confusion: Dict[str, Dict[str, int]] = {
            actual.value: {pred.value: 0 for pred in FAULT_TYPES} for actual in FAULT_TYPES
        }
        for row in results:
            confusion[row["injected_fault_layer"]][row["predicted_fault_layer"]] += 1
        diagnosis_accuracy = sum(r["diagnosis_correct"] for r in results) / len(results) if results else 0.0
        summary = {
            "status": "PHASE0_PILOT_EXECUTED_NOT_AUTOMATICALLY_APPROVED",
            "conditions": CONDITIONS,
            "fault_types": [f.value for f in FAULT_TYPES],
            "expected_factorial_combinations": len(CONDITIONS) * len(FAULT_TYPES),
            "observed_factorial_combinations": len(results),
            "factorial_complete": len(results) == len(CONDITIONS) * len(FAULT_TYPES),
            "mock_heuristic_diagnosis_accuracy": diagnosis_accuracy,
            "diagnosis_confusion_matrix": confusion,
            "factorial_results": results,
            "attribution_results": attribution,
            "cost_reporting": "Mock pipeline executions only. Token usage and API cost were not measured and are not reported.",
            "metric_caution": "LOO and cluster-ablation values are correctness effects, not validated ambiguity estimates. Repair rate is N/A when the no-fault baseline has no eligible failures.",
        }
        out = Path("logs") / "pilot_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._print_summary(summary)
        print(f"\nDetailed summary written to: {out}")
        return summary

    @staticmethod
    def _print_summary(summary: Dict[str, Any]) -> None:
        print("\nPHASE 0 FACTORIAL PILOT")
        print("Condition          Fault       Predicted    Answer  Diagnosis")
        print("-" * 68)
        for row in summary["factorial_results"]:
            print(f"{row['condition']:<19} {row['injected_fault_layer']:<11} {str(row['predicted_fault_layer']):<12} {str(row['answer_correct']):<7} {str(row['diagnosis_correct'])}")
        print("-" * 68)
        print(f"Factorial combinations: {summary['observed_factorial_combinations']}/{summary['expected_factorial_combinations']}")
        print(f"Mock heuristic diagnosis accuracy (NOT a research result): {summary['mock_heuristic_diagnosis_accuracy']:.1%}")
        print("Token/API costs: NOT REPORTED (not measured)")

    # Backwards-compatible entry point used by previous instructions.
    def execute_full_pilot(self) -> None:
        self.run_factorial_pilot()


if __name__ == "__main__":
    PilotEvaluator("data/seed_sample.json").execute_full_pilot()
