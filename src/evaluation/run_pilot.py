
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.pipeline.agent_runner import BaselinePipeline
from src.attribution.loo import ItemLevelLOO
from src.attribution.cluster_ablation import ClusterAblation
from src.schema import FaultType
from src.evaluation.scoring import AnswerScorer


CONDITIONS = [
    "A_lowC_lowP",
    "B_highC_lowP",
    "C_lowC_highP",
    "D_highC_highP",
]

FAULT_TYPES = [
    FaultType.NONE,
    FaultType.RETRIEVAL,
    FaultType.SPECIALIST,
    FaultType.ARBITER,
]


class PilotEvaluator:
    def __init__(self, sample_path: str):
        self.sample_path = sample_path
        self.pipeline = BaselinePipeline()
        self.loo_runner = ItemLevelLOO(self.pipeline)
        self.cluster_runner = ClusterAblation(self.pipeline)

    def execute_full_pilot(self) -> dict[str, Any]:
        rows = []
        confusion = Counter()
        condition_diagnosis = Counter()

        for condition in CONDITIONS:
            for fault_type in FAULT_TYPES:
                trace = self.pipeline.execute_pipeline(
                    self.sample_path, condition, fault_type
                )

                answer_correct = AnswerScorer.score_answer(
                    trace.final_verdict, trace.gold_answer
                )

                predicted = trace.predicted_fault_layer
                actual = trace.injected_fault_layer
                diagnosis_correct = predicted == actual

                confusion[(actual.value, predicted.value)] += 1
                condition_diagnosis[
                    (condition, actual.value, diagnosis_correct)
                ] += 1

                rows.append({
                    "query_id": trace.query_id,
                    "condition": condition,
                    "injected_fault_layer": actual.value,
                    "predicted_fault_layer": predicted.value,
                    "diagnosis_correct": diagnosis_correct,
                    "answer_correct": answer_correct,
                    "is_failed_trajectory": not answer_correct,
                    "final_verdict": trace.final_verdict,
                })

        total = len(rows)
        correct_diagnoses = sum(
            row["diagnosis_correct"] for row in rows
        )

        # Attribution analyses are run separately on the no-fault
        # baseline for each experimental condition.
        attribution_results = {}

        for condition in CONDITIONS:
            loo = self.loo_runner.evaluate_loo(
                self.sample_path, condition
            )
            cluster = self.cluster_runner.evaluate_cluster_ablation(
                self.sample_path, condition
            )

            attribution_results[condition] = {
                "loo": loo,
                "cluster_ablation": cluster,
            }

        fault_labels = [fault.value for fault in FAULT_TYPES]
        confusion_matrix = {
            actual: {
                predicted: confusion[(actual, predicted)]
                for predicted in fault_labels
            }
            for actual in fault_labels
        }

        report = {
            "pilot_type": "deterministic_mock_pipeline",
            "expected_condition_fault_combinations": 16,
            "executed_condition_fault_combinations": total,
            "complete_factorial_coverage": total == 16,
            "total_trajectories": total,
            "failed_trajectories": sum(
                row["is_failed_trajectory"] for row in rows
            ),
            "diagnosis_correct_count": correct_diagnoses,
            "diagnosis_accuracy": (
                correct_diagnoses / total if total else None
            ),
            "confusion_matrix": confusion_matrix,
            "runs": rows,
            "attribution_results": attribution_results,
            "token_usage": None,
            "api_cost": None,
            "limitations": [
                "Deterministic mock pipeline; not real LLM evaluation.",
                "One synthetic query; no statistical generalization.",
                "LOO and cluster-ablation effects are correctness effects, "
                "not validated causal attribution ambiguity.",
                "Cluster repair rate is not measured by ordinary ablation.",
            ],
        }

        Path("logs").mkdir(exist_ok=True)
        Path("logs/pilot_summary.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        print("\nPHASE 0 PILOT SUMMARY")
        print("=" * 76)
        print(f"Condition/fault combinations: {total}/16")
        print(f"Failed trajectories: {report['failed_trajectories']}")
        print(
            "Fault diagnosis accuracy: "
            f"{correct_diagnoses}/{total} "
            f"({report['diagnosis_accuracy']:.1%})"
            if total else "Fault diagnosis accuracy: N/A"
        )
        print("\nConfusion matrix:")
        print(json.dumps(confusion_matrix, indent=2))
        print("\nReport written to logs/pilot_summary.json")
        print("Token usage and API cost: NOT MEASURED")
        print("=" * 76)

        return report


if __name__ == "__main__":
    evaluator = PilotEvaluator("data/seed_sample.json")
    evaluator.execute_full_pilot()
