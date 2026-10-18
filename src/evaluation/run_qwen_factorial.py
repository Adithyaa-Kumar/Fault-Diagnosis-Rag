"""Run the real local Qwen model across the controlled 4x4 Phase 0 design."""
import json
from pathlib import Path
from src.pipeline.qwen_pipeline import QwenParallelPipeline
from src.schema import FaultType
from src.evaluation.scoring import AnswerScorer

CONDITIONS = ["A_lowC_lowP", "B_highC_lowP", "C_lowC_highP", "D_highC_highP"]
FAULTS = [FaultType.NONE, FaultType.RETRIEVAL, FaultType.SPECIALIST, FaultType.ARBITER]


def main(sample_path: str = "data/seed_sample.json"):
    pipeline = QwenParallelPipeline()
    rows = []
    for condition in CONDITIONS:
        for fault in FAULTS:
            trace = pipeline.execute_pipeline(sample_path, condition, fault)
            metrics = AnswerScorer.score_metrics(trace.final_verdict, trace.gold_answer,
                                                 trace.required_claims or None)
            rows.append({
                "query_id": trace.query_id,
                "condition": condition,
                "injected_fault_layer": fault.value,
                "predicted_fault_layer": trace.predicted_fault_layer.value if trace.predicted_fault_layer else None,
                "diagnosis_correct": trace.predicted_fault_layer == fault,
                "answer_correct": metrics["correct"],
                "exact_match": metrics["exact_match"],
                "token_f1": metrics["token_f1"],
                "final_verdict": trace.final_verdict,
                "prompt_tokens_measured": trace.prompt_tokens_measured,
                "generated_tokens_measured": trace.generated_tokens_measured,
                "model_calls_measured": trace.model_calls_measured,
                "model_id": trace.model_id,
            })
    labels = [f.value for f in FAULTS]
    confusion = {a: {p: 0 for p in labels + ["unknown"]} for a in labels}
    for row in rows:
        predicted = row["predicted_fault_layer"] or "unknown"
        confusion[row["injected_fault_layer"]][predicted] += 1
    prompt_counts = [r["prompt_tokens_measured"] for r in rows]
    generated_counts = [r["generated_tokens_measured"] for r in rows]
    calls = [r["model_calls_measured"] for r in rows]
    report = {
        "experiment": "real_local_qwen_controlled_factorial_phase0",
        "dataset": "synthetic_controlled_fixture",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "conditions": CONDITIONS,
        "fault_types": labels,
        "expected_combinations": len(CONDITIONS) * len(FAULTS),
        "executed_combinations": len(rows),
        "complete_factorial_coverage": len(rows) == len(CONDITIONS) * len(FAULTS),
        "diagnosis_accuracy_engineering_only": (
            sum(r["diagnosis_correct"] for r in rows) / len(rows) if rows else None
        ),
        "diagnosis_confusion_matrix": confusion,
        "measured_costs": {
            "prompt_tokens_total": sum(prompt_counts) if all(isinstance(x, int) for x in prompt_counts) else None,
            "generated_tokens_total": sum(generated_counts) if all(isinstance(x, int) for x in generated_counts) else None,
            "model_calls_total": sum(calls) if all(isinstance(x, int) for x in calls) else None,
            "note": "Totals are reported only from actual model-wrapper measurements; missing measurements remain null.",
        },
        "runs": rows,
        "limitations": [
            "One controlled synthetic query; no statistical generalization.",
            "Synthetic claims and source-cluster IDs are experimental fixtures, not real-world evidence or lineage.",
            "Fault diagnosis uses a fixed observation-only heuristic, not a learned diagnoser.",
            "This run tests real-model pipeline integration, not the research hypothesis.",
        ],
    }
    out = Path("results/qwen_factorial_phase0.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Real-Qwen factorial coverage: {len(rows)}/{len(CONDITIONS) * len(FAULTS)}")
    print(f"Diagnosis accuracy (engineering check only): {report['diagnosis_accuracy_engineering_only']:.1%}" if rows else "No rows")
    print("Measured costs:", json.dumps(report["measured_costs"]))
    print(f"Report: {out}")
    return report


if __name__ == "__main__":
    main()
