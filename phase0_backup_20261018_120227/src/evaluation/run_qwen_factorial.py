import json
from collections import Counter
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
                "query_id": trace.query_id, "condition": condition,
                "injected_fault_layer": fault.value,
                "predicted_fault_layer": trace.predicted_fault_layer.value if trace.predicted_fault_layer else None,
                "diagnosis_correct": trace.predicted_fault_layer == fault,
                "answer_correct": metrics["correct"], "exact_match": metrics["exact_match"],
                "token_f1": metrics["token_f1"], "final_verdict": trace.final_verdict,
            })
    confusion = {a.value: {p.value: 0 for p in FAULTS} for a in FAULTS}
    for row in rows:
        confusion[row["injected_fault_layer"]][row["predicted_fault_layer"]] += 1
    report = {
        "experiment": "real_local_qwen_controlled_factorial_phase0",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "conditions": CONDITIONS, "fault_types": [x.value for x in FAULTS],
        "expected_combinations": 16, "executed_combinations": len(rows),
        "complete_factorial_coverage": len(rows) == 16,
        "diagnosis_accuracy": sum(x["diagnosis_correct"] for x in rows) / len(rows) if rows else None,
        "diagnosis_confusion_matrix": confusion, "runs": rows,
        "limitations": ["One controlled synthetic query; no statistical generalization.",
                        "Synthetic claims are experimental fixtures, not real-world factual claims.",
                        "Fault diagnosis uses a fixed observation-only heuristic, not an LLM diagnoser.",
                        "This run tests real-LLM pipeline integration, not the research hypothesis."],
    }
    out = Path("results/qwen_factorial_phase0.json"); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Real-Qwen factorial coverage: {len(rows)}/16")
    print(f"Diagnosis accuracy (engineering check only): {report['diagnosis_accuracy']:.1%}" if rows else "No rows")
    print(f"Report: {out}")
    return report

if __name__ == "__main__":
    main()