
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "multihoprag" / "MultiHopRAG.json"
OUTPUT = ROOT / "data" / "processed" / "multihoprag_empty_evidence_audit.json"


def main():
    with INPUT.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    empty_records = []
    field_counts = Counter()
    evidence_types = Counter()

    for index, item in enumerate(questions):
        evidence = item.get("evidence_list")

        if isinstance(evidence, list):
            evidence_types["list"] += 1
            if not evidence:
                reason = "empty_list"
            else:
                reason = None
        elif evidence is None:
            evidence_types["missing_or_null"] += 1
            reason = "missing_or_null"
        else:
            evidence_types[type(evidence).__name__] += 1
            reason = "unexpected_type"

        if reason:
            field_counts[reason] += 1
            empty_records.append({
                "index": index,
                "query": item.get("query"),
                "answer": item.get("answer"),
                "question_type": item.get("question_type"),
                "evidence_list_type": type(evidence).__name__,
                "evidence_list_value": evidence,
                "available_fields": sorted(item.keys()),
                "reason": reason,
            })

    summary = {
        "raw_question_count": len(questions),
        "evidence_field_types": dict(evidence_types),
        "empty_record_reasons": dict(field_counts),
        "empty_record_count": len(empty_records),
        "examples": empty_records[:10],
        "all_empty_records": empty_records,
        "interpretation": (
            "This audit describes the source records only. "
            "It does not infer why evidence is missing."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== EMPTY-EVIDENCE QUESTION AUDIT ===")
    print("Raw questions:", len(questions))
    print("Evidence field types:", dict(evidence_types))
    print("Empty-record reasons:", dict(field_counts))
    print("Records inspected:", len(empty_records))
    print("Output:", OUTPUT)
    print("\nFirst 5 examples:")
    for record in empty_records[:5]:
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
