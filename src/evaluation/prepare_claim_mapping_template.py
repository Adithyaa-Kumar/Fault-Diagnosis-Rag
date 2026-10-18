
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "multihoprag"
    / "claim_annotations_30.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "multihoprag"
    / "claim_mapping_template_27.json"
)

VALID_LABELS = {
    "supports",
    "contradicts",
    "insufficient",
    "unrelated",
}


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("Expected an annotations array.")

    # Select the 27 questions that do not yet have atomic claims.
    remaining = [
        a for a in annotations
        if not a.get("required_claims")
    ]

    if len(remaining) != 27:
        raise ValueError(
            f"Expected 27 questions without claims, found {len(remaining)}. "
            "No output was written."
        )

    output_records = []

    for annotation in remaining:
        sample_id = annotation["sample_id"]
        evidence_items = annotation.get("evidence_items", [])

        output_records.append({
            "sample_id": sample_id,
            "question_type": annotation.get("question_type"),
            "query": annotation.get("query"),
            "annotation_status": "pending",
            "required_claims": [],
            "evidence_items": evidence_items,
            "evidence_support": [],
            "mapping_instructions": {
                "claim_id_format": "C1, C2, C3, ...",
                "allowed_labels": sorted(VALID_LABELS),
                "mapping_fields": [
                    "claim_id",
                    "evidence_id",
                    "label",
                    "rationale",
                ],
                "rules": [
                    "Split the query into atomic claims.",
                    "Map evidence to claims individually.",
                    "Do not use the gold answer to infer evidence support.",
                    "Do not infer provenance from publisher identity.",
                    "Use insufficient when the excerpt does not establish a claim.",
                    "Keep uncertain mappings pending human review.",
                ],
            },
        })

    result = {
        "dataset": "MultiHop-RAG",
        "artifact_type": "claim_by_evidence_mapping_template",
        "record_count": len(output_records),
        "annotation_status": "pending",
        "records": output_records,
    }

    # Write to a new file; never overwrite the main annotation file.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Verify the written JSON.
    with OUTPUT_PATH.open("r", encoding="utf-8") as f:
        verified = json.load(f)

    ids = [r["sample_id"] for r in verified["records"]]

    if len(ids) != 27 or len(set(ids)) != 27:
        OUTPUT_PATH.unlink(missing_ok=True)
        raise ValueError("Output verification failed: record IDs are invalid.")

    if any(r["required_claims"] for r in verified["records"]):
        raise ValueError("Unexpected claims were generated automatically.")

    print("\nCLAIM MAPPING TEMPLATE CREATED")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Questions: {len(verified['records'])}")
    print(f"Unique IDs: {len(set(ids))}")
    print("Claims inferred automatically: False")
    print("Evidence mappings inferred automatically: False")
    print("Main annotation JSON modified: False")
    print("Pilot and bundles modified: False")
    print("Validation: PASSED")


if __name__ == "__main__":
    main()
