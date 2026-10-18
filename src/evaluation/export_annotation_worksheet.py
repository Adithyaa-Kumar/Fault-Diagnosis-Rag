
import csv
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
    / "annotation_review_worksheet.csv"
)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("Expected an annotations array.")

    # Export only the remaining questions without draft claims.
    remaining = [
        a for a in annotations
        if not a.get("required_claims")
    ]

    if len(remaining) != 27:
        raise ValueError(
            f"Expected 27 unannotated questions, found {len(remaining)}. "
            "No worksheet was written; inspect the annotation JSON."
        )

    columns = [
        "sample_id",
        "question_type",
        "query",
        "evidence_id",
        "source",
        "title",
        "url",
        "evidence_fact",
        "claim_id",
        "atomic_claim",
        "support_label",
        "rationale",
        "reviewer_notes",
    ]

    rows = []

    for annotation in remaining:
        sample_id = annotation["sample_id"]
        evidence_items = annotation.get("evidence_items", [])

        # Preserve a question even if it has no evidence items.
        if not evidence_items:
            evidence_items = [{}]

        for evidence in evidence_items:
            rows.append({
                "sample_id": sample_id,
                "question_type": annotation.get("question_type", ""),
                "query": annotation.get("query", ""),
                "evidence_id": evidence.get("evidence_id", ""),
                "source": evidence.get("source") or "",
                "title": evidence.get("title") or "",
                "url": evidence.get("url") or "",
                "evidence_fact": evidence.get("fact", ""),
                "claim_id": "",
                "atomic_claim": "",
                "support_label": "",
                "rationale": "",
                "reviewer_notes": "",
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("\nANNOTATION REVIEW WORKSHEET CREATED")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Questions: {len(remaining)}")
    print(f"Evidence rows: {len(rows)}")
    print("Gold answers included: False")
    print("Original annotation JSON modified: False")
    print("Pilot and controlled bundles modified: False")
    print("\nAllowed support labels:")
    print("supports | contradicts | insufficient | unrelated")


if __name__ == "__main__":
    main()
