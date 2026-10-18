
import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

PILOT_PATH = (
    BASE_DIR / "data" / "processed" / "multihoprag" / "pilot_30.json"
)
OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "multihoprag"
    / "claim_annotations_30.json"
)

VALID_LABELS = {
    "supports",
    "contradicts",
    "insufficient",
    "unrelated",
}

# Preserve these records exactly as they currently exist.
PRESERVE_IDS = {"mh_0174", "mh_0533", "mh_0639"}


def first_value(obj, keys, default=None):
    """Return the first non-empty value among possible field names."""
    if not isinstance(obj, dict):
        return default

    for key in keys:
        value = obj.get(key)
        if value is not None and value != "":
            return value

    return default


def get_sample_id(record):
    sample_id = first_value(
        record,
        ["sample_id", "id", "question_id"],
    )

    if not sample_id:
        original_index = first_value(
            record, ["original_index", "index"]
        )
        if original_index is not None:
            sample_id = f"mh_{int(original_index):04d}"

    if not sample_id:
        raise ValueError(
            "A pilot record has no sample ID or usable original index. "
            f"Available keys: {list(record.keys())}"
        )

    return str(sample_id)


def normalize_evidence(sample_id, index, item):
    """Convert a pilot evidence item into the annotation schema."""
    evidence_id = f"{sample_id}_ev_{index:02d}"

    if isinstance(item, str):
        return {
            "evidence_id": evidence_id,
            "source": None,
            "title": None,
            "url": None,
            "fact": item,
        }

    if not isinstance(item, dict):
        raise ValueError(
            f"{sample_id}: evidence item {index} is not an object "
            "or string. Inspect the pilot data before proceeding."
        )

    # Support common field names without inventing missing metadata.
    fact = first_value(
        item,
        [
            "fact",
            "evidence",
            "text",
            "content",
            "snippet",
            "body",
            "sentence",
        ],
    )

    # Some datasets nest article metadata or evidence under another key.
    nested = first_value(
        item,
        ["article", "document", "metadata"],
        default={},
    )

    if not isinstance(nested, dict):
        nested = {}

    if fact is None:
        fact = first_value(
            nested,
            ["fact", "evidence", "text", "content", "body"],
        )

    if fact is None:
        raise ValueError(
            f"{sample_id}: cannot identify the evidence text for "
            f"item {index}. Available keys: {list(item.keys())}. "
            "Inspect this record; do not substitute the gold answer."
        )

    source = first_value(
        item, ["source", "publisher", "website", "domain"]
    )
    title = first_value(
        item, ["title", "article_title", "document_title"]
    )
    url = first_value(item, ["url", "article_url", "source_url"])

    if source is None:
        source = first_value(
            nested, ["source", "publisher", "website", "domain"]
        )
    if title is None:
        title = first_value(
            nested, ["title", "article_title", "document_title"]
        )
    if url is None:
        url = first_value(
            nested, ["url", "article_url", "source_url"]
        )

    # Retain the original fields for traceability.
    result = dict(item)
    result.update({
        "evidence_id": evidence_id,
        "source": source,
        "title": title,
        "url": url,
        "fact": fact,
    })

    return result


def validate_record(annotation):
    sample_id = annotation["sample_id"]
    claims = annotation.get("required_claims", [])
    evidence_items = annotation.get("evidence_items", [])
    mappings = annotation.get("evidence_support", [])

    claim_ids = [c.get("claim_id") for c in claims]
    evidence_ids = [e.get("evidence_id") for e in evidence_items]

    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError(f"{sample_id}: duplicate claim IDs.")

    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError(f"{sample_id}: duplicate evidence IDs.")

    known_claims = set(claim_ids)
    known_evidence = set(evidence_ids)

    for mapping in mappings:
        if mapping.get("claim_id") not in known_claims:
            raise ValueError(
                f"{sample_id}: mapping has an unknown claim ID."
            )
        if mapping.get("evidence_id") not in known_evidence:
            raise ValueError(
                f"{sample_id}: mapping has an unknown evidence ID."
            )
        if mapping.get("label") not in VALID_LABELS:
            raise ValueError(
                f"{sample_id}: invalid support label."
            )

    if annotation.get("annotation_status") != "pending":
        raise ValueError(
            f"{sample_id}: new annotation status must be pending."
        )


def main():
    if not PILOT_PATH.exists():
        raise FileNotFoundError(f"Pilot file not found: {PILOT_PATH}")

    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Existing annotation file not found: {OUTPUT_PATH}"
        )

    with PILOT_PATH.open("r", encoding="utf-8") as f:
        pilot_data = json.load(f)

    with OUTPUT_PATH.open("r", encoding="utf-8") as f:
        old_data = json.load(f)

    # Accept either a list or a dictionary containing the pilot records.
    if isinstance(pilot_data, list):
        pilot_records = pilot_data
    elif isinstance(pilot_data, dict):
        pilot_records = first_value(
            pilot_data, ["questions", "samples", "records", "data"]
        )
    else:
        pilot_records = None

    if not isinstance(pilot_records, list):
        raise ValueError(
            "Could not locate the pilot record list in pilot_30.json. "
            "Inspect its top-level JSON structure before continuing."
        )

    if len(pilot_records) != 30:
        raise ValueError(
            f"Expected 30 pilot records, found {len(pilot_records)}. "
            "No annotation file has been changed."
        )

    old_annotations = old_data.get("annotations")
    if not isinstance(old_annotations, list):
        raise ValueError("Existing annotations must be a JSON array.")

    old_by_id = {}
    for annotation in old_annotations:
        sample_id = annotation.get("sample_id")
        if not sample_id or sample_id in old_by_id:
            raise ValueError(
                "Existing annotation file has a missing or duplicate ID."
            )
        old_by_id[sample_id] = annotation

    if not PRESERVE_IDS.issubset(old_by_id):
        raise ValueError(
            "One or more of the three existing pilot annotations "
            "are missing. Refusing to rebuild the file."
        )

    output_annotations = []
    seen_ids = set()

    for record in pilot_records:
        if not isinstance(record, dict):
            raise ValueError("A pilot record is not a JSON object.")

        sample_id = get_sample_id(record)

        if sample_id in seen_ids:
            raise ValueError(f"Duplicate pilot sample ID: {sample_id}")
        seen_ids.add(sample_id)

        # Keep the three existing draft annotations and mappings intact.
        if sample_id in PRESERVE_IDS:
            output_annotations.append(old_by_id[sample_id])
            continue

        query = first_value(record, ["query", "question"])
        answer = first_value(record, ["answer", "gold_answer"])
        question_type = first_value(
            record, ["question_type", "type"]
        )

        if query is None or answer is None:
            raise ValueError(
                f"{sample_id}: missing query or gold answer. "
                "No annotation file has been changed."
            )

        raw_evidence = first_value(
            record,
            ["evidence_list", "evidence_items", "evidence"],
            default=[],
        )

        if raw_evidence is None:
            raw_evidence = []

        if not isinstance(raw_evidence, list):
            raise ValueError(
                f"{sample_id}: evidence field is not a list. "
                "Inspect the pilot record before continuing."
            )

        evidence_items = [
            normalize_evidence(sample_id, i, item)
            for i, item in enumerate(raw_evidence)
        ]

        annotation = {
            "sample_id": sample_id,
            "question_type": question_type,
            "query": query,
            "gold_answer": answer,
            "required_claims": [],
            "evidence_items": evidence_items,
            "evidence_support": [],
            "annotation_status": "pending",
            "review_notes": (
                "Pending annotation. Decompose the question into "
                "atomic claims and map evidence to each claim. "
                "Do not infer support from the gold answer, shared "
                "keywords, or publisher identity."
            ),
        }

        validate_record(annotation)
        output_annotations.append(annotation)

    if len(output_annotations) != 30:
        raise ValueError(
            f"Expected 30 annotations, found {len(output_annotations)}."
        )

    output_ids = [a["sample_id"] for a in output_annotations]
    if len(output_ids) != len(set(output_ids)):
        raise ValueError("Duplicate sample IDs in output.")

    # Check the preserved records were not altered.
    for sample_id in PRESERVE_IDS:
        original = old_by_id[sample_id]
        preserved = next(
            a for a in output_annotations
            if a["sample_id"] == sample_id
        )
        if original != preserved:
            raise ValueError(
                f"{sample_id}: existing annotation changed unexpectedly."
            )

    # Build output in memory and validate before touching the file.
    result = dict(old_data)
    result["dataset"] = "MultiHop-RAG"
    result["artifact_type"] = "manual_claim_annotation_template"
    result["annotations"] = output_annotations

    # Write a temporary file and verify it parses before replacing output.
    temp_path = OUTPUT_PATH.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    with temp_path.open("r", encoding="utf-8") as f:
        verified = json.load(f)

    if len(verified["annotations"]) != 30:
        temp_path.unlink(missing_ok=True)
        raise ValueError("Written-file verification failed.")

    # Back up the current annotation file, then replace it.
    backup_path = OUTPUT_PATH.with_suffix(".json.bak")
    shutil.copy2(OUTPUT_PATH, backup_path)
    temp_path.replace(OUTPUT_PATH)

    pending_count = sum(
        a.get("annotation_status") == "pending"
        for a in output_annotations
    )

    print("\nANNOTATION SET EXPANDED")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Backup: {backup_path}")
    print(f"Pilot questions: {len(pilot_records)}")
    print(f"Annotation records: {len(output_annotations)}")
    print(f"Pending records: {pending_count}")
    print("Existing three draft annotations preserved: True")
    print("New records contain inferred claims: False")
    print("Raw dataset, pilot, and bundles modified: False")
    print("Validation: PASSED")
    print(
        "\nThe remaining records are templates only. "
        "They must be annotated before research evaluation."
    )


if __name__ == "__main__":
    main()
