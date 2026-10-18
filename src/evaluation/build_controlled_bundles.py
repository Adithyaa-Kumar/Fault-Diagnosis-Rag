import copy
import hashlib
import json
from pathlib import Path

INPUT = Path("data/processed/multihoprag/pilot_30.json")
OUTPUT = Path("data/processed/multihoprag/controlled_bundles_30.json")

CONDITIONS = {
    "A_lowC_lowP": {"redundancy": "low", "dependence": "low"},
    "B_highC_lowP": {"redundancy": "high", "dependence": "low"},
    "C_lowC_highP": {"redundancy": "low", "dependence": "high"},
    "D_highC_highP": {"redundancy": "high", "dependence": "high"},
}


def stable_id(prefix, value, length=12):
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def main():
    pilot = json.loads(INPUT.read_text(encoding="utf-8"))
    output_questions = []

    for question in pilot["questions"]:
        qid = question["sample_id"]

        # Preserve source evidence records as immutable input copies.
        source_items = question["evidence_list"]
        evidence_units = []

        for index, item in enumerate(source_items):
            evidence_units.append({
                "evidence_unit_id": f"{qid}_ev_{index:02d}",
                "parent_source_url": item.get("url"),
                "source_publisher": item.get("source"),
                "title": item.get("title"),
                "published_at": item.get("published_at"),
                "fact": item.get("fact", ""),
                "original_provenance_status": "unknown",
            })

        conditions = {}

        for condition, design in CONDITIONS.items():
            high_redundancy = design["redundancy"] == "high"
            high_dependence = design["dependence"] == "high"

            docs = []

            for unit in evidence_units:
                # Lineage is explicitly synthetic experimental metadata.
                if high_dependence:
                    lineage = f"synthetic_shared_parent_{qid}"
                else:
                    lineage = f"synthetic_independent_parent_{unit['evidence_unit_id']}"

                base_doc = {
                    "doc_id": f"{condition}_{unit['evidence_unit_id']}_copy0",
                    "question_id": qid,
                    "evidence_unit_id": unit["evidence_unit_id"],
                    "content": unit["fact"],
                    "parent_source_url": unit["parent_source_url"],
                    "source_publisher": unit["source_publisher"],
                    "source_title": unit["title"],
                    "published_at": unit["published_at"],
                    "original_provenance_status": "unknown",
                    "experimental_lineage_id": lineage,
                    "lineage_is_synthetic": True,
                    "transformation": "original_evidence_fact",
                    "content_sha256": stable_id("sha256", unit["fact"], 64),
                }
                docs.append(base_doc)

                if high_redundancy:
                    duplicate = copy.deepcopy(base_doc)
                    duplicate["doc_id"] = (
                        f"{condition}_{unit['evidence_unit_id']}_copy1"
                    )
                    duplicate["transformation"] = "exact_duplicate_for_control"
                    duplicate["duplicate_of_doc_id"] = base_doc["doc_id"]
                    docs.append(duplicate)

            conditions[condition] = {
                **design,
                "lineage_semantics": (
                    "synthetic_shared_parent"
                    if high_dependence
                    else "synthetic_distinct_parent_assignments"
                ),
                "documents": docs,
            }

        output_questions.append({
            "sample_id": qid,
            "original_index": question["original_index"],
            "question_type": question["question_type"],
            "query": question["query"],
            "answer": question["answer"],
            "source_evidence": source_items,
            "conditions": conditions,
        })

    payload = {
        "dataset": "MultiHop-RAG",
        "artifact_type": "controlled_synthetic_lineage_intervention",
        "sample_size": len(output_questions),
        "conditions": CONDITIONS,
        "important_limitations": [
            "Lineage IDs are synthetic experimental assignments, not verified real-world provenance.",
            "Different original URLs or publishers do not establish independence.",
            "Exact duplicate copies increase redundancy but do not add new independent evidence.",
            "This bundle builder does not establish that each original fact supports the answer.",
            "The resulting bundles require further design validation before model evaluation.",
        ],
        "questions": output_questions,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("CONTROLLED BUNDLES CREATED")
    print("Output:", OUTPUT)
    print("Questions:", len(output_questions))
    for condition in CONDITIONS:
        counts = [
            len(q["conditions"][condition]["documents"])
            for q in output_questions
        ]
        print(
            f"{condition}: {len(counts)} questions; "
            f"documents/question range={min(counts)}–{max(counts)}"
        )
    print("Raw source files modified: False")


if __name__ == "__main__":
    main()
