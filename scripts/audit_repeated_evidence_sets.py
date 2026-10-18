
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "multihoprag" / "MultiHopRAG.json"
OUTPUT = ROOT / "data" / "processed" / "multihoprag_repeated_evidence_sets.json"


def evidence_signature(evidence_list):
    """Create a stable signature from evidence URLs and fact text."""
    entries = []

    for item in evidence_list or []:
        url = " ".join(str(item.get("url", "")).split()).strip().lower()
        fact = " ".join(str(item.get("fact", "")).split()).strip().lower()
        entries.append((url, fact))

    # Sorting makes the signature insensitive to evidence order.
    serialized = json.dumps(sorted(entries), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def main():
    with INPUT.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    groups = defaultdict(list)
    skipped_empty = 0

    for index, question in enumerate(questions):
        evidence = question.get("evidence_list") or []

        if not evidence:
            skipped_empty += 1
            continue

        question_id = str(
            question.get("_id", question.get("id", index))
        )

        groups[evidence_signature(evidence)].append({
            "question_id": question_id,
            "question_index": index,
            "query": question.get("query", ""),
            "answer": question.get("answer", ""),
            "question_type": question.get("question_type", ""),
            "evidence_count": len(evidence),
        })

    repeated_groups = [
        {
            "group_id": signature,
            "question_count": len(members),
            "members": members,
        }
        for signature, members in groups.items()
        if len(members) > 1
    ]

    repeated_groups.sort(
        key=lambda group: group["question_count"],
        reverse=True,
    )

    questions_in_repeated_groups = sum(
        group["question_count"] for group in repeated_groups
    )

    summary = {
        "raw_question_count": len(questions),
        "questions_with_evidence": len(questions) - skipped_empty,
        "questions_without_evidence": skipped_empty,
        "unique_evidence_set_count": len(groups),
        "repeated_evidence_set_group_count": len(repeated_groups),
        "questions_in_repeated_evidence_groups": questions_in_repeated_groups,
        "questions_in_singleton_groups": (
            len(questions) - skipped_empty - questions_in_repeated_groups
        ),
        "repeated_groups": repeated_groups,
        "interpretation": (
            "Identical URL-and-fact sets indicate reused evidence records, "
            "not necessarily duplicate questions or independent samples."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Repeated evidence-set audit completed.")
    print("Questions with evidence:", summary["questions_with_evidence"])
    print("Unique evidence sets:", summary["unique_evidence_set_count"])
    print("Repeated groups:", summary["repeated_evidence_set_group_count"])
    print(
        "Questions in repeated groups:",
        summary["questions_in_repeated_evidence_groups"],
    )
    print("Results:", OUTPUT)

    print("\nLargest repeated groups:")
    for group in repeated_groups[:10]:
        ids = [m["question_id"] for m in group["members"]]
        print(
            f"  {group['question_count']} questions: "
            + ", ".join(ids)
        )


if __name__ == "__main__":
    main()
    