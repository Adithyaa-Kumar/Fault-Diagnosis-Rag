
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "multihoprag" / "MultiHopRAG.json"
OUTPUT = (
    ROOT / "data" / "processed"
    / "multihoprag_evidence_backed_audit.json"
)


def main():
    with INPUT.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    evidence_backed = [
        (i, q) for i, q in enumerate(questions)
        if isinstance(q.get("evidence_list"), list)
        and len(q["evidence_list"]) > 0
    ]

    evidence_counts = Counter()
    publisher_counts = Counter()
    domain_counts = Counter()
    question_types = Counter()
    evidence_pair_counts = Counter()

    examples = []

    for index, q in evidence_backed:
        evidence = q["evidence_list"]
        evidence_counts[len(evidence)] += 1
        question_types[str(q.get("question_type", "missing"))] += 1

        publishers = {
            str(item.get("source", "")).strip()
            for item in evidence
            if item.get("source")
        }
        publisher_counts[len(publishers)] += 1

        domains = set()
        for item in evidence:
            url = str(item.get("url", "")).strip()
            if url:
                host = url.split("//")[-1].split("/")[0].lower()
                host = host.removeprefix("www.")
                domains.add(host)

        domain_counts[len(domains)] += 1

        facts = [
            str(item.get("fact", "")).strip()
            for item in evidence
            if item.get("fact")
        ]
        normalized_facts = {
            " ".join(fact.lower().split()) for fact in facts
        }
        duplicate_fact_count = len(facts) - len(normalized_facts)
        evidence_pair_counts[
            "has_exact_duplicate_fact" if duplicate_fact_count > 0
            else "no_exact_duplicate_fact"
        ] += 1

        if len(examples) < 15:
            examples.append({
                "original_index": index,
                "query": q.get("query", ""),
                "answer": q.get("answer", ""),
                "question_type": q.get("question_type", ""),
                "evidence_count": len(evidence),
                "publishers": sorted(publishers),
                "domains": sorted(domains),
                "exact_duplicate_fact_count": duplicate_fact_count,
            })

    summary = {
        "raw_question_count": len(questions),
        "evidence_backed_question_count": len(evidence_backed),
        "empty_evidence_question_count": (
            len(questions) - len(evidence_backed)
        ),
        "evidence_items_per_question": dict(
            sorted(evidence_counts.items())
        ),
        "distinct_publishers_per_question": dict(
            sorted(publisher_counts.items())
        ),
        "distinct_domains_per_question": dict(
            sorted(domain_counts.items())
        ),
        "question_types": dict(question_types),
        "exact_duplicate_fact_screen": dict(evidence_pair_counts),
        "sample_questions": examples,
        "limitations": [
            "Different publishers or domains do not prove independence.",
            "Exact duplicate facts detect only literal repetition.",
            "No provenance clusters or experimental conditions are assigned.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== EVIDENCE-BACKED QUESTION AUDIT ===")
    print("Raw questions:", len(questions))
    print("Evidence-backed questions:", len(evidence_backed))
    print("Empty-evidence questions:", len(questions) - len(evidence_backed))
    print("Evidence items per question:", dict(sorted(evidence_counts.items())))
    print("Publishers per question:", dict(sorted(publisher_counts.items())))
    print("Domains per question:", dict(sorted(domain_counts.items())))
    print("Question types:", dict(question_types))
    print("Exact-duplicate screen:", dict(evidence_pair_counts))
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
