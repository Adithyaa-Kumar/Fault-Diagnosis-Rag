"""Build a human-review CSV for the 30-question MultiHop-RAG pilot.

This utility does not decide whether annotations are correct. Existing support
labels are copied as DRAFT context; reviewer fields are blank by design.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

DEFAULT_BASE = Path("data/processed/multihoprag")
FIELDS = [
    "sample_id", "question_type", "query", "gold_answer",
    "claim_id", "atomic_claim", "draft_support_status", "draft_review_rationale",
    "evidence_id", "source", "title", "author", "published_at", "url", "evidence_fact",
    "review_claim_support", "review_answer_entailment", "review_source_context",
    "reviewer_decision", "reviewer_notes",
]

def build_review_rows(pilot: dict, annotation_doc: dict) -> list[dict[str, str]]:
    pilot_by_id = {q["sample_id"]: q for q in pilot.get("questions", [])}
    rows: list[dict[str, str]] = []
    for ann in annotation_doc.get("annotations", []):
        sample_id = ann.get("sample_id", "")
        q = pilot_by_id.get(sample_id, {})
        evidence_by_id = {e.get("evidence_id", ""): e for e in ann.get("evidence_items", [])}
        claim_by_id = {c.get("claim_id", ""): c for c in ann.get("required_claims", [])}
        support = ann.get("evidence_support", [])
        # Preserve the full claim/evidence cross-product so unsupported claims
        # are reviewable against every evidence item, not only pre-annotated pairs.
        # For samples whose atomic claims have not yet been drafted, still emit
        # one row per evidence item so no pilot question disappears from review.
        claim_pairs = list(claim_by_id.items()) or [("", {"claim": ""})]
        for claim_id, claim in claim_pairs:
            for evidence_id, evidence in evidence_by_id.items():
                draft = next((x for x in support if x.get("claim_id") == claim_id and x.get("evidence_id") == evidence_id), {})
                rows.append({
                    "sample_id": sample_id,
                    "question_type": ann.get("question_type", q.get("question_type", "")),
                    "query": ann.get("query", q.get("query", "")),
                    "gold_answer": ann.get("gold_answer", q.get("answer", "")),
                    "claim_id": claim_id,
                    "atomic_claim": claim.get("claim", ""),
                    "draft_support_status": draft.get("label", "no_draft_claims" if not claim_by_id else "not_annotated"),
                    "draft_review_rationale": draft.get("rationale", ""),
                    "evidence_id": evidence_id,
                    "source": evidence.get("source", ""),
                    "title": evidence.get("title", ""),
                    "author": evidence.get("author", ""),
                    "published_at": evidence.get("published_at", ""),
                    "url": evidence.get("url", ""),
                    "evidence_fact": evidence.get("fact", ""),
                    "review_claim_support": "",
                    "review_answer_entailment": "",
                    "review_source_context": "",
                    "reviewer_decision": "",
                    "reviewer_notes": "",
                })
    return rows

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE, help="Directory containing pilot_30.json and claim_annotations_30.json")
    parser.add_argument("--output", type=Path, default=Path("results/multihoprag/evidence_review_v2.csv"))
    args = parser.parse_args()
    pilot_path = args.base_dir / "pilot_30.json"
    annotation_path = args.base_dir / "claim_annotations_30.json"
    if not pilot_path.exists() or not annotation_path.exists():
        parser.error(f"Expected {pilot_path} and {annotation_path}; no files were changed.")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    rows = build_review_rows(pilot, annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    samples = len({r["sample_id"] for r in rows})
    print(f"Wrote {len(rows)} claim-evidence review rows across {samples} samples: {args.output}")
    print("Draft support labels are context only. Reviewer fields are blank; no labels were auto-approved.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
