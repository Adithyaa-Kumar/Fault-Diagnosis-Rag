"""Offline Phase 0 audit for MultiHopRAG evidence and experiment artifacts.

This is a structural/provenance-risk audit, not a semantic fact checker. It never
changes source files or auto-approves annotations.
"""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / "data/processed/multihoprag"
CSV_PATH = ROOT / "results/multihoprag/evidence_review_v2.csv"
OUT = ROOT / "results/multihoprag/phase0_dataset_validation.md"
CONDITIONS = ("A_lowC_lowP", "B_highC_lowP", "C_lowC_highP", "D_highC_highP")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_audit(data_dir: Path = DATA, csv_path: Path = CSV_PATH, out_path: Path = OUT) -> dict:
    pilot = load_json(data_dir / "pilot_30.json")
    annotations = load_json(data_dir / "claim_annotations_30.json")
    bundles = load_json(data_dir / "controlled_bundles_30.json")
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        review_rows = list(reader)
        columns = reader.fieldnames or []

    pilot_by_id = {x["sample_id"]: x for x in pilot.get("questions", [])}
    ann_by_id = {x["sample_id"]: x for x in annotations.get("annotations", [])}
    bundle_by_id = {x["sample_id"]: x for x in bundles.get("questions", [])}
    errors, warnings = [], []

    if len(pilot_by_id) != len(pilot.get("questions", [])):
        errors.append("pilot_30.json has duplicate sample_id values.")
    if set(pilot_by_id) != set(ann_by_id):
        errors.append("Pilot and claim-annotation sample IDs do not match.")
    if set(pilot_by_id) != set(bundle_by_id):
        errors.append("Pilot and controlled-bundle sample IDs do not match.")
    required_cols = {"sample_id", "claim_id", "evidence_id", "gold_answer", "review_claim_support", "review_answer_entailment", "review_source_context", "reviewer_decision", "reviewer_notes"}
    missing = sorted(required_cols - set(columns))
    if missing:
        errors.append("Review CSV missing required columns: " + ", ".join(missing))

    review_ids = {r.get("sample_id", "") for r in review_rows}
    if review_ids != set(pilot_by_id):
        warnings.append(f"Review CSV sample coverage differs: {len(review_ids)} sample IDs vs {len(pilot_by_id)} pilot samples.")
    # The worksheet is a claim-by-evidence cross-product; it intentionally has
    # more rows than the sparse draft evidence_support annotations.

    # Compare draft labels where a canonical pair exists; otherwise validate IDs
    # against the sample's claim/evidence inventories without treating a missing
    # pair as an error (the row is there for human review).
    ann_lookup = {}
    for sample in annotations.get("annotations", []):
        for pair in sample.get("evidence_support", []):
            ann_lookup[(sample["sample_id"], pair.get("claim_id"), pair.get("evidence_id"))] = pair
    mismatches = []
    invalid_refs = []
    for row in review_rows:
        sid = row.get("sample_id")
        key = (sid, row.get("claim_id"), row.get("evidence_id"))
        pair = ann_lookup.get(key)
        sample = ann_by_id.get(sid, {})
        claim_ids = {c.get("claim_id") for c in sample.get("required_claims", [])}
        evidence_ids = {e.get("evidence_id") for e in sample.get("evidence_items", [])}
        if row.get("claim_id") and row.get("claim_id") not in claim_ids:
            invalid_refs.append(f"Unknown claim ID in review row {key}.")
        if row.get("evidence_id") and row.get("evidence_id") not in evidence_ids:
            invalid_refs.append(f"Unknown evidence ID in review row {key}.")
        if pair is not None and row.get("draft_support_status") != pair.get("label"):
            mismatches.append(f"Draft support label differs from source annotation for {key}.")
    if invalid_refs:
        warnings.extend(invalid_refs[:20])
        if len(invalid_refs) > 20:
            warnings.append(f"Additional invalid claim/evidence references omitted: {len(invalid_refs)-20}.")
    if mismatches:
        warnings.extend(mismatches[:20])
        if len(mismatches) > 20:
            warnings.append(f"Additional draft-label mismatches omitted: {len(mismatches)-20}.")

    decisions = Counter((r.get("reviewer_decision") or "<blank>").strip() for r in review_rows)
    statuses = Counter((x.get("annotation_status") or "<blank>") for x in annotations.get("annotations", []))
    if any(k in ("", "<blank>", "revise_pending_verification") for k in decisions):
        warnings.append("Reviewer decisions are blank or pending; no human-validated evidence labels can be claimed.")
    if statuses.get("pending", 0):
        warnings.append(f"Canonical claim annotations still pending: {statuses['pending']} of {len(annotations.get('annotations', []))}.")

    # The gold-answer audit flags possible issues for human review; it does not infer truth.
    suspected = []
    for sample in annotations.get("annotations", []):
        labels = [p.get("label") for p in sample.get("evidence_support", [])]
        if any(label in ("insufficient", "contradicts") for label in labels):
            suspected.append(sample.get("sample_id"))
    warnings.append(f"Gold-answer review required for {len(set(suspected))} samples with at least one draft unsupported/contradictory claim: {', '.join(sorted(set(suspected)))}.")
    if "mh_0174" in pilot_by_id:
        q = pilot_by_id["mh_0174"]
        warnings.append("mh_0174: its first cited excerpt says sportsbooks earn profit built into odds/lines; it does not explicitly establish that the named article states sportsbooks adjust odds. Gold answer 'Yes' requires full-source review.")

    # Check experimental design confounds across each sample's four controlled conditions.
    condition_stats = Counter()
    confound_samples = []
    synthetic_lineage_docs = 0
    total_docs = 0
    unknown_provenance_docs = 0
    for sample in bundles.get("questions", []):
        conds = sample.get("conditions", {})
        counts = {}
        for cond in CONDITIONS:
            condition_data = conds.get(cond, {})
            docs = condition_data.get("documents", []) if isinstance(condition_data, dict) else []
            counts[cond] = len(docs)
            condition_stats[cond] += len(docs)
            total_docs += len(docs)
            for d in docs:
                synthetic_lineage_docs += bool(d.get("lineage_is_synthetic", False))
                unknown_provenance_docs += d.get("original_provenance_status") == "unknown"
        if len(set(counts.values())) > 1:
            confound_samples.append(sample.get("sample_id"))
    if confound_samples:
        warnings.append(f"Document-count confound: condition document counts differ for {len(confound_samples)} samples: {', '.join(confound_samples)}.")
    if total_docs and synthetic_lineage_docs:
        warnings.append(f"Synthetic lineage IDs: {synthetic_lineage_docs}/{total_docs} documents; these encode experimental assignments, not verified real-world provenance.")
    if unknown_provenance_docs:
        warnings.append(f"Original provenance marked unknown for {unknown_provenance_docs}/{total_docs} documents; do not interpret publisher/URL similarity as proof of dependence.")

    report = {
        "pilot_samples": len(pilot_by_id),
        "annotation_samples": len(ann_by_id),
        "bundle_samples": len(bundle_by_id),
        "review_rows": len(review_rows),
        "review_columns": len(columns),
        "review_decisions": dict(decisions),
        "annotation_statuses": dict(statuses),
        "condition_document_totals": dict(condition_stats),
        "documents_across_conditions": total_docs,
        "synthetic_lineage_documents": synthetic_lineage_docs,
        "unknown_provenance_documents": unknown_provenance_docs,
        "sample_document_count_confounds": confound_samples,
        "errors": errors,
        "warnings": warnings,
        "semantic_validation": "NOT_PERFORMED; requires source-level human review",
    }
    lines = ["# Phase 0 MultiHopRAG Dataset Validation", "", "## Scope", "", "Offline structural consistency and experimental-design risk checks only. This report does not verify article contents, factual correctness, or gold-answer validity.", "", "## Counts", "", f"- Pilot samples: {report['pilot_samples']}", f"- Annotation samples: {report['annotation_samples']}", f"- Controlled-bundle samples: {report['bundle_samples']}", f"- Review CSV rows / columns: {report['review_rows']} / {report['review_columns']}", f"- Reviewer decisions: `{json.dumps(report['review_decisions'], sort_keys=True)}`", f"- Canonical annotation statuses: `{json.dumps(report['annotation_statuses'], sort_keys=True)}`", f"- Documents across four conditions: {total_docs}", f"- Documents with synthetic lineage IDs: {synthetic_lineage_docs}", f"- Documents with unknown original provenance: {unknown_provenance_docs}", "", "## Errors", ""]
    lines += [f"- {x}" for x in errors] or ["- None found by structural checks."]
    lines += ["", "## Warnings / Required follow-up", ""]
    lines += [f"- {x}" for x in warnings] or ["- None."]
    lines += ["", "## Gate decision", "", "**NOT CLEARED for the real factorial.** Human source/evidence review and experimental-design correction remain required. A clean structural audit is not evidence that claims or gold answers are true.", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    report = run_audit()
    print("PHASE 0 DATASET AUDIT")
    print(f"Samples: pilot={report['pilot_samples']}, annotations={report['annotation_samples']}, bundles={report['bundle_samples']}")
    print(f"Review CSV: {report['review_rows']} rows x {report['review_columns']} columns")
    print(f"Structural errors: {len(report['errors'])}")
    print(f"Warnings: {len(report['warnings'])}")
    print("Semantic validation: NOT PERFORMED")
    print("Real factorial gate: NOT CLEARED")
    print(f"Report written to {OUT.relative_to(ROOT)}")
    if report['errors']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
