import json
from pathlib import Path

EXPECTED = {"A_lowC_lowP", "B_highC_lowP", "C_lowC_highP", "D_highC_highP"}


def validate(path: str = "data/seed_sample.json") -> None:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = []
    if data.get("dataset_type") != "synthetic_controlled_fixture":
        errors.append("The controlled seed fixture must be explicitly labeled synthetic.")
    if not data.get("fictional_notice") or not data.get("bundle_claim_map"):
        errors.append("Synthetic notice and bundle_claim_map are required.")
    conditions = data.get("conditions", {})
    if set(conditions) != EXPECTED:
        errors.append(f"Expected conditions {sorted(EXPECTED)}, got {sorted(conditions)}")
    claims = data.get("required_claims", [])
    claim_ids = [c.get("claim_id") for c in claims]
    if not claims or len(claim_ids) != len(set(claim_ids)):
        errors.append("required_claims must exist and have unique claim_id values")
    for claim in claims:
        if not claim.get("claim_text") or not claim.get("match_terms"):
            errors.append(f"Claim {claim.get('claim_id')} lacks claim_text or match_terms")
    expected_doc_count = None
    for condition, docs in conditions.items():
        if expected_doc_count is None:
            expected_doc_count = len(docs)
        if len(docs) != expected_doc_count:
            errors.append(f"{condition}: expected {expected_doc_count} docs, got {len(docs)}")
        ids = [d.get("doc_id") for d in docs]
        if len(ids) != len(set(ids)):
            errors.append(f"{condition}: duplicate doc_id")
        for doc in docs:
            if doc.get("claim_id") not in set(claim_ids) | {"claim_bundle_all"}:
                errors.append(f"{condition}/{doc.get('doc_id')}: unknown claim_id {doc.get('claim_id')}")
            expected_claim_ids = data.get("bundle_claim_map", {}).get(doc.get("claim_id"), [doc.get("claim_id")])
            if doc.get("claim_ids") != expected_claim_ids:
                errors.append(f"{condition}/{doc.get('doc_id')}: claim_ids inconsistent with claim_id {doc.get('claim_id')}")
            text = doc.get("content", "").lower()
            # Evidence coverage: every required claim must be supported at least once in every condition.
            # High-redundancy conditions intentionally repeat all three claims in all three documents.
        full_text = " ".join(d.get("content", "") for d in docs).lower()
        for claim in claims:
            missing = [term for term in claim["match_terms"] if term.lower() not in full_text]
            if missing:
                errors.append(f"{condition}: missing evidence terms for {claim['claim_id']}: {missing}")
    # Provenance manipulation must be orthogonal to content redundancy.
    for condition in ("A_lowC_lowP", "B_highC_lowP"):
        clusters = {d.get("source_cluster_id") for d in conditions.get(condition, [])}
        if len(clusters) != 3:
            errors.append(f"{condition}: low-provenance-dependence condition must have 3 distinct source clusters")
    for condition in ("C_lowC_highP", "D_highC_highP"):
        clusters = {d.get("source_cluster_id") for d in conditions.get(condition, [])}
        if len(clusters) != 1:
            errors.append(f"{condition}: high-provenance-dependence condition must use exactly 1 source cluster")
    # Explicit factorial design assertions: A/C each contain 1 doc per claim; B/D repeat the bundle.
    for condition in ("A_lowC_lowP", "C_lowC_highP"):
        counts = {cid: sum(d.get("claim_id") == cid for d in conditions[condition]) for cid in claim_ids}
        if any(counts[cid] != 1 for cid in claim_ids):
            errors.append(f"{condition}: low-redundancy condition must contain exactly one document per claim; got {counts}")
    for condition in ("B_highC_lowP", "D_highC_highP"):
        for doc in conditions[condition]:
            content = doc.get("content", "").lower()
            for claim in claims:
                if not all(term.lower() in content for term in claim["match_terms"]):
                    errors.append(f"{condition}/{doc['doc_id']}: high-redundancy document does not repeat {claim['claim_id']}")
    if errors:
        raise SystemExit("DATASET VALIDATION FAILED:\n- " + "\n- ".join(errors))
    print("DATASET VALIDATION PASSED: 4 conditions, 3 documents each, consistent claim coverage, unique claim IDs.")


if __name__ == "__main__":
    validate()
