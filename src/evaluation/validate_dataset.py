
import json
import sys
from pathlib import Path


EXPECTED_CONDITIONS = {
    "A_lowC_lowP": (False, False),
    "B_highC_lowP": (True, False),
    "C_lowC_highP": (False, True),
    "D_highC_highP": (True, True),
}


def validate_dataset(path: str = "data/seed_sample.json") -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = []

    if not data.get("query_id"):
        errors.append("Missing query_id.")
    if not data.get("query"):
        errors.append("Missing query.")
    if not data.get("gold_answer"):
        errors.append("Missing gold_answer.")

    conditions = data.get("conditions", {})

    if set(conditions) != set(EXPECTED_CONDITIONS):
        errors.append(
            f"Expected exactly {sorted(EXPECTED_CONDITIONS)}, "
            f"got {sorted(conditions)}."
        )

    reference_claims = None
    reference_gold = data.get("gold_answer")
    reference_doc_count = None

    for condition, (high_c, high_p) in EXPECTED_CONDITIONS.items():
        docs = conditions.get(condition)
        if not isinstance(docs, list) or not docs:
            errors.append(f"{condition}: missing documents.")
            continue

        if reference_doc_count is None:
            reference_doc_count = len(docs)
        elif len(docs) != reference_doc_count:
            errors.append(f"{condition}: document count differs.")

        doc_ids = [doc.get("doc_id") for doc in docs]
        if len(doc_ids) != len(set(doc_ids)):
            errors.append(f"{condition}: duplicate doc_id.")

        claims = set()
        clusters = set()
        normalized_claim_texts = []

        for doc in docs:
            for field in (
                "doc_id", "content", "source_cluster_id", "claim_id"
            ):
                if not doc.get(field):
                    errors.append(
                        f"{condition}: document missing {field}."
                    )

            claims.add(doc.get("claim_id"))
            clusters.add(doc.get("source_cluster_id"))
            normalized_claim_texts.append(
                " ".join(doc.get("content", "").lower().split())
            )

        if reference_claims is None:
            reference_claims = claims
        elif claims != reference_claims:
            errors.append(
                f"{condition}: claim-ID coverage differs from reference."
            )

        if reference_gold != data.get("gold_answer"):
            errors.append(f"{condition}: gold answer differs.")

        # Operational checks for this deliberately controlled pilot.
        if high_p and len(clusters) != 1:
            errors.append(
                f"{condition}: high provenance dependence must use "
                "one source cluster."
            )

        if not high_p and len(clusters) != len(docs):
            errors.append(
                f"{condition}: low provenance dependence must use "
                "independent source clusters."
            )

        # This pilot defines high C as repeated near-identical passages.
        unique_texts = len(set(normalized_claim_texts))
        if high_c and unique_texts != 1:
            errors.append(
                f"{condition}: high-C documents must have identical "
                "normalized text under this operational definition."
            )

        if not high_c and unique_texts == 1:
            errors.append(
                f"{condition}: low-C documents must use distinct wording."
            )

    if errors:
        print("DATASET VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
    else:
        print("DATASET STRUCTURAL VALIDATION PASSED")
        print("Four conditions, document counts, claim IDs and")
        print("operational C/P settings are structurally consistent.")
        print("Manual semantic review is still required.")

    return errors


if __name__ == "__main__":
    problems = validate_dataset()
    sys.exit(1 if problems else 0)
