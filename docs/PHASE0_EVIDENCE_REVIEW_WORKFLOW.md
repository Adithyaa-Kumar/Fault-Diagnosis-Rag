# Phase 0 — Evidence review workflow

This utility creates `results/multihoprag/evidence_review_v2.csv` from the existing 30-question pilot and draft claim annotations.

## Run

```powershell
python -m src.evaluation.build_evidence_review
```

It writes one row per **claim × evidence item**. Existing labels/rationales are copied to columns prefixed `draft_`; they are not treated as validated. The human-review columns remain blank.

## Review fields

- `review_claim_support`: `supports`, `contradicts`, `insufficient`, or `not_applicable`.
- `review_answer_entailment`: after considering all required claims and the complete supplied evidence, record `entailed`, `contradicted`, or `undetermined` for the gold answer. Repeat consistently across rows for the sample, or document the sample-level decision in reviewer notes.
- `review_source_context`: `excerpt_only`, `full_source_checked`, or `unavailable`.
- `reviewer_decision`: `accept`, `revise_gold`, `exclude`, or `needs_adjudication`.
- `reviewer_notes`: concise evidence-based explanation; include a checked full-source URL or reason it could not be checked.

Do not infer source dependence from a shared publisher, domain, or URL pattern. Keep original provenance `unknown` unless actual lineage evidence is available. Do not start the real-model factorial until gold/evidence mapping has been reviewed and condition confounds are addressed.
