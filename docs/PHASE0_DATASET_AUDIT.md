# Phase 0 offline dataset audit

Apply this patch by extracting it into the existing repository root and merging files. It adds `src/evaluation/validate_multihop_evidence.py`.

Run `python -m src.evaluation.validate_multihop_evidence`. The command reads `data/processed/multihoprag/{pilot_30,claim_annotations_30,controlled_bundles_30}.json` and `results/multihoprag/evidence_review_v2.csv`; it writes only `results/multihoprag/phase0_dataset_validation.md`. It does not overwrite or edit any input.

The audit checks ID/row consistency, compares draft labels to canonical annotation pairs, reports pending review decisions, identifies samples with draft unsupported/contradictory claims, and summarizes synthetic/unknown provenance and condition document-count risks. It does **not** validate article contents or gold answers; it explicitly leaves semantic validation unperformed and does not clear the real-factorial gate.
