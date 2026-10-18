from pathlib import Path
import json

from src.evaluation.validate_multihop_evidence import run_audit


def test_audit_does_not_modify_inputs(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pilot = {"questions": [{"sample_id": "s1", "answer": "yes"}]}
    annotations = {"annotations": [{"sample_id": "s1", "annotation_status": "pending", "evidence_support": [{"claim_id": "C1", "evidence_id": "e1", "label": "insufficient"}]}]}
    bundles = {"questions": [{"sample_id": "s1", "conditions": {c: [{"lineage_is_synthetic": True, "original_provenance_status": "unknown"}] for c in ("A_lowC_lowP", "B_highC_lowP", "C_lowC_highP", "D_highC_highP")}}]}
    for name, obj in (("pilot_30.json", pilot), ("claim_annotations_30.json", annotations), ("controlled_bundles_30.json", bundles)):
        (data_dir / name).write_text(json.dumps(obj), encoding="utf-8")
    csv_path = tmp_path / "review.csv"
    csv_path.write_text("sample_id,claim_id,evidence_id,gold_answer,draft_support_status,review_claim_support,review_answer_entailment,review_source_context,reviewer_decision,reviewer_notes\ns1,C1,e1,yes,insufficient,,,,revise_pending_verification,\n", encoding="utf-8")
    before = csv_path.read_bytes()
    report = run_audit(data_dir, csv_path, tmp_path / "out.md")
    assert report["review_rows"] == 1
    assert report["semantic_validation"].startswith("NOT_PERFORMED")
    assert csv_path.read_bytes() == before
    assert (tmp_path / "out.md").exists()
