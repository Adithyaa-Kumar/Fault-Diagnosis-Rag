import json
from pathlib import Path
from src.evaluation.prepare_balanced_design import build_design_audit


def make_bundle(counts):
    factors = {
        "A_lowC_lowP": ("low", "low"),
        "B_highC_lowP": ("high", "low"),
        "C_lowC_highP": ("low", "high"),
        "D_highC_highP": ("high", "high"),
    }
    q = {"sample_id": "q1", "source_evidence": [{"fact": "x"}], "conditions": {}}
    for name, (c, p) in factors.items():
        q["conditions"][name] = {"redundancy": c, "dependence": p, "documents": [{"doc_id": f"{name}-{i}"} for i in range(counts[name])]}
    return {"questions": [q]}


def test_preflight_preserves_input_and_refuses_to_claim_candidate(tmp_path):
    source = tmp_path / "bundles.json"
    source.write_text(json.dumps(make_bundle({"A_lowC_lowP": 1, "B_highC_lowP": 2, "C_lowC_highP": 1, "D_highC_highP": 2})), encoding="utf-8")
    before = source.read_bytes()
    result = build_design_audit(source, tmp_path / "out")
    assert source.read_bytes() == before
    assert result["candidate_bundle_created"] is False
    assert result["gate_decision"] == "NOT CLEARED"
    assert not result["document_counts_equal_within_every_sample"]
    assert (tmp_path / "out" / "balanced_design_preflight.md").exists()


def test_equal_counts_do_not_clear_semantic_gate(tmp_path):
    source = tmp_path / "bundles.json"
    source.write_text(json.dumps(make_bundle({"A_lowC_lowP": 2, "B_highC_lowP": 2, "C_lowC_highP": 2, "D_highC_highP": 2})), encoding="utf-8")
    result = build_design_audit(source, tmp_path / "out")
    assert result["document_counts_equal_within_every_sample"]
    assert result["gate_decision"] == "NOT CLEARED"
    assert result["candidate_bundle_created"] is False
