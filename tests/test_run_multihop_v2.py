import json
import re

import pytest

import src.evaluation.run_multihop_v2 as rv
from src.schema import FaultType
from test_diagnosis_isolation import mini_question  # sibling test module (rootdir import mode)


class FakeModel:
    """Grounded specialist; arbiter answer decided by `rule(context)`."""
    model_id = "fake-model"

    def __init__(self, rule=lambda ctx: "Yes", tokens=True, fail_on=None):
        self.rule, self.tokens, self.fail_on, self.calls = rule, tokens, fail_on, []

    def generate_answer(self, context, question, system_prompt, max_new_tokens=None):
        self.calls.append((system_prompt, context))
        if self.fail_on and self.fail_on(len(self.calls), context):
            raise RuntimeError("CUDA out of memory (simulated)")
        if "specialist in a parallel" in system_prompt:
            m = re.search(r"Evidence text: (.+)", context)
            answer = ("Facts: " + m.group(1).split(". ")[0]) if m else "No relevant facts."
        else:
            answer = f"Reasoning.\nAnswer: {self.rule(context)}"
        out = {"answer": answer, "model_id": self.model_id}
        if self.tokens:
            out.update(prompt_tokens=len(context.split()), generated_tokens=len(answer.split()))
        return out


@pytest.fixture()
def payload(tmp_path):
    q = mini_question(tmp_path)
    return {"dataset": "t", "artifact_type": "t", "questions": [q]}


def test_rendered_context_hides_condition_role_and_lineage(payload):
    q = payload["questions"][0]
    for cond in rv.CONDITIONS:
        for d in q["conditions"][cond]["documents"]:
            text = rv.render_doc(d)
            assert d["doc_id"] not in text
            for bad in ("lowC", "highC", "lowP", "highP", "fill", "pool", "synthetic",
                        "lineage", "_c0", "_c1", "http"):
                assert bad not in text, (cond, bad)


def test_trajectory_uses_canonical_scorer_and_records_everything(payload, monkeypatch):
    seen = []
    real = rv.score_multihop_answer
    monkeypatch.setattr(rv, "score_multihop_answer", lambda o, g: (seen.append((o, g)), real(o, g))[1])
    q = payload["questions"][0]
    rec = rv.run_trajectory(q, "A_lowC_lowP", FaultType.NONE, FakeModel())
    assert seen == [(rec["raw_final_output"], "Yes")]
    assert rec["status"] == "ok" and rec["scoring"]["correct"] is True
    assert rec["answer_status"] == "answered"
    assert len(rec["stage_records"]) == 4
    assert all("prompt" in s and "raw_output" in s and s["latency_seconds_measured"] >= 0
               for s in rec["stage_records"])
    assert rec["prompt_tokens_measured"] == sum(s["prompt_tokens_measured"] for s in rec["stage_records"])
    assert rec["injected_layer"] == "none" and rec["injection"]["layer"] == "none"
    assert "injected_layer" not in rec["observation"]


def test_missing_token_counts_stay_null_not_estimated(payload):
    rec = rv.run_trajectory(payload["questions"][0], "A_lowC_lowP", FaultType.NONE,
                            FakeModel(tokens=False))
    assert rec["prompt_tokens_measured"] is None and rec["generated_tokens_measured"] is None


def test_model_exception_is_recorded_not_raised(payload):
    model = FakeModel(fail_on=lambda n, ctx: n == 2)
    rec = rv.run_trajectory(payload["questions"][0], "A_lowC_lowP", FaultType.NONE, model)
    assert rec["status"] == "error" and "out of memory" in rec["error"]
    assert "scoring" not in rec
    assert "error" in rec["stage_records"][-1]


def test_interventions_structure_and_stable_assignment(payload):
    q = payload["questions"][0]
    rep = rv.run_interventions(payload, "q1", "D_highC_highP", FakeModel())
    docs = q["conditions"]["D_highC_highP"]["documents"]
    kinds = [t["intervention"]["kind"] for t in rep["trials"]]
    assert kinds == ["baseline"] + ["item_removal"] * len(docs) + ["claim_family_removal"]
    loo_removed = [t["intervention"]["removed_doc_ids"][0] for t in rep["trials"][1:-1]]
    assert loo_removed == [d["doc_id"] for d in docs]
    fam = rep["trials"][-1]["intervention"]["removed_doc_ids"]
    assert len(fam) == 2
    base_ids = rep["trials"][0]["input_doc_ids"]
    assert rep["trials"][-1]["input_doc_ids"] == [d for d in base_ids if d not in fam]
    # remaining documents keep their specialist after removal
    base_obs = rep["trials"][0]["observation"]["specialists"]
    for t in rep["trials"][1:]:
        for b, s in zip(base_obs, t["observation"]["specialists"]):
            assert set(s["retrieved_doc_ids"]) == set(b["retrieved_doc_ids"]) - set(t["intervention"]["removed_doc_ids"])


def test_cluster_guard_rejects_single_document_families(payload):
    with pytest.raises(ValueError):
        rv.run_interventions(payload, "q1", "A_lowC_lowP", FakeModel())
    with pytest.raises(ValueError):
        rv.run_interventions(payload, "q1", "D_highC_highP", FakeModel(), cluster_unit="nope")


def test_repair_requires_failed_baseline_and_denominator(payload):
    q = payload["questions"][0]
    docs = q["conditions"]["D_highC_highP"]["documents"]
    target_text = docs[0]["content"].split(". ")[0]
    # Fails while the target document's text reaches the arbiter; correct once removed.
    rule = lambda ctx: "No" if target_text in ctx else "Yes"

    rep = rv.run_interventions(payload, "q1", "D_highC_highP", FakeModel(rule))
    assert rep["baseline_eligible_for_repair"] is True
    repairing = [t for t in rep["trials"][1:] if t["is_repair"]]
    assert repairing, "removing the offending document should repair"
    assert all(t["scoring"]["correct"] for t in repairing)
    s = rep["repair_summary"]
    assert s["eligible_failed_baselines"] == 1 and s["repair_success"] == 1.0

    ok = rv.run_interventions(payload, "q1", "D_highC_highP", FakeModel())
    assert ok["baseline_eligible_for_repair"] is False
    assert not any(t["is_repair"] for t in ok["trials"][1:])
    assert ok["repair_summary"]["repair_success"] is None  # undefined, not zero


def test_factorial_exact_coverage_and_resume(payload, tmp_path):
    out = tmp_path / "f.jsonl"
    model = FakeModel()
    cov = rv.run_factorial(payload, ["q1"], model, out)
    assert cov["expected_cells"] == 16 and cov["recorded_cells"] == 16
    assert cov["exact_coverage"] and cov["all_ok"]
    calls = len(model.calls)
    cov2 = rv.run_factorial(payload, ["q1"], model, out)  # resume: nothing re-run
    assert len(model.calls) == calls and cov2["recorded_cells"] == 16
    recs = rv.read_records(out)
    assert {r["injected_layer"] for r in recs} == {"none", "retrieval", "specialist", "arbiter"}
    assert all(r["diagnosis_correct"] for r in recs)


def test_factorial_resume_after_partial_run_fills_only_missing(payload, tmp_path):
    out = tmp_path / "f.jsonl"
    recs = []
    rv.run_factorial(payload, ["q1"], FakeModel(), out)
    lines = out.read_text(encoding="utf-8").splitlines()
    out.write_text("\n".join(lines[:5]) + "\n", encoding="utf-8")
    cov = rv.coverage_report(["q1"], rv.read_records(out))
    assert not cov["exact_coverage"] and len(cov["missing"]) == 11
    cov = rv.run_factorial(payload, ["q1"], FakeModel(), out)
    assert cov["exact_coverage"]


def test_coverage_detects_missing_duplicate_unexpected_and_errors(payload, tmp_path):
    out = tmp_path / "f.jsonl"
    rv.run_factorial(payload, ["q1"], FakeModel(), out)
    recs = rv.read_records(out)
    assert rv.coverage_report(["q1"], recs[:-1])["missing"]
    assert rv.coverage_report(["q1"], recs + [recs[0]])["duplicates"]
    assert rv.coverage_report(["q1", "q2"], recs)["missing"]
    bad = [dict(r) for r in recs]
    bad[3]["status"] = "error"
    rep = rv.coverage_report(["q1"], bad)
    assert rep["exact_coverage"] and not rep["all_ok"] and len(rep["error_cells"]) == 1


def test_factorial_records_errors_and_still_covers_cell(payload, tmp_path):
    out = tmp_path / "f.jsonl"
    model = FakeModel(fail_on=lambda n, ctx: n == 3)
    cov = rv.run_factorial(payload, ["q1"], model, out)
    assert cov["exact_coverage"] and len(cov["error_cells"]) == 1 and not cov["all_ok"]


def test_prompt_manifest_is_stable_and_hashed():
    a, b = rv.prompt_manifest(), rv.prompt_manifest()
    assert a == b and len(a["sha256"]) == 64 and a["version"] == rv.PROMPT_VERSION


def test_unreviewed_samples_are_refused_without_flag(tmp_path):
    import argparse
    ns = argparse.Namespace(sample_manifest=None, sample_ids=["q1"], allow_unreviewed=False)
    with pytest.raises(SystemExit):
        rv.resolve_samples(ns)
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"reviewed": False, "sample_ids": ["q1"]}), encoding="utf-8")
    with pytest.raises(SystemExit):
        rv.resolve_samples(argparse.Namespace(sample_manifest=str(m), sample_ids=None, allow_unreviewed=False))
    m.write_text(json.dumps({"reviewed": True, "sample_ids": ["q1"]}), encoding="utf-8")
    ids, meta = rv.resolve_samples(argparse.Namespace(sample_manifest=str(m), sample_ids=None, allow_unreviewed=False))
    assert ids == ["q1"] and meta["reviewed"] is True


def test_none_escape_word_is_not_flagged_as_specialist_fault(payload):
    class NoneSpecialist(FakeModel):
        def generate_answer(self, context, question, system_prompt, max_new_tokens=None):
            r = super().generate_answer(context, question, system_prompt, max_new_tokens)
            if "specialist in a parallel" in system_prompt:
                r["answer"] = "NONE"
            return r
    rec = rv.run_trajectory(payload["questions"][0], "A_lowC_lowP", FaultType.NONE, NoneSpecialist())
    assert rec["predicted_layer"] == "none" and rec["diagnosis_correct"]


def test_baseline_sweep_summary_counts_abstentions(payload):
    class Abstain(FakeModel):
        def generate_answer(self, context, question, system_prompt, max_new_tokens=None):
            r = super().generate_answer(context, question, system_prompt, max_new_tokens)
            r["answer"] = "NONE" if "specialist in a parallel" in system_prompt else "Answer: Insufficient evidence"
            return r
    out = rv.run_baseline_sweep(payload, ["q1"], "D_highC_highP", Abstain())
    s = out["summary"]
    assert s["n"] == 1 and s["abstained"] == 1 and s["correct"] == 0 and s["specialists_with_no_facts"] == 3
    good = rv.run_baseline_sweep(payload, ["q1"], "D_highC_highP", FakeModel())["summary"]
    assert good["answered"] == 1 and good["correct"] == 1 and good["specialists_with_no_facts"] == 0


def test_dev_selection_excludes_review_sample_and_is_deterministic():
    from pathlib import Path
    from src.evaluation import build_phase0_review_sheet as rs
    path = Path("data/processed/multihoprag/controlled_bundles_v2.json")
    if not path.exists():
        pytest.skip("bundle not generated")
    b = json.loads(path.read_text(encoding="utf-8"))
    sample = rs.select_sample(b)
    dev = rs.select_dev(b, sample, 8)
    assert dev == rs.select_dev(b, sample, 8) and len(dev) == 8
    assert not set(dev) & (set(sample) | set(rs.EXCLUDED))


def test_resolve_excluded_falls_back_to_rederived_sample_when_manifest_missing(tmp_path):
    from pathlib import Path
    from src.evaluation import build_phase0_review_sheet as rs
    path = Path("data/processed/multihoprag/controlled_bundles_v2.json")
    if not path.exists():
        pytest.skip("bundle not generated")
    b = json.loads(path.read_text(encoding="utf-8"))
    assert rv.resolve_excluded(b, str(tmp_path / "missing.json")) == rs.select_sample(b)
    assert rv.resolve_excluded(b, None) == rs.select_sample(b)
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"sample_ids": ["x"]}), encoding="utf-8")
    assert rv.resolve_excluded(b, str(m)) == ["x"]


def test_summaries_render_from_saved_outputs(payload, tmp_path):
    rep = rv.run_interventions(payload, "q1", "D_highC_highP", FakeModel())
    text = rv.summarize_interventions(rep)
    assert "repair_summary" in text and "item_removal" in text and "claim_family_removal" in text
    assert "BASELINE specialist extracts" in text
    out = tmp_path / "f.jsonl"
    rv.run_factorial(payload, ["q1"], FakeModel(), out)
    table = rv.summarize_factorial(rv.read_records(out), ["q1"])
    assert "coverage: 16/16 exact=True all_ok=True" in table
    assert table.count("\n") >= 18 and "D_highC_highP" in table and "arbiter" in table
