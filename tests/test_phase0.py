import inspect
import json
from pathlib import Path

import pytest

from src.attribution.cluster_ablation import ClusterAblation, is_repair
from src.attribution.loo import ItemLevelLOO
from src.evaluation.scoring import AnswerScorer
from src.evaluation.validate_dataset import validate
from src.pipeline.agent_runner import BaselinePipeline
from src.schema import DiagnosticObservation, FaultType

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "seed_sample.json"
CONDITIONS = ["A_lowC_lowP", "B_highC_lowP", "C_lowC_highP", "D_highC_highP"]
FAULTS = [FaultType.NONE, FaultType.RETRIEVAL, FaultType.SPECIALIST, FaultType.ARBITER]


def load_data():
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


def test_canonical_scoring_required_claims_and_edges():
    data = load_data()
    gold = data["gold_answer"]
    claims = data["required_claims"]
    assert AnswerScorer.score_answer(gold, gold, claims)
    assert not AnswerScorer.score_answer("The Department of Treasury authorized the action.", gold, claims)
    assert not AnswerScorer.score_answer("Insufficient evidence to determine the claims.", gold, claims)
    assert not AnswerScorer.score_answer("The Trade Oversight Commission did not authorize the subsidy.", gold, claims)
    assert not AnswerScorer.score_answer("", gold, claims)
    assert AnswerScorer.score_answer("Authorized by the Trade Oversight Commission (TOC).", "The Trade Oversight Commission (TOC).")
    assert not AnswerScorer.score_answer("The Trade Oversight Commission did not authorize it.", "The Trade Oversight Commission (TOC).")
    assert not AnswerScorer.score_answer("Authorized by the Department of Treasury.", "The Trade Oversight Commission (TOC).")


def test_dataset_factorial_and_evidence_coverage():
    validate(str(SAMPLE))
    data = load_data()
    assert set(data["conditions"]) == set(CONDITIONS)
    assert len(data["required_claims"]) == 3
    ids = [c["claim_id"] for c in data["required_claims"]]
    assert len(ids) == len(set(ids))


def test_all_16_factorial_combinations_and_diagnosis():
    pipeline = BaselinePipeline()
    observed = set()
    for condition in CONDITIONS:
        for fault in FAULTS:
            trace = pipeline.execute_pipeline(str(SAMPLE), condition, fault, write_log=False)
            observed.add((condition, fault.value))
            assert trace.predicted_fault_layer is not None
            assert trace.predicted_fault_layer == fault, (condition, fault, trace.predicted_fault_layer)
            scored = AnswerScorer.score_answer(trace.final_verdict, trace.gold_answer, trace.required_claims)
            assert trace.is_failed_trajectory == (not scored)
    assert len(observed) == 16


def test_diagnostic_api_has_no_ground_truth_argument_or_field():
    assert list(inspect.signature(BaselinePipeline.diagnose).parameters) == ["observation"]
    fields = set(DiagnosticObservation.model_fields)
    assert "injected_fault_layer" not in fields
    assert "injected_fault" not in fields
    assert "gold_answer" not in fields
    assert "required_claims" not in fields


def test_repair_predicate_with_actual_fault_and_recovery_intervention():
    pipeline = BaselinePipeline()
    baseline = pipeline.execute_pipeline(str(SAMPLE), "A_lowC_lowP", FaultType.RETRIEVAL, write_log=False)
    baseline_success = AnswerScorer.score_answer(baseline.final_verdict, baseline.gold_answer, baseline.required_claims)
    repaired = pipeline.execute_pipeline(str(SAMPLE), "A_lowC_lowP", FaultType.NONE, write_log=False)
    intervention_success = AnswerScorer.score_answer(repaired.final_verdict, repaired.gold_answer, repaired.required_claims)
    repair_detected = is_repair(baseline_success, intervention_success)
    assert baseline_success is False
    assert intervention_success is True
    assert repair_detected is True
    assert is_repair(True, False) is False
    assert is_repair(False, False) is False
    assert is_repair(True, True) is False


def test_loo_uses_canonical_scorer_and_reports_effects_without_temp_files(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    before = set(Path("data").glob("temp_*.json"))
    result = ItemLevelLOO(BaselinePipeline()).evaluate_loo(str(SAMPLE), "A_lowC_lowP")
    after = set(Path("data").glob("temp_*.json"))
    assert result["loo_correctness_effect_count"] >= 1
    assert all("is_loo_correctness_effect" in x for x in result["loo_effects"].values())
    assert before == after


def test_cluster_ablation_repair_denominator_and_n_a():
    result = ClusterAblation(BaselinePipeline()).evaluate_cluster_ablation(str(SAMPLE), "A_lowC_lowP")
    assert result["eligible_failed_baseline_trajectories"] == 0
    assert result["cluster_ablation_repair_rate"] is None
    assert all("is_cluster_ablation_correctness_effect" in x and "is_repair" in x
               for x in result["cluster_ablation_effects"].values())


def test_all_conditions_baseline_scoring_is_consistent():
    pipeline = BaselinePipeline()
    for condition in CONDITIONS:
        trace = pipeline.execute_pipeline(str(SAMPLE), condition, FaultType.NONE, write_log=False)
        assert AnswerScorer.score_answer(trace.final_verdict, trace.gold_answer, trace.required_claims)
        assert trace.predicted_fault_layer == FaultType.NONE


def test_trace_integrity_records_actual_inputs_and_preserves_original_corpus():
    pipeline = BaselinePipeline()
    trace = pipeline.execute_pipeline(str(SAMPLE), "A_lowC_lowP", FaultType.RETRIEVAL, write_log=False)
    assert len(trace.original_corpus) == 3
    first = trace.specialist_traces[0]
    assert first.assigned_docs == ["faulty_doc"]
    assert [d.doc_id for d in first.retrieved_docs] == ["d1"]
    assert "faulty_doc" in first.actual_inputs[1]
    assert all(d.doc_id != "faulty_doc" for d in trace.original_corpus)


def test_dataset_validator_rejects_missing_required_evidence(tmp_path):
    from src.evaluation.validate_dataset import validate
    data = load_data()
    data["conditions"]["A_lowC_lowP"][0]["content"] = "Unrelated text."
    bad = tmp_path / "bad_seed.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SystemExit, match="DATASET VALIDATION FAILED"):
        validate(str(bad))


def test_synthetic_fixture_is_explicit_and_claim_ids_are_consistent():
    data = load_data()
    assert data["dataset_type"] == "synthetic_controlled_fixture"
    assert "fictional" in data["fictional_notice"].lower()
    bundle_map = data["bundle_claim_map"]
    for docs in data["conditions"].values():
        for doc in docs:
            expected = bundle_map.get(doc["claim_id"], [doc["claim_id"]])
            assert doc["claim_ids"] == expected


def test_hotpotqa_em_f1_scoring_edge_cases():
    exact = AnswerScorer.score_metrics("The Eiffel Tower", "Eiffel Tower")
    assert exact["exact_match"] is True and exact["correct"] is True
    partial = AnswerScorer.score_metrics("Paris", "Paris France")
    assert partial["token_f1"] > 0 and partial["correct"] is True
    assert AnswerScorer.score_metrics("", "Paris")["correct"] is False
    assert AnswerScorer.score_metrics("Insufficient information.", "Paris")["correct"] is False


def test_qwen_pipeline_adapter_runs_factorial_with_injected_fake_model(tmp_path):
    from src.pipeline.qwen_pipeline import QwenParallelPipeline

    class FakeLocalQwen:
        def generate_answer(self, context, question, system_prompt, max_new_tokens=96):
            # Deterministic stand-in tests the adapter contract only, not LLM quality.
            return {"answer": "The Trade Oversight Commission authorized the Sector 7 supply chain subsidy; regional infrastructure spending increased by 15%; Sector 7 employment rates rose steadily through Q3.",
                    "prompt_tokens": 10, "generated_tokens": 20, "model_id": "fake-test-model"}

    pipeline = QwenParallelPipeline(model=FakeLocalQwen(), logs_dir=str(tmp_path / "logs"))
    seen = set()
    for condition in CONDITIONS:
        for fault in FAULTS:
            trace = pipeline.execute_pipeline(str(SAMPLE), condition, fault, write_log=False)
            seen.add((condition, fault))
            assert trace.condition == condition
            assert trace.injected_fault_layer == fault
            assert trace.predicted_fault_layer is not None
            assert trace.original_corpus
            assert all("injected_fault_layer" not in str(x.actual_inputs) for x in trace.specialist_traces)
    assert len(seen) == 16
