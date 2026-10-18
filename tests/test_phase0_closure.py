import json
from pathlib import Path
from types import SimpleNamespace

from src.attribution.cluster_ablation import ClusterAblation, is_repair
from src.attribution.loo import ItemLevelLOO
from src.evaluation.scoring import AnswerScorer
from src.pipeline.qwen_pipeline import QwenParallelPipeline
from src.schema import FaultType

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "seed_sample.json"
CONDITIONS = ["A_lowC_lowP", "B_highC_lowP", "C_lowC_highP", "D_highC_highP"]
FAULTS = [FaultType.NONE, FaultType.RETRIEVAL, FaultType.SPECIALIST, FaultType.ARBITER]


class FakeMeasuredModel:
    def generate_answer(self, context, question, system_prompt, max_new_tokens=96):
        return {
            "answer": (
                "The Trade Oversight Commission authorized the Sector 7 supply chain subsidy; "
                "regional infrastructure spending increased by 15%; Sector 7 employment rates "
                "rose steadily through Q3."
            ),
            "prompt_tokens": 11,
            "generated_tokens": 7,
            "model_id": "fake-measurement-test",
        }


def test_repair_predicate_truth_table():
    assert is_repair(False, True) is True
    assert is_repair(False, False) is False
    assert is_repair(True, False) is False
    assert is_repair(True, True) is False


class ActualInterventionFixture:
    """Deterministic fixture: removing the poison document repairs the answer."""
    def execute_pipeline(self, sample_path, condition_key, injected_layer,
                         documents_override=None, write_log=False):
        data = json.loads(Path(sample_path).read_text(encoding="utf-8"))
        docs = data["conditions"][condition_key] if documents_override is None else documents_override
        poison_present = any(d["doc_id"] == "poison" for d in docs)
        verdict = "wrong response" if poison_present else data["gold_answer"]
        return SimpleNamespace(
            query_id=data["query_id"], query=data["query"], gold_answer=data["gold_answer"],
            required_claims=[], condition=condition_key, final_verdict=verdict,
            injected_fault_layer=injected_layer, predicted_fault_layer=FaultType.NONE,
            is_failed_trajectory=verdict != data["gold_answer"],
        )


def _write_intervention_fixture(path):
    data = {
        "query_id": "INTERVENTION-1",
        "query": "What is the correct answer?",
        "gold_answer": "correct answer",
        "required_claims": [],
        "conditions": {
            "A_lowC_lowP": [
                {"doc_id": "poison", "content": "misleading answer", "source_cluster_id": "poison_cluster", "claim_id": "c1"},
                {"doc_id": "support", "content": "correct answer", "source_cluster_id": "support_cluster", "claim_id": "c1"},
            ]
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_repair_predicate_through_real_loo_and_cluster_interventions(tmp_path):
    fixture = tmp_path / "intervention.json"
    _write_intervention_fixture(fixture)
    pipeline = ActualInterventionFixture()

    loo = ItemLevelLOO(pipeline).evaluate_loo(str(fixture), "A_lowC_lowP")
    assert loo["baseline_correct"] is False
    assert loo["loo_effects"]["poison"]["is_repair"] is True
    assert loo["loo_repair_count"] == 1
    assert loo["loo_repair_rate"] == 1.0

    cluster = ClusterAblation(pipeline).evaluate_cluster_ablation(str(fixture), "A_lowC_lowP")
    assert cluster["baseline_correct"] is False
    assert cluster["cluster_ablation_effects"]["poison_cluster"]["is_repair"] is True
    assert cluster["repair_count"] == 1
    assert cluster["cluster_ablation_repair_rate"] == 1.0


def test_qwen_factorial_coverage_trace_integrity_and_measured_token_metrics(tmp_path):
    pipeline = QwenParallelPipeline(model=FakeMeasuredModel(), logs_dir=str(tmp_path / "logs"))
    seen = set()
    for condition in CONDITIONS:
        for fault in FAULTS:
            trace = pipeline.execute_pipeline(str(SAMPLE), condition, fault, write_log=False)
            seen.add((condition, fault))
            assert trace.condition == condition
            assert trace.injected_fault_layer == fault
            assert trace.predicted_fault_layer == fault
            assert trace.model_calls_measured == 4
            assert trace.prompt_tokens_measured == 44
            assert trace.generated_tokens_measured == 28
            assert trace.model_id == "fake-measurement-test"
            # Gold labels must not enter diagnostic observations or specialist prompts.
            assert all("injected_fault_layer" not in " ".join(t.actual_inputs) for t in trace.specialist_traces)
            if fault == FaultType.RETRIEVAL:
                first = trace.specialist_traces[0]
                assert first.assigned_docs == ["faulty_doc"]
                assert first.retrieved_docs[0].doc_id == "d1"
                assert "faulty_doc" in first.actual_inputs[1]
                assert all(d.doc_id != "faulty_doc" for d in trace.original_corpus)
            else:
                assigned = [doc_id for t in trace.specialist_traces for doc_id in t.assigned_docs]
                assert sorted(assigned) == ["d1", "d2", "d3"]
    assert len(seen) == 16


def test_answer_scorer_metric_values_and_missing_measurements():
    metrics = AnswerScorer.score_metrics("the Eiffel Tower", "Eiffel Tower")
    assert metrics["correct"] is True
    assert metrics["exact_match"] is True
    assert metrics["token_f1"] == 1.0
    assert AnswerScorer.score_metrics("insufficient evidence", "Eiffel Tower")["correct"] is False

    # Model traces must represent unavailable counters as null, never invented zeroes.
    class NoUsageModel:
        def generate_answer(self, context, question, system_prompt, max_new_tokens=96):
            return {"answer": "unknown answer", "model_id": "no-usage-model"}
    trace = QwenParallelPipeline(model=NoUsageModel()).execute_pipeline(
        str(SAMPLE), "A_lowC_lowP", FaultType.NONE, write_log=False
    )
    assert trace.prompt_tokens_measured is None
    assert trace.generated_tokens_measured is None
    assert trace.model_calls_measured == 4


def test_qwen_assigns_every_document_once_when_corpus_exceeds_three(tmp_path):
    pipeline = QwenParallelPipeline(model=FakeMeasuredModel(), logs_dir=str(tmp_path / "logs"))
    data = json.loads(SAMPLE.read_text(encoding="utf-8"))
    docs = list(data["conditions"]["A_lowC_lowP"])
    for i in range(3, 7):
        docs.append({
            "doc_id": f"d{i + 1}",
            "content": f"Extra synthetic evidence item {i + 1}.",
            "source_cluster_id": f"extra_cluster_{i + 1}",
            "claim_id": "claim_toc_auth",
            "claim_ids": ["claim_toc_auth"],
        })
    trace = pipeline.execute_pipeline(
        str(SAMPLE), "A_lowC_lowP", FaultType.NONE,
        documents_override=docs, write_log=False,
    )
    assigned = [doc_id for specialist in trace.specialist_traces for doc_id in specialist.assigned_docs]
    assert sorted(assigned) == sorted(doc["doc_id"] for doc in docs)
    assert len(assigned) == len(set(assigned))
