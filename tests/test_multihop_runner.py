import json
from pathlib import Path

import pytest

from src.evaluation.run_multihop_qwen import (
    MultiHopQwenEvaluator,
    load_question,
    remove_one_document,
    remove_one_evidence_unit,
    run_intervention_set,
)


class FakeModel:
    model_id = "fake-test-model"

    def __init__(self, answer="Yes"):
        self.answer = answer
        self.calls = []

    def generate_answer(self, *, context, question, system_prompt, max_new_tokens):
        self.calls.append({
            "context": context,
            "question": question,
            "system_prompt": system_prompt,
            "max_new_tokens": max_new_tokens,
        })
        return {
            "answer": self.answer,
            "prompt_tokens": 11,
            "generated_tokens": 2,
            "model_id": self.model_id,
        }


def fixture_question():
    docs = [
        {"doc_id": "d1", "content": "fact one", "evidence_unit_id": "u1"},
        {"doc_id": "d1-copy", "content": "fact one", "evidence_unit_id": "u1"},
        {"doc_id": "d2", "content": "fact two", "evidence_unit_id": "u2"},
    ]
    condition = {"redundancy": "high", "dependence": "low", "lineage_semantics": "synthetic", "documents": docs}
    return {
        "sample_id": "sample-x",
        "query": "Is this supported?",
        "answer": "Yes",
        "conditions": {"A_lowC_lowP": condition},
    }


def test_removal_interventions_record_exact_documents():
    docs = fixture_question()["conditions"]["A_lowC_lowP"]["documents"]
    item_docs, item = remove_one_document(docs)
    assert item["removed_doc_ids"] == ["d1"]
    assert len(item_docs) == 2

    cluster_docs, cluster = remove_one_evidence_unit(docs)
    assert cluster["removed_evidence_unit_id"] == "u1"
    assert set(cluster["removed_doc_ids"]) == {"d1", "d1-copy"}
    assert [doc["doc_id"] for doc in cluster_docs] == ["d2"]
    assert "not verified" in cluster["interpretation"]


def test_runner_pairs_actual_output_with_same_sample_gold():
    question = fixture_question()
    model = FakeModel("Yes. The evidence supports this.")
    result = MultiHopQwenEvaluator(model).run_trial(
        question,
        "A_lowC_lowP",
        question["conditions"]["A_lowC_lowP"]["documents"],
        {"kind": "baseline"},
    )
    assert result["sample_id"] == "sample-x"
    assert result["gold_answer"] == "Yes"
    assert result["raw_final_output"] == "Yes. The evidence supports this."
    assert result["scoring"]["raw_output"] == result["raw_final_output"]
    assert result["scoring"]["gold_answer"] == result["gold_answer"]
    assert result["scoring"]["correct"] is True
    assert len(result["stage_records"]) == 4
    assert result["prompt_tokens_measured"] == 44
    assert result["generated_tokens_measured"] == 8


def test_load_question_requires_unique_sample_id(tmp_path):
    path = tmp_path / "bundle.json"
    q = fixture_question()
    path.write_text(json.dumps({"dataset": "test", "questions": [q, q]}), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        load_question(path, "sample-x")


def test_intervention_set_produces_baseline_item_and_family_trials(tmp_path):
    path = tmp_path / "bundle.json"
    q = fixture_question()
    path.write_text(json.dumps({"dataset": "test", "artifact_type": "synthetic", "questions": [q]}), encoding="utf-8")
    model = FakeModel("Yes")
    report = run_intervention_set(path, "sample-x", "A_lowC_lowP", model)
    assert [t["intervention"]["kind"] for t in report["trials"]] == [
        "baseline", "item_removal", "evidence_unit_removal"
    ]
    assert all(t["sample_id"] == "sample-x" for t in report["trials"])
    assert all(t["gold_answer"] == "Yes" for t in report["trials"])
    assert all(t["scoring"]["correct"] for t in report["trials"])
    assert all(not t["is_repair"] for t in report["trials"])
    assert len(model.calls) == 12


def test_prompt_context_includes_source_metadata():
    question = fixture_question()
    docs = question["conditions"]["A_lowC_lowP"]["documents"]
    docs[0].update({
        "source_title": "Example article",
        "source_publisher": "Example Publisher",
        "published_at": "2025-01-01",
        "parent_source_url": "https://example.test/article",
    })
    model = FakeModel("Yes")
    result = MultiHopQwenEvaluator(model).run_trial(
        question, "A_lowC_lowP", docs, {"kind": "baseline"}
    )
    first_context = result["stage_records"][0]["prompt"]["context"]
    assert "Article title: Example article" in first_context
    assert "Publisher: Example Publisher" in first_context
    assert "Source URL: https://example.test/article" in first_context
    assert "Evidence text: fact one" in first_context


def test_evidence_unit_removal_rejects_single_document_family():
    docs = [{"doc_id": "d1", "content": "fact", "evidence_unit_id": "u1"},
            {"doc_id": "d2", "content": "other", "evidence_unit_id": "u2"}]
    with pytest.raises(ValueError, match="identical to item removal"):
        remove_one_evidence_unit(docs)
