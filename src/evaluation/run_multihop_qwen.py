"""Run one real MultiHop-RAG question and controlled evidence interventions.

This is deliberately separate from the synthetic seed_sample.json pipeline.
It saves raw prompts/outputs and scores the extracted answer against the gold
answer belonging to the same sample_id. It does not claim provenance is real:
condition lineage IDs in the current bundle are synthetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.multihop_scoring import score_multihop_answer

DEFAULT_BUNDLE = Path("data/processed/multihoprag/controlled_bundles_30.json")
DEFAULT_OUTPUT = Path("results/multihoprag/real_qwen_interventions.json")
CONDITIONS = ("A_lowC_lowP", "B_highC_lowP", "C_lowC_highP", "D_highC_highP")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_question(bundle_path: Path, sample_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    matches = [item for item in payload.get("questions", []) if item.get("sample_id") == sample_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one question for sample_id={sample_id!r}; found {len(matches)}")
    return payload, matches[0]


def remove_one_document(documents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(documents) < 2:
        raise ValueError("Cannot perform document-removal intervention with fewer than two documents")
    removed = documents[0]
    remaining = [doc for doc in documents if doc.get("doc_id") != removed.get("doc_id")]
    return remaining, {"kind": "item_removal", "removed_doc_ids": [removed["doc_id"]]}


def remove_one_evidence_unit(documents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove one controlled duplicate family, not a claimed real provenance cluster."""
    if not documents:
        raise ValueError("Cannot perform evidence-unit removal on an empty document list")
    unit_id = documents[0].get("evidence_unit_id")
    if not unit_id:
        raise ValueError("Documents need evidence_unit_id for controlled-family removal")
    removed_ids = [doc["doc_id"] for doc in documents if doc.get("evidence_unit_id") == unit_id]
    remaining = [doc for doc in documents if doc.get("evidence_unit_id") != unit_id]
    if len(removed_ids) < 2:
        raise ValueError(
            "Evidence-unit removal is identical to item removal for this family "
            f"({unit_id!r} contains {len(removed_ids)} document). Choose a condition "
            "with a multi-document controlled family, and do not interpret synthetic "
            "lineage as verified real-world provenance."
        )
    if not remaining:
        raise ValueError("Evidence-unit removal would remove all evidence; choose another unit")
    return remaining, {
        "kind": "evidence_unit_removal",
        "removed_evidence_unit_id": unit_id,
        "removed_doc_ids": removed_ids,
        "interpretation": "controlled evidence-unit family; not verified real-world provenance",
    }


class MultiHopQwenEvaluator:
    def __init__(self, model: Any):
        self.model = model

    def run_trial(
        self,
        question: dict[str, Any],
        condition: str,
        documents: list[dict[str, Any]],
        intervention: dict[str, Any],
    ) -> dict[str, Any]:
        query = question["query"]
        stage_records: list[dict[str, Any]] = []
        specialist_reports: list[str] = []
        total_prompt_tokens = 0
        total_generated_tokens = 0
        all_tokens_measured = True
        start_total = time.perf_counter()

        for specialist_index in range(3):
            chunk = [doc for i, doc in enumerate(documents) if i % 3 == specialist_index]
            context = "\n\n".join(
                "\n".join(
                    part for part in (
                        f"Document ID: {d.get('doc_id', 'unknown')}",
                        f"Article title: {d.get('source_title', 'unknown')}",
                        f"Publisher: {d.get('source_publisher', 'unknown')}",
                        f"Published at: {d.get('published_at', 'unknown')}",
                        f"Source URL: {d.get('parent_source_url', 'unknown')}",
                        f"Evidence text: {d.get('content', '')}",
                    ) if part
                )
                for d in chunk
            ) or "No evidence provided."
            system_prompt = (
                "You are a specialist in a parallel evidence-reasoning system. Use only the supplied context. "
                "Extract relevant facts. If evidence is missing, say so. Do not guess. Be concise."
            )
            prompt_snapshot = {"system_prompt": system_prompt, "context": context, "question": query}
            stage_start = time.perf_counter()
            result = self.model.generate_answer(
                context=context, question=query, system_prompt=system_prompt, max_new_tokens=96
            )
            latency = time.perf_counter() - stage_start
            output = str(result.get("answer", ""))
            specialist_reports.append(f"Specialist_{specialist_index + 1} report:\n{output}")
            stage_records.append({
                "stage": f"specialist_{specialist_index + 1}",
                "input_doc_ids": [doc["doc_id"] for doc in chunk],
                "prompt": prompt_snapshot,
                "raw_output": output,
                "prompt_tokens_measured": result.get("prompt_tokens"),
                "generated_tokens_measured": result.get("generated_tokens"),
                "latency_seconds_measured": latency,
            })
            if isinstance(result.get("prompt_tokens"), int) and isinstance(result.get("generated_tokens"), int):
                total_prompt_tokens += result["prompt_tokens"]
                total_generated_tokens += result["generated_tokens"]
            else:
                all_tokens_measured = False

        context = "\n\n".join(specialist_reports)
        system_prompt = (
            "You are the arbiter in a parallel retrieval-augmented system. Answer using only the specialist reports. "
            "Combine supported facts, do not invent missing facts, and state uncertainty when evidence is insufficient. "
            "Return a direct answer."
        )
        prompt_snapshot = {"system_prompt": system_prompt, "context": context, "question": query}
        stage_start = time.perf_counter()
        result = self.model.generate_answer(
            context=context, question=query, system_prompt=system_prompt, max_new_tokens=128
        )
        latency = time.perf_counter() - stage_start
        final_output = str(result.get("answer", ""))
        stage_records.append({
            "stage": "arbiter",
            "input_doc_ids": [doc["doc_id"] for doc in documents],
            "prompt": prompt_snapshot,
            "raw_output": final_output,
            "prompt_tokens_measured": result.get("prompt_tokens"),
            "generated_tokens_measured": result.get("generated_tokens"),
            "latency_seconds_measured": latency,
        })
        if isinstance(result.get("prompt_tokens"), int) and isinstance(result.get("generated_tokens"), int):
            total_prompt_tokens += result["prompt_tokens"]
            total_generated_tokens += result["generated_tokens"]
        else:
            all_tokens_measured = False

        # This is the single scoring adapter for this real-data path.
        scoring = score_multihop_answer(final_output, question["answer"])
        return {
            "sample_id": question["sample_id"],
            "query": query,
            "gold_answer": question["answer"],
            "condition": condition,
            "condition_design": {
                "redundancy": question["conditions"][condition].get("redundancy"),
                "dependence": question["conditions"][condition].get("dependence"),
                "lineage_semantics": question["conditions"][condition].get("lineage_semantics"),
            },
            "intervention": intervention,
            "input_doc_ids": [doc["doc_id"] for doc in documents],
            "raw_final_output": final_output,
            "scoring": scoring,
            "stage_records": stage_records,
            "latency_seconds_end_to_end_measured": time.perf_counter() - start_total,
            "prompt_tokens_measured": total_prompt_tokens if all_tokens_measured else None,
            "generated_tokens_measured": total_generated_tokens if all_tokens_measured else None,
            "model_id": result.get("model_id"),
        }


def run_intervention_set(
    bundle_path: Path,
    sample_id: str,
    condition: str,
    model: Any,
) -> dict[str, Any]:
    payload, question = load_question(bundle_path, sample_id)
    if condition not in question.get("conditions", {}):
        raise ValueError(f"Condition {condition!r} missing for sample {sample_id!r}")
    original_docs = question["conditions"][condition]["documents"]
    if not original_docs:
        raise ValueError("Selected condition has no documents")

    evaluator = MultiHopQwenEvaluator(model)
    trials = []
    trials.append(evaluator.run_trial(question, condition, list(original_docs), {"kind": "baseline"}))
    item_docs, item_meta = remove_one_document(list(original_docs))
    trials.append(evaluator.run_trial(question, condition, item_docs, item_meta))
    cluster_docs, cluster_meta = remove_one_evidence_unit(list(original_docs))
    trials.append(evaluator.run_trial(question, condition, cluster_docs, cluster_meta))

    baseline_correct = bool(trials[0]["scoring"]["correct"])
    for trial in trials:
        trial["baseline_correct"] = baseline_correct
        trial["is_repair"] = bool(not baseline_correct and trial["scoring"]["correct"] and trial["intervention"]["kind"] != "baseline")
        trial["correctness_changed_vs_baseline"] = bool(trial["scoring"]["correct"] != baseline_correct)

    return {
        "schema_version": 1,
        "experiment": "real_multihop_qwen_single_question_interventions",
        "dataset": payload.get("dataset"),
        "bundle_artifact_type": payload.get("artifact_type"),
        "bundle_sha256": sha256_file(bundle_path),
        "sample_id": sample_id,
        "condition": condition,
        "gold_answer": question["answer"],
        "model_id": getattr(model, "model_id", None),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Single-question feasibility run; not a research result or statistical estimate.",
            "The bundle assigns synthetic lineage IDs; provenance dependence is not verified.",
            "High-redundancy conditions currently contain twice as many documents as low-redundancy conditions; redundancy is confounded with document count.",
            "Evidence-unit removal targets a controlled duplicate family, not a verified real-world provenance cluster.",
            "The prompt includes article title, publisher, publication time, URL, and evidence text; metadata does not itself prove answer support or source independence.",
            "A question must receive a manual evidence-support review before its correctness score is treated as a valid benchmark outcome.",
            "Raw prompts and outputs are preserved for manual review.",
        ],
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--sample-id", default="mh_0174")
    parser.add_argument("--condition", choices=CONDITIONS, default="A_lowC_lowP")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Controlled bundle not found: {bundle_path}")
    # Lazy import avoids loading torch/transformers for tests and CLI validation.
    from src.models.local_qwen import LocalQwen

    model = LocalQwen()
    report = run_intervention_set(bundle_path, args.sample_id, args.condition, model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(report['trials'])} trials for {report['sample_id']} / {report['condition']}: {output_path}")
    print("Correctness by trial:", json.dumps([
        {"intervention": row["intervention"]["kind"], "correct": row["scoring"]["correct"], "is_repair": row["is_repair"]}
        for row in report["trials"]
    ]))
    print("Reminder: this is a one-question feasibility run; do not interpret it as research confirmation.")


if __name__ == "__main__":
    main()
