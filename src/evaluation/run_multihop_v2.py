"""Real-model MultiHop-RAG runner for controlled_bundles_v2.json.

Subcommands
  interventions  baseline + full leave-one-out + one explicit claim-family removal
  factorial      4 conditions x 4 fault layers x reviewed samples (checkpointed JSONL)

Separate from the synthetic fixture pipeline. All trajectories go through
src.pipeline.multihop_faults.execute_fault_trial and the canonical scorer
(src.evaluation.multihop_scoring.score_multihop_answer).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.evaluation.multihop_scoring import score_multihop_answer
from src.pipeline.multihop_faults import ABSTENTION, execute_fault_trial
from src.schema import FaultType

DEFAULT_BUNDLE = Path("data/processed/multihoprag/controlled_bundles_v2.json")
CONDITIONS = ("A_lowC_lowP", "B_highC_lowP", "C_lowC_highP", "D_highC_highP")
FAULTS = tuple(FaultType)  # none, retrieval, specialist, arbiter

PROMPT_VERSION = "mh-v2-prompts-1"
SPECIALIST_PROMPT = (
    "You are one specialist in a parallel evidence system. You see only a subset of the "
    "documents; other specialists see the rest. Use only the supplied documents. List the "
    "facts in your documents that are relevant to the question, one per line, citing the "
    "Document ID. Do not answer the whole question and do not comment on documents you "
    "cannot see. If nothing is relevant, write exactly: No relevant facts."
)
ARBITER_PROMPT = (
    "You are the arbiter in a parallel retrieval-augmented system. Answer using only the "
    "specialist reports. Combine supported facts and do not invent missing facts. End with "
    "one final line. For yes/no questions use 'Answer: Yes' or 'Answer: No'. For other "
    "questions use 'Answer: <short answer>'. If the reports are insufficient use "
    "'Answer: Insufficient evidence'."
)
MAX_NEW_TOKENS = {"specialist": 160, "arbiter": 160}


def prompt_manifest() -> dict[str, Any]:
    text = PROMPT_VERSION + SPECIALIST_PROMPT + ARBITER_PROMPT + json.dumps(MAX_NEW_TOKENS)
    return {"version": PROMPT_VERSION, "specialist_prompt": SPECIALIST_PROMPT,
            "arbiter_prompt": ARBITER_PROMPT, "max_new_tokens": MAX_NEW_TOKENS,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def alias(doc_id: str) -> str:
    """Neutral display ID: hides condition, copy and role information from the model."""
    return "d_" + hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:8]


def render_doc(doc: dict[str, Any]) -> str:
    return "\n".join([
        f"Document ID: {alias(doc['doc_id'])}",
        f"Article title: {doc.get('source_title', 'unknown')}",
        f"Publisher: {doc.get('source_publisher', 'unknown')}",
        f"Published at: {doc.get('published_at', 'unknown')}",
        f"Evidence text: {doc.get('content', '')}",
    ])


class StageRecorder:
    """generate(role, context, question) adapter that keeps every stage's raw record."""

    def __init__(self, model: Any):
        self.model = model
        self.records: list[dict[str, Any]] = []

    def __call__(self, role: str, context: str, question: str) -> str:
        system = SPECIALIST_PROMPT if role == "specialist" else ARBITER_PROMPT
        record: dict[str, Any] = {
            "stage": f"{len(self.records) + 1:02d}_{role}", "role": role,
            "prompt": {"system_prompt": system, "context": context, "question": question},
        }
        start = time.perf_counter()
        try:
            result = self.model.generate_answer(
                context=context, question=question, system_prompt=system,
                max_new_tokens=MAX_NEW_TOKENS[role])
        except Exception as exc:  # recorded, then re-raised for the trajectory handler
            record.update(error=f"{type(exc).__name__}: {exc}",
                          latency_seconds_measured=time.perf_counter() - start)
            self.records.append(record)
            raise
        record.update(
            raw_output=str(result.get("answer", "")),
            prompt_tokens_measured=result.get("prompt_tokens"),
            generated_tokens_measured=result.get("generated_tokens"),
            latency_seconds_measured=time.perf_counter() - start,
            model_id=result.get("model_id"))
        self.records.append(record)
        return record["raw_output"]


def _sum_measured(records: list[dict[str, Any]], key: str) -> Optional[int]:
    values = [r.get(key) for r in records]
    return sum(values) if values and all(isinstance(v, int) for v in values) else None


def answer_status(scoring: dict[str, Any]) -> str:
    if scoring["extracted_answer"]:
        return "answered"
    if ABSTENTION.search(scoring["raw_output"] or ""):
        return "abstained"
    return "empty_extraction"


def run_trajectory(question: dict[str, Any], condition: str, layer: FaultType, model: Any,
                   intervention: Optional[dict[str, Any]] = None,
                   exclude_doc_ids=()) -> dict[str, Any]:
    recorder = StageRecorder(model)
    docs = question["conditions"][condition]["documents"]
    excluded = set(exclude_doc_ids)
    base = {
        "sample_id": question["sample_id"], "condition": condition,
        "injected_layer": layer.value, "gold_answer": question["answer"],
        "query": question["query"], "intervention": intervention or {"kind": "baseline"},
        "input_doc_ids": [d["doc_id"] for d in docs if d["doc_id"] not in excluded],
    }
    start = time.perf_counter()
    try:
        trial = execute_fault_trial(question, condition, layer, recorder,
                                    exclude_doc_ids=excluded, render_doc=render_doc)
    except Exception as exc:
        return {**base, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "stage_records": recorder.records,
                "latency_seconds_end_to_end_measured": time.perf_counter() - start}

    final_output = trial.observation.final_verdict
    scoring = score_multihop_answer(final_output, question["answer"])  # canonical scorer
    return {
        **base, "status": "ok", "error": None,
        "raw_final_output": final_output, "scoring": scoring,
        "answer_status": answer_status(scoring),
        "observation": trial.observation.model_dump(),
        "injection": {**asdict(trial.injection), "layer": trial.injection.layer.value},
        "predicted_layer": trial.predicted_layer.value,
        "diagnosis_correct": trial.predicted_layer == layer,
        "stage_records": recorder.records,
        "prompt_tokens_measured": _sum_measured(recorder.records, "prompt_tokens_measured"),
        "generated_tokens_measured": _sum_measured(recorder.records, "generated_tokens_measured"),
        "model_calls_measured": len(recorder.records),
        "latency_seconds_end_to_end_measured": time.perf_counter() - start,
    }


# ------------------------------------------------------------------ bundle
def load_bundle(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_question(payload: dict[str, Any], sample_id: str) -> dict[str, Any]:
    matches = [q for q in payload["questions"] if q["sample_id"] == sample_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one question for {sample_id!r}; found {len(matches)}")
    return matches[0]


def default_cluster_unit(documents: list[dict[str, Any]]) -> str:
    """First claim family (>=2 docs) whose removal leaves other evidence."""
    families: dict[str, int] = {}
    for d in documents:
        families[d["evidence_unit_id"]] = families.get(d["evidence_unit_id"], 0) + 1
    for unit in sorted(families):
        if families[unit] >= 2 and families[unit] < len(documents):
            return unit
    raise ValueError("No multi-document claim family in this condition; use B_highC_lowP or D_highC_highP")


# ----------------------------------------------------------- interventions
def run_interventions(payload: dict[str, Any], sample_id: str, condition: str, model: Any,
                      cluster_unit: Optional[str] = None) -> dict[str, Any]:
    question = get_question(payload, sample_id)
    docs = question["conditions"][condition]["documents"]
    unit = cluster_unit or default_cluster_unit(docs)
    family = [d["doc_id"] for d in docs if d["evidence_unit_id"] == unit]
    if len(family) < 2:
        raise ValueError(f"Family {unit!r} has {len(family)} document(s); identical to item removal")
    if len(family) == len(docs):
        raise ValueError(f"Family {unit!r} is all evidence")

    trials = [run_trajectory(question, condition, FaultType.NONE, model)]
    for d in docs:  # full leave-one-out, stable document IDs
        trials.append(run_trajectory(
            question, condition, FaultType.NONE, model,
            intervention={"kind": "item_removal", "removed_doc_ids": [d["doc_id"]]},
            exclude_doc_ids=[d["doc_id"]]))
    trials.append(run_trajectory(
        question, condition, FaultType.NONE, model,
        intervention={"kind": "claim_family_removal", "removed_evidence_unit_id": unit,
                      "removed_doc_ids": family,
                      "interpretation": "controlled claim family; not verified provenance"},
        exclude_doc_ids=family))

    base = trials[0]
    base_ok = base["status"] == "ok"
    base_correct = bool(base_ok and base["scoring"]["correct"])
    eligible = base_ok and not base_correct
    for t in trials[1:]:
        ok = base_ok and t["status"] == "ok"
        t["effect"] = None if not ok else {
            "correctness_changed": bool(t["scoring"]["correct"]) != base_correct,
            "extracted_answer_changed": t["scoring"]["extracted_answer"] != base["scoring"]["extracted_answer"],
            "token_f1_delta": t["scoring"]["token_f1"] - base["scoring"]["token_f1"],
        }
        t["is_repair"] = bool(eligible and ok and t["scoring"]["correct"])
    repairs = sum(1 for t in trials[1:] if t.get("is_repair"))
    return {
        "schema_version": 2, "experiment": "multihop_v2_interventions",
        "sample_id": sample_id, "condition": condition, "cluster_unit": unit,
        "gold_answer": question["answer"], "baseline_eligible_for_repair": eligible,
        "repair_summary": {
            "eligible_failed_baselines": 1 if eligible else 0,
            "interventions_tried": len(trials) - 1, "interventions_that_repair": repairs,
            "repair_success": (1.0 if repairs else 0.0) if eligible else None,
            "note": "None = no eligible failed baseline; repair success is undefined, not zero.",
        },
        "trials": trials,
    }


# --------------------------------------------------------------- factorial
def cell_key(sample_id: str, condition: str, layer: str) -> str:
    return f"{sample_id}|{condition}|{layer}"


def expected_cells(sample_ids) -> list[str]:
    return [cell_key(s, c, f.value) for s, c, f in itertools.product(sample_ids, CONDITIONS, FAULTS)]


def read_records(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def coverage_report(sample_ids, records: list[dict[str, Any]]) -> dict[str, Any]:
    expected = expected_cells(sample_ids)
    keys = [cell_key(r["sample_id"], r["condition"], r["injected_layer"]) for r in records]
    present = set(keys)
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    missing = sorted(set(expected) - present)
    unexpected = sorted(present - set(expected))
    errors = [cell_key(r["sample_id"], r["condition"], r["injected_layer"])
              for r in records if r.get("status") != "ok"]
    return {
        "expected_cells": len(expected), "recorded_cells": len(keys),
        "missing": missing, "duplicates": dupes, "unexpected": unexpected, "error_cells": errors,
        "exact_coverage": not missing and not dupes and not unexpected,
        "all_ok": not missing and not dupes and not unexpected and not errors,
    }


def run_factorial(payload: dict[str, Any], sample_ids, model: Any, output_jsonl: Path,
                  resume: bool = True) -> dict[str, Any]:
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = {cell_key(r["sample_id"], r["condition"], r["injected_layer"])
            for r in read_records(output_jsonl)} if resume else set()
    if not resume and output_jsonl.exists():
        raise FileExistsError(f"{output_jsonl} exists; pass resume or choose a new path")
    for sample_id in sample_ids:
        question = get_question(payload, sample_id)
        for condition, layer in itertools.product(CONDITIONS, FAULTS):
            if cell_key(sample_id, condition, layer.value) in done:
                continue
            record = run_trajectory(question, condition, layer, model)
            with output_jsonl.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return coverage_report(sample_ids, read_records(output_jsonl))


# --------------------------------------------------------------------- CLI
def resolve_samples(args) -> tuple[list[str], dict[str, Any]]:
    if args.sample_manifest:
        manifest = json.loads(Path(args.sample_manifest).read_text(encoding="utf-8"))
        if not manifest.get("reviewed"):
            raise SystemExit("Sample manifest is not marked reviewed=true")
        return list(manifest["sample_ids"]), {"sample_manifest_sha256": sha256_file(Path(args.sample_manifest)),
                                             "reviewed": True}
    if args.allow_unreviewed and args.sample_ids:
        return list(args.sample_ids), {"reviewed": False}
    raise SystemExit("Provide --sample-manifest (reviewed) or --allow-unreviewed with --sample-ids")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    sub = parser.add_subparsers(dest="command", required=True)
    inter = sub.add_parser("interventions")
    inter.add_argument("--sample-id", required=True)
    inter.add_argument("--condition", choices=CONDITIONS, default="D_highC_highP")
    inter.add_argument("--cluster-unit")
    inter.add_argument("--output", default="results/phase0/interventions.json")
    fact = sub.add_parser("factorial")
    fact.add_argument("--sample-manifest")
    fact.add_argument("--sample-ids", nargs="*")
    fact.add_argument("--allow-unreviewed", action="store_true")
    fact.add_argument("--output", default="results/phase0/factorial.jsonl")
    fact.add_argument("--coverage-output", default="results/phase0/factorial_coverage.json")
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    payload = load_bundle(bundle_path)
    from src.models.local_qwen import LocalQwen  # lazy: keeps torch out of tests

    model = LocalQwen()
    meta = {"bundle": str(bundle_path), "bundle_sha256": sha256_file(bundle_path),
            "model_id": getattr(model, "model_id", None), "prompts": prompt_manifest(),
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat()}
    if args.command == "interventions":
        report = {**meta, **run_interventions(payload, args.sample_id, args.condition, model,
                                              args.cluster_unit)}
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(report['trials'])} trials to {out}")
        print(json.dumps(report["repair_summary"]))
    else:
        sample_ids, review = resolve_samples(args)
        coverage = run_factorial(payload, sample_ids, model, Path(args.output))
        coverage.update(meta, **review)
        Path(args.coverage_output).write_text(json.dumps(coverage, indent=2), encoding="utf-8")
        print(f"Coverage: {coverage['recorded_cells']}/{coverage['expected_cells']} "
              f"exact={coverage['exact_coverage']} errors={len(coverage['error_cells'])}")
        if not coverage["exact_coverage"]:
            raise SystemExit("Factorial coverage is not exact; see coverage report")


if __name__ == "__main__":
    main()
