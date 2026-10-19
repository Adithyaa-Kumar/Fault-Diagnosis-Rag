"""Real-model MultiHop-RAG runner for controlled_bundles_v2.json.

Subcommands
  baseline-sweep prompt development on dev questions outside the reviewed sample
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

PROMPT_VERSION = "mh-v2-prompts-3"
SPECIALIST_PROMPT = (
    "You are one specialist in a parallel evidence system. You see only some of the documents; "
    "other specialists read the rest. For EACH document below, write one line starting with its "
    "Document ID in brackets, followed by the one or two sentences of its evidence text that state "
    "its main factual claim, copied verbatim. If the evidence text has several sentences, prefer "
    "those that bear on the question. Do not judge relevance, do not answer the question, and do "
    "not add comments."
)
ARBITER_PROMPT = (
    "You are the arbiter in a parallel retrieval-augmented system. The reports are verbatim extracts "
    "from documents; some may be irrelevant to the question. Combine the extracts: a name given in "
    "one can identify the subject described in another. Answer using only the reports, with the "
    "best-supported answer, and do not invent facts. Use 'Insufficient evidence' only if no extract "
    "bears on the question. End with one final line. For yes/no questions use 'Answer: Yes' or "
    "'Answer: No'. For other questions use 'Answer: <short answer>'."
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


# ---------------------------------------------------------- baseline sweep
def run_baseline_sweep(payload: dict[str, Any], sample_ids, condition: str, model: Any) -> dict[str, Any]:
    """Baseline-only pass for prompt development on questions OUTSIDE the reviewed sample."""
    trials = [run_trajectory(get_question(payload, sid), condition, FaultType.NONE, model)
              for sid in sample_ids]
    ok = [t for t in trials if t["status"] == "ok"]
    summary = {
        "condition": condition, "n": len(trials), "errors": len(trials) - len(ok),
        "answered": sum(t["answer_status"] == "answered" for t in ok),
        "abstained": sum(t["answer_status"] == "abstained" for t in ok),
        "empty_extraction": sum(t["answer_status"] == "empty_extraction" for t in ok),
        "correct": sum(bool(t["scoring"]["correct"]) for t in ok),
        "specialists_with_no_facts": sum(
            1 for t in ok for st in t["stage_records"] if st["role"] == "specialist"
            and ABSTENTION.search(st["raw_output"]) or (st["role"] == "specialist" and st["raw_output"].strip().upper().startswith("NONE"))),
    }
    return {"summary": summary, "trials": trials}


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


# ----------------------------------------------------------------- summaries
def _short(text: Any, n: int = 90) -> str:
    return " ".join(str(text or "").split())[:n]


def summarize_interventions(report: dict[str, Any]) -> str:
    out = [f"{report['sample_id']} / {report['condition']} | gold={report['gold_answer']!r} | "
           f"cluster_unit={report['cluster_unit']} | prompts={report.get('prompts', {}).get('version')} | "
           f"bundle={str(report.get('bundle_sha256'))[:12]}",
           f"repair_summary: {json.dumps(report['repair_summary'])}", ""]
    base = report["trials"][0]
    for t in report["trials"]:
        iv = t["intervention"]
        removed = ",".join(d.split("_")[-3] + d.split("_")[-2][-2:] + "_" + d.split("_")[-1]
                           for d in iv.get("removed_doc_ids", [])) or "-"
        if t["status"] != "ok":
            out.append(f"{iv['kind']:21s} removed={removed:18s} ERROR {_short(t.get('error'))}")
            continue
        sc, eff = t["scoring"], t.get("effect") or {}
        out.append(f"{iv['kind']:21s} removed={removed:18s} status={t['answer_status']:10s} "
                   f"extracted={sc['extracted_answer']!r:16} correct={sc['correct']!s:5} "
                   f"f1d={eff.get('token_f1_delta', 0):+.2f} repair={t.get('is_repair')} "
                   f"tok={t['prompt_tokens_measured']}/{t['generated_tokens_measured']} "
                   f"{t['latency_seconds_end_to_end_measured']:.1f}s diag={t['predicted_layer']}")
        out.append(f"    final: {_short(t['raw_final_output'], 120)!r}")
    out.append("")
    if base["status"] == "ok":
        out.append("BASELINE specialist extracts:")
        for st in base["stage_records"]:
            if st["role"] == "specialist":
                out.append(f"  {st['stage']}: {_short(st.get('raw_output'), 220)!r}")
    return "\n".join(out)


def summarize_factorial(records: list[dict[str, Any]], sample_ids=None) -> str:
    rows: dict[tuple, list[dict[str, Any]]] = {}
    for r in records:
        rows.setdefault((r["condition"], r["injected_layer"]), []).append(r)
    head = f"{'condition':14s} {'fault':11s} {'n':>3s} {'ok':>3s} {'correct':>7s} {'abst':>4s} {'diag':>5s} {'lat_s':>6s} {'prompt_tok':>10s}"
    out = [head, "-" * len(head)]
    for key in sorted(rows):
        rs = rows[key]
        ok = [r for r in rs if r["status"] == "ok"]
        lat = [r["latency_seconds_end_to_end_measured"] for r in ok]
        pt = [r["prompt_tokens_measured"] for r in ok if r["prompt_tokens_measured"] is not None]
        out.append(f"{key[0]:14s} {key[1]:11s} {len(rs):3d} {len(ok):3d} "
                   f"{sum(bool(r['scoring']['correct']) for r in ok):7d} "
                   f"{sum(r['answer_status'] == 'abstained' for r in ok):4d} "
                   f"{sum(bool(r['diagnosis_correct']) for r in ok):5d} "
                   f"{(sum(lat) / len(lat) if lat else float('nan')):6.1f} "
                   f"{(sum(pt) / len(pt) if pt else float('nan')):10.0f}")
    ok_all = [r for r in records if r["status"] == "ok"]
    out += ["", f"records={len(records)} ok={len(ok_all)} errors={len(records) - len(ok_all)}"]
    if sample_ids:
        cov = coverage_report(sample_ids, records)
        out.append(f"coverage: {cov['recorded_cells']}/{cov['expected_cells']} exact={cov['exact_coverage']} all_ok={cov['all_ok']}")
    return "\n".join(out)


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


def resolve_excluded(payload: dict[str, Any], manifest_path: Optional[str]) -> list[str]:
    """Review-sample IDs to keep out of prompt development: the manifest if present, else re-derived."""
    if manifest_path and Path(manifest_path).is_file():
        return list(json.loads(Path(manifest_path).read_text(encoding="utf-8"))["sample_ids"])
    from src.evaluation.build_phase0_review_sheet import select_sample
    return select_sample(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    sub = parser.add_subparsers(dest="command", required=True)
    inter = sub.add_parser("interventions")
    inter.add_argument("--sample-id", required=True)
    inter.add_argument("--condition", choices=CONDITIONS, default="D_highC_highP")
    inter.add_argument("--cluster-unit")
    inter.add_argument("--output", default="results/phase0/interventions.json")
    sweep_p = sub.add_parser("baseline-sweep", help="prompt development; never use the reviewed sample")
    sweep_p.add_argument("--dev-count", type=int, default=8)
    sweep_p.add_argument("--exclude-manifest", default=None,
                         help="JSON with sample_ids to exclude; default: re-derive the review sample")
    sweep_p.add_argument("--condition", choices=CONDITIONS, default="D_highC_highP")
    sweep_p.add_argument("--output", default="results/phase0/baseline_sweep.json")
    summ = sub.add_parser("summarize", help="print a compact table from a saved result; loads no model")
    summ.add_argument("--input", required=True)
    fact = sub.add_parser("factorial")
    fact.add_argument("--sample-manifest")
    fact.add_argument("--sample-ids", nargs="*")
    fact.add_argument("--allow-unreviewed", action="store_true")
    fact.add_argument("--output", default="results/phase0/factorial.jsonl")
    fact.add_argument("--coverage-output", default="results/phase0/factorial_coverage.json")
    args = parser.parse_args()

    if args.command == "summarize":
        path = Path(args.input)
        if not path.is_file():
            raise SystemExit(f"{path} not found; nothing to summarize (has the run happened?)")
        if path.suffix == ".jsonl":
            records = read_records(path)
            if not records:
                raise SystemExit(f"{path} has no records")
            print(summarize_factorial(records))
        else:
            report = json.loads(path.read_text(encoding="utf-8"))
            print(summarize_interventions(report) if "trials" in report and "repair_summary" in report
                  else json.dumps(report.get("summary", report), indent=2))
        return

    bundle_path = Path(args.bundle)
    payload = load_bundle(bundle_path)
    from src.evaluation.build_bundles_v2 import assert_current_bundle
    assert_current_bundle(payload)

    # Resolve and validate every input BEFORE the slow model load.
    plan: dict[str, Any] = {}
    if args.command == "baseline-sweep":
        from src.evaluation.build_phase0_review_sheet import select_dev
        plan["excluded"] = resolve_excluded(payload, args.exclude_manifest)
        plan["dev_ids"] = select_dev(payload, plan["excluded"], args.dev_count)
    elif args.command == "interventions":
        question = get_question(payload, args.sample_id)
        docs = question["conditions"][args.condition]["documents"]
        unit = args.cluster_unit or default_cluster_unit(docs)
        if sum(d["evidence_unit_id"] == unit for d in docs) < 2:
            raise SystemExit(f"Cluster family {unit!r} needs >= 2 documents in {args.condition}")
    else:
        plan["sample_ids"], plan["review"] = resolve_samples(args)
        for sid in plan["sample_ids"]:
            get_question(payload, sid)

    from src.models.local_qwen import LocalQwen  # lazy: keeps torch out of tests

    model = LocalQwen()
    meta = {"bundle": str(bundle_path), "bundle_sha256": sha256_file(bundle_path),
            "model_id": getattr(model, "model_id", None), "prompts": prompt_manifest(),
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat()}
    if args.command == "baseline-sweep":
        dev_ids, excl = plan["dev_ids"], plan["excluded"]
        report = {**meta, "dev_sample_ids": dev_ids, "excluded_review_sample": excl,
                  **run_baseline_sweep(payload, dev_ids, args.condition, model)}
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report["summary"]))
        for t in report["trials"]:
            print(f"  {t['sample_id']:9s} gold={t['gold_answer']!r:22} out={(t.get('scoring') or {}).get('raw_output','ERR')[:70]!r}")
            for st in t["stage_records"]:
                if st["role"] == "specialist":
                    print(f"      {st['stage']}: {st.get('raw_output', st.get('error', ''))[:110]!r}")
        return
    if args.command == "interventions":
        report = {**meta, **run_interventions(payload, args.sample_id, args.condition, model,
                                              args.cluster_unit)}
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(report['trials'])} trials to {out}")
        print(json.dumps(report["repair_summary"]))
    else:
        sample_ids, review = plan["sample_ids"], plan["review"]
        coverage = run_factorial(payload, sample_ids, model, Path(args.output))
        coverage.update(meta, **review)
        Path(args.coverage_output).write_text(json.dumps(coverage, indent=2), encoding="utf-8")
        print(f"Coverage: {coverage['recorded_cells']}/{coverage['expected_cells']} "
              f"exact={coverage['exact_coverage']} errors={len(coverage['error_cells'])}")
        if not coverage["exact_coverage"]:
            raise SystemExit("Factorial coverage is not exact; see coverage report")


if __name__ == "__main__":
    main()
