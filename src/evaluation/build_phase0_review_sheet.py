"""Select the Phase 0 sample (2 per question type) and write a human review sheet.

  python -m src.evaluation.build_phase0_review_sheet

Outputs
  data/processed/multihoprag/phase0_review_sheet.md      read and tick this
  data/processed/multihoprag/phase0_sample_DRAFT.json    reviewed=false; rename after review

Selection is seeded and deterministic. Nothing here validates evidence: the sheet exists so
a human can. Claims shown are DRAFTS taken verbatim from each evidence fact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

BUNDLE = Path("data/processed/multihoprag/controlled_bundles_v2.json")
SHEET = Path("data/processed/multihoprag/phase0_review_sheet.md")
DRAFT = Path("data/processed/multihoprag/phase0_sample_DRAFT.json")
EXCLUDED = ("mh_0174", "mh_0533", "mh_0639")  # gold flagged as possibly unsupported
# Questions the prompts were tuned on (mh-v2-prompts-3 baseline sweep). Never review-sample material.
PROMPT_DEV_USED = ("mh_1893", "mh_2539", "mh_0232", "mh_2211", "mh_2496", "mh_0620", "mh_1651", "mh_1089")
PER_TYPE = 2
MAX_K = 3  # keeps specialist prompts short on a 6 GB GPU


def rank_key(seed: Any, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def select_sample(bundle: dict[str, Any], per_type: int = PER_TYPE, rejected=()) -> list[str]:
    """Seeded top-`per_type` per question type; rejected IDs are skipped, so the next-ranked
    candidate of the same type replaces them and every other pick stays unchanged."""
    banned = set(EXCLUDED) | set(PROMPT_DEV_USED) | set(rejected)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in bundle["questions"]:
        if q["sample_id"] in banned or q["required_claim_count"] > MAX_K:
            continue
        by_type[q["question_type"]].append(q)
    chosen: list[str] = []
    for qtype in sorted(by_type):
        ranked = sorted(by_type[qtype], key=lambda q: rank_key(bundle["seed"], q["sample_id"]))
        if len(ranked) < per_type:
            raise ValueError(f"Only {len(ranked)} eligible {qtype} questions")
        chosen += [q["sample_id"] for q in ranked[:per_type]]
    return chosen


def select_dev(bundle: dict[str, Any], exclude_ids, n: int = 8) -> list[str]:
    """Seeded dev questions for prompt tuning, round-robin over types, never the review sample."""
    banned = set(exclude_ids) | set(EXCLUDED)
    by_type: dict[str, list[str]] = defaultdict(list)
    for q in sorted(bundle["questions"], key=lambda q: rank_key(f"dev:{bundle['seed']}", q["sample_id"])):
        if q["sample_id"] not in banned and q["required_claim_count"] <= MAX_K:
            by_type[q["question_type"]].append(q["sample_id"])
    out: list[str] = []
    while len(out) < n and any(by_type.values()):
        for t in sorted(by_type):
            if by_type[t] and len(out) < n:
                out.append(by_type[t].pop(0))
    return out


def render_sheet(bundle: dict[str, Any], sample_ids: list[str]) -> str:
    lines = [
        "# Phase 0 review sheet", "",
        "Tick `[x]` only after reading the SOURCE ARTICLE text, not just the excerpt. "
        "Claims below are drafts copied from each evidence fact; edit them if wrong.", "",
        f"Bundle seed: `{bundle['seed']}`  |  Sample: {', '.join(sample_ids)}", ""]
    by_id = {q["sample_id"]: q for q in bundle["questions"]}
    for sid in sample_ids:
        q = by_id[sid]
        lines += [f"## {sid} ({q['question_type']}, K={q['required_claim_count']})", "",
                  f"**Question:** {q['query']}", "", f"**Gold answer:** {q['answer']}", ""]
        docs = q["conditions"]["D_highC_highP"]["documents"]
        for ev_i, ev in enumerate(q["source_evidence"]):
            unit = f"{sid}_ev_{ev_i:02d}"
            lines += [f"### Claim C{ev_i + 1} (draft, from evidence fact)", "",
                      f"> {ev.get('fact', '').strip()}", "",
                      f"Source: {ev.get('source')} - {ev.get('title')}", ""]
            for d in docs:
                if d["evidence_unit_id"] == unit:
                    lines += [f"- Passage `copy{d['doc_id'][-1]}` (chars "
                              f"{d['source_char_start']}-{d['source_char_end']}):",
                              f"  > {d['content'].strip()}"]
            lines += ["", f"- [ ] C{ev_i + 1} is supported by the source article",
                      f"- [ ] Both passages contain the fact verbatim and nothing contradicting it", ""]
        lines += ["### Filler passages (must NOT support or contradict the gold answer)", ""]
        for p in q["filler_pool"]:
            lines += [f"- Filler for {p['slot_unit_id']} (lexical overlap {p['lexical_overlap']}):",
                      f"  > {p['content'].strip()}",
                      f"  - [ ] neutral: says nothing about the question or the gold answer", ""]
        lines += ["### Decision", "",
                  "- [ ] Gold answer is entailed by claims C1..C" + str(q["required_claim_count"]) + " together",
                  "- [ ] ACCEPT this question   - [ ] REJECT (reason: ______)", "", "---", ""]
    return "\n".join(lines)


def draft_manifest(bundle: dict[str, Any], sample_ids: list[str], rejected=()) -> dict[str, Any]:
    return {"reviewed": False, "reviewer": None, "sample_ids": sample_ids,
            "rejected_in_review": list(rejected),
            "bundle_seed": bundle["seed"], "bundle_input_sha256": bundle["input_sha256"],
            "excluded_unreviewed": list(EXCLUDED),
            "instructions": "After completing the review sheet set reviewed=true, fill reviewer, "
                            "remove rejected IDs, and save as phase0_sample.json."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(BUNDLE))
    parser.add_argument("--reject", nargs="*", default=[],
                        help="IDs that failed review; replaced by the next-ranked question of the same type")
    args = parser.parse_args()
    bundle = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
    from src.evaluation.build_bundles_v2 import assert_current_bundle
    assert_current_bundle(bundle)
    ids = select_sample(bundle, rejected=args.reject)
    SHEET.write_text(render_sheet(bundle, ids), encoding="utf-8")
    DRAFT.write_text(json.dumps(draft_manifest(bundle, ids, args.reject), indent=2), encoding="utf-8")
    print("Sample:", ids)
    print("Wrote", SHEET, "and", DRAFT)


if __name__ == "__main__":
    main()
