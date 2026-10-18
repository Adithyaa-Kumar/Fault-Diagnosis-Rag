"""Non-destructive Phase 0 balanced-design preflight.

It deliberately does not fabricate evidence documents or silently relabel provenance.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CONDITIONS = {
    "A_lowC_lowP": ("low", "low"),
    "B_highC_lowP": ("high", "low"),
    "C_lowC_highP": ("low", "high"),
    "D_highC_highP": ("high", "high"),
}

def build_design_audit(bundle_path: Path, output_dir: Path) -> dict:
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    questions = data.get("questions", [])
    totals = Counter()
    per_sample = {}
    issues = []
    for q in questions:
        sid = q.get("sample_id", "<missing>")
        conds = q.get("conditions", {})
        counts = {}
        for name, (expected_c, expected_p) in CONDITIONS.items():
            cell = conds.get(name, {})
            docs = cell.get("documents", [])
            counts[name] = len(docs)
            totals[name] += len(docs)
            if cell.get("redundancy") != expected_c or cell.get("dependence") != expected_p:
                issues.append(f"{sid}/{name}: factor labels do not match expected design.")
        per_sample[sid] = counts
        if len(set(counts.values())) > 1:
            issues.append(f"{sid}: document counts differ across conditions: {counts}")
    flat_counts = [n for cells in per_sample.values() for n in cells.values()]
    equal_global = bool(flat_counts) and len(set(flat_counts)) == 1
    evidence_counts = [len(q.get("source_evidence", [])) for q in questions]
    evidence_count_constant = bool(evidence_counts) and len(set(evidence_counts)) == 1
    warnings = [
        "This audit does not verify article contents, claim support, or gold-answer validity.",
        "Current lineage IDs are synthetic; original provenance is unknown.",
        "Do not equalize counts by blindly duplicating documents: that would change the redundancy manipulation.",
        "A defensible balanced factorial needs source-grounded distinct passages for low redundancy and repeated/overlapping evidence for high redundancy, with equal document counts and equivalent claim coverage in all four cells.",
        "Describe provenance dependence as a synthetic intervention unless real source lineage is verified.",
        "Gold-answer concerns for mh_0174, mh_0533, and mh_0639 remain unresolved pending full-source review.",
    ]
    result = {
        "report_title": "Phase 0 Balanced Experimental Design Preflight",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_bundle": str(bundle_path),
        "input_modified": False,
        "sample_count": len(questions),
        "document_totals_by_condition": dict(totals),
        "document_counts_equal_within_every_sample": all(len(set(c.values())) <= 1 for c in per_sample.values()),
        "document_counts_equal_globally": equal_global,
        "source_evidence_count_is_constant": evidence_count_constant,
        "structural_issues": issues,
        "warnings": warnings,
        "candidate_bundle_created": False,
        "gate_decision": "NOT CLEARED",
        "reason": "The current excerpts do not justify fabricating a balanced v2 bundle. Human source review and source-grounded construction are required.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "balanced_design_preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Phase 0 Balanced Experimental Design Preflight", "",
        f"- Samples: {len(questions)}",
        f"- Document totals by condition: `{json.dumps(dict(totals), sort_keys=True)}`",
        f"- Counts equal within every sample: `{result['document_counts_equal_within_every_sample']}`",
        f"- Counts equal globally: `{equal_global}`",
        f"- Source-evidence count constant: `{evidence_count_constant}`",
        "- Candidate v2 bundle created: `No`", "", "## Structural issues", "",
    ]
    lines.extend([f"- {x}" for x in issues] if issues else ["- None detected by this preflight."])
    lines += ["", "## Required safeguards", ""] + [f"- {x}" for x in warnings]
    lines += ["", "## Gate decision", "", "**NOT CLEARED.** This is intentional: arbitrary duplicate padding would invalidate the low-redundancy condition, and fabricated paraphrases would not be source-grounded.", ""]
    (output_dir / "balanced_design_preflight.md").write_text("\n".join(lines), encoding="utf-8")
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", default="data/processed/multihoprag/controlled_bundles_30.json")
    parser.add_argument("--output-dir", default="results/multihoprag")
    args = parser.parse_args()
    r = build_design_audit(Path(args.bundles), Path(args.output_dir))
    print("PHASE 0 BALANCED DESIGN PREFLIGHT")
    print(f"Samples: {r['sample_count']}")
    print(f"Document totals: {r['document_totals_by_condition']}")
    print(f"Structural issues: {len(r['structural_issues'])}")
    print("Candidate bundle created: NO (source-grounded construction required)")
    print("Gate decision: NOT CLEARED")
    print(f"Report written to {Path(args.output_dir) / 'balanced_design_preflight.md'}")
if __name__ == "__main__": main()
