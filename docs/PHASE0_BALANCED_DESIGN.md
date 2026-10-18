# Phase 0 balanced-design preflight

Merge the ZIP into the existing repository root (`C:\\fault-diagnosis-rag`). This adds an evaluation module, tests, and this note. It does not overwrite existing data or change the pipeline.

Run:

```powershell
python -m pytest -q
python -m src.evaluation.prepare_balanced_design
```

Outputs:
- `results/multihoprag/balanced_design_preflight.md`
- `results/multihoprag/balanced_design_preflight.json`

## Why it does not generate v2 bundles

The current sample has only a few source-evidence excerpts per question. Padding low-redundancy cells with duplicates would make them redundant; inventing paraphrases would fabricate evidence; and labeling URLs as independent lineage would make unsupported provenance claims. This preflight therefore preserves all current artifacts and refuses to call the factorial balanced or valid.

A defensible next design needs a reviewed set of source-grounded passages per question, a fixed document count per cell, matched claim coverage, an explicit synthetic dependence intervention, and tests demonstrating that only the intended factor changes. This tool is a diagnostic preflight, not semantic evidence verification.
