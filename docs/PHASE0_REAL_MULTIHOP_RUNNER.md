# Phase 0: real MultiHop-RAG intervention runner

## What this adds

`src/evaluation/run_multihop_qwen.py` is separate from the synthetic `seed_sample.json` pipeline. It reads `controlled_bundles_30.json`, selects exactly one `sample_id` and condition, runs the existing `LocalQwen` model, and executes three trials with the same question and gold answer:

1. Baseline evidence.
2. One-document removal (records the removed `doc_id`).
3. One controlled `evidence_unit_id` family removal (records every removed `doc_id`).

Every trial saves each specialist/arbiter prompt, raw stage output, input document IDs, measured token counts where returned, stage latency, final raw output, extracted answer, score, end-to-end latency, and bundle SHA-256. It pairs the gold answer by `sample_id` from the same bundle question record. `is_repair` is true only for an incorrect baseline followed by a correct intervention.

## Run on the existing Windows project root

Activate the existing virtual environment and run from `C:\fault-diagnosis-rag`:

```powershell
python -m src.evaluation.run_multihop_qwen --sample-id mh_0174 --condition A_lowC_lowP
```

Default output:

`results/multihoprag/real_qwen_interventions.json`

To choose another sample/condition or output path, use `--sample-id`, `--condition`, and `--output`. This run invokes the local Qwen model and consumes GPU time. It is intentionally a one-question intervention feasibility run, not the factorial.

## Important interpretation limits

- The current controlled bundle uses synthetic lineage IDs. They do not prove real-world provenance dependence.
- Current high-redundancy conditions have twice the documents of low-redundancy conditions (6 versus 3 for the inspected sample). Therefore redundancy is confounded with document count. Do not use these conditions for a causal redundancy/dependence claim or a research-confirmatory factorial until the design is repaired.
- Evidence-unit removal targets copies associated with one controlled evidence unit. It is not represented as a verified real-world provenance-cluster intervention.
- The runner records outputs and effects; answer changes alone do not establish the injected fault layer.
- This environment cannot verify a Qwen run on the user's RTX 3050. Run the command locally and preserve the generated JSON as evidence.

## Verification performed for this patch

- `python -m pytest -q`: 32 passed in the isolated archive review workspace.
- `python -m src.evaluation.run_multihop_qwen --help`: passed without loading the model.
- `python -m src.evaluation.validate_dataset`: passed for the existing synthetic fixture validator. This does not validate the scientific quality of the MultiHop-RAG controlled conditions.

The above are local review-workspace checks, not claims that the user's Windows working tree was edited or that GPU inference ran.


## Evidence-context and intervention guard (follow-up patch)

The runner now includes each document's article title, publisher, publication timestamp, URL, and evidence text in the actual specialist prompt. These metadata fields help the model identify which article a fact came from, but do not establish that the fact supports the gold answer or that sources are independent.

Evidence-unit removal now fails fast if the selected evidence-unit family contains only one document, because that would duplicate item removal rather than test a cluster-level intervention. For the current synthetic bundles, a high-redundancy condition (B or D) has duplicate families, but the document-count confound remains and the synthetic lineage IDs are not verified provenance. Do not run the factorial until the experimental design and evidence support have been reviewed.
