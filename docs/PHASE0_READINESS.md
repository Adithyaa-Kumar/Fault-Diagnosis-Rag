# Phase 0 Readiness Gate — Attribution Ambiguity in Parallel Multi-Agent RAG

## Fixed project decisions

- **Research dataset for Phase 1:** MultiHop-RAG only.
- **Synthetic engineering fixture:** `data/seed_sample.json`; it is not real-world evidence and cannot support the research hypothesis.
- **Model:** `Qwen/Qwen2.5-7B-Instruct`, local 4-bit quantization on the RTX 3050 6 GB.
- **Architecture:** three isolated specialist contexts followed by one arbiter; calls are sequential due to local VRAM constraints.
- **Research comparison:** item-level LOO versus provenance-aware cluster-level counterfactual interventions.

Do not switch back to HotpotQA. Preserve the raw MultiHop-RAG JSON files. The fact that an evidence URL matches a corpus URL does not establish source independence or dependence. Unknown lineage stays unknown. Any controlled provenance augmentation must record parent items and transformations.

## Phase 0 checklist

- [x] One canonical `AnswerScorer` API is used by the baseline, LOO, and cluster-ablation paths. Exact match, token F1, and correctness are derived by the same scorer.
- [x] Repair predicate is centralized as `AnswerScorer.is_repair`: baseline incorrect AND intervention correct.
- [x] Repair predicate is exercised through actual LOO and cluster-removal interventions in a deterministic intervention fixture.
- [x] Diagnoser accepts only `DiagnosticObservation`; the observation schema excludes gold answer, required claims, and injected fault label. Predicted labels are compared with injected labels only in the evaluator.
- [x] Synthetic fixture validator checks all four factorial conditions, required-claim coverage, unique claim IDs, `claim_ids` consistency, and cluster design.
- [x] Qwen pipeline distributes every input document once across the three specialists rather than silently dropping documents after the first three.
- [x] Trace semantics distinguish pre-injection retrieved documents from the actual documents assigned to the specialist, so retrieval corruption is observable without exposing the injected label.
- [x] Qwen trace records model ID, model-call count, and actual prompt/generated token counts when the model wrapper returns them. Missing measurements remain `null`; no token estimates are fabricated.
- [x] Tests cover all 4 conditions x 4 fault types, scorer edge cases, diagnosis label isolation, trace integrity, document assignment, measured-token aggregation, and actual intervention repair metrics.
- [x] Synthetic fixture claims and claim IDs are explicitly marked fictional and validated as internally consistent.
- [ ] **Final local GPU gate:** rerun the updated Qwen 4x4 factorial once after applying this overlay. The earlier 16/16 run used pre-patch code and does not validate the new trace/token fields.

## Verification already performed on this patch overlay

- `python -m pytest -q`: **20 passed**.
- `python -m src.evaluation.validate_dataset`: **passed**.
- `python -m src.evaluation.run_pilot`: **16/16 mock factorial combinations**; heuristic accuracy is engineering-only and not a research result.

These checks were run in the patch workspace, not on the user's Windows machine. The local GPU gate must be run in the project venv after applying the overlay.

## One-time final Phase 0 command sequence (PowerShell)

```powershell
python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
python -m src.evaluation.validate_dataset
if ($LASTEXITCODE -ne 0) { throw "Dataset validation failed" }
python -m src.evaluation.run_qwen_factorial
if ($LASTEXITCODE -ne 0) { throw "Real-Qwen factorial failed" }
```

The last command loads Qwen and runs 16 condition/fault trajectories. It writes `results/qwen_factorial_phase0.json`, including actual measured token totals when available.

## Phase 0 exit criteria

Exit Phase 0 when the command sequence passes, the Qwen report shows 16/16 combinations and contains measured counts (or `null` where unavailable), and no diagnostic input contains gold answers or injected labels. Do not rerun this checklist unless code changes affect these contracts.

## Phase 1 begins immediately after the gate

Use the frozen MultiHop-RAG manifest, evidence-backed questions only, and a deterministic sample seed. Run the defined redundancy x provenance-dependence design and compare item-level LOO with cluster-level interventions against the same baseline trajectory. Provenance cluster labels must be supported by documented lineage or explicitly constructed with recorded transformations. Report repair among failed baselines, diagnosis accuracy against held-out injected labels, measured token/model-call cost, uncertainty across questions, and limitations. Do not treat the synthetic fixture's scores as evidence for the hypothesis.
