# Phase 0: LLM integration — execute in this order

## Dataset roles

- `data/seed_sample.json` is explicitly a **fictional controlled fixture**. It is used only to test the four C/P conditions, controlled fault injections, leakage isolation, and interventions. Its source-cluster IDs are assigned by experimental design, not claimed as real provenance.
- `data/raw/hotpot_dev_distractor_v1.json` is the real HotpotQA dev distractor dataset used for a small real-model integration smoke test. The smoke test reports normalized exact match and token F1; three examples are a pipeline check, not a benchmark result.

## Commands (Windows PowerShell, repository root, activated venv)

```powershell
python -m src.evaluation.validate_dataset
python -m pytest -q
python -m src.evaluation.run_pilot
python -m src.evaluation.hotpotqa_runner
python scripts/run_qwen_factorial.py
```

- The first three commands validate the controlled fixture and deterministic engineering baseline.
- `hotpotqa_runner` loads Qwen once and evaluates three valid HotpotQA examples. Measured tokenizer counts and elapsed time are recorded; no estimates are fabricated.
- `run_qwen_factorial.py` uses the real local Qwen model across 4 conditions × 4 fault settings. It is compute-intensive and may take several minutes. It writes `results/qwen_factorial_phase0.json` and individual traces under `logs/`.
- The factorial fixture is still synthetic. Do not describe its accuracy as research evidence.
- LOO and cluster ablation continue to use the same `AnswerScorer` class. Their results are correctness effects; only an incorrect baseline that becomes correct after a defined intervention counts as a repair.
- The diagnostic function receives `DiagnosticObservation` only; it does not receive the injected label or gold answer. The evaluator compares the resulting prediction with `injected_fault_layer` after diagnosis.

## Current limitation

This package integrates the local Qwen wrapper into the controlled pipeline and provides a HotpotQA smoke path. It does not claim to establish the research hypothesis or verify real-world provenance. The HotpotQA smoke path uses the benchmark's provided contexts and therefore does not evaluate retrieval quality.
