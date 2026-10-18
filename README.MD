# Dependence-Aware Fault Diagnosis in Parallel RAG — Phase 0

This repository contains (1) a deterministic controlled fixture for testing the 4-condition × 4-fault engineering design and interventions, and (2) local Qwen integration code plus a HotpotQA smoke test.

## Important dataset distinction

- `data/seed_sample.json` is **fictional and synthetic**. It is only for pipeline plumbing, controlled fault injection, and factorial coverage. Its source clusters are assigned experimental labels, not verified provenance.
- `data/raw/hotpot_dev_distractor_v1.json` should contain the real HotpotQA dev distractor file for the real-model smoke test. The dataset itself is not redistributed in this archive. Keep raw benchmark data unmodified.
- The 3-example HotpotQA smoke run is not a benchmark result and does not evaluate retrieval, because it supplies the provided context directly.

## Commands

Activate the existing virtual environment, then run from the repository root:

```powershell
python -m src.evaluation.validate_dataset
python -m pytest -q
python -m src.evaluation.run_pilot
python -m src.evaluation.hotpotqa_runner
python scripts/run_qwen_factorial.py
```

The last two commands load Qwen locally and require the existing CUDA/Transformers/bitsandbytes setup. The factorial run performs 16 controlled real-model executions (4 content/provenance conditions × 4 fault settings) and may take several minutes on a 6 GB GPU.

## Interpretation

- `AnswerScorer` is shared by baseline evaluation, item-level LOO, cluster ablation, and the Qwen pipeline.
- A repair requires a failed baseline and a correct post-intervention answer. Ordinary ablation correctness effects are not automatically repairs.
- The diagnosis function receives observation-only trace data; injected labels are attached to the evaluator trace only after diagnosis. The evaluator then scores predicted labels against injected labels.
- Token counts and runtime are reported only when measured. No API cost or token estimates are fabricated.
- The synthetic pilot can verify code paths; it cannot establish the research hypothesis or support a claim about real-world source lineage.
