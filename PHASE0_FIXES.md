# Phase 0 Fixes Included

This archive contains an implementation pass for the seven outstanding Phase 0 requirements.

## Files changed
- `data/seed_sample.json`: fixed question, gold answer, three distinct claim IDs, required-claim manifest, and four controlled content-redundancy/provenance conditions.
- `src/evaluation/scoring.py`: one canonical deterministic answer scorer with required-claim coverage and edge-case handling.
- `src/schema.py`: added `DiagnosticObservation`, which does not contain injected fault labels; removed per-specialist ground-truth fault metadata.
- `src/pipeline/agent_runner.py`: diagnosis consumes only observations; removed the arbiter label leak; records actual specialist inputs; supports in-memory intervention inputs.
- `src/attribution/loo.py`: canonical scoring and correctness-effect naming; no predictable temporary files.
- `src/attribution/cluster_ablation.py`: canonical scoring, corrected repair predicate, failed-baseline denominator and `N/A` for zero eligible failures.
- `src/evaluation/run_pilot.py`: all 4 conditions × 4 fault labels, confusion matrix, no fabricated token estimates, JSON report.
- `src/evaluation/validate_dataset.py`: standalone design and evidence-coverage validation.
- `tests/`: expanded scoring, factorial, diagnosis, repair, dataset and trace-integrity coverage.

## Verified commands
Run from the project directory:

```bash
python -m src.evaluation.validate_dataset
python -m pytest -q
python -m src.evaluation.run_pilot
```

Latest local verification: dataset validator passed; **12 tests passed**; pilot executed **16/16** condition/fault combinations.

## Important limitation
The pilot uses a deterministic mock pipeline and a transparent heuristic diagnostic function. The displayed 100% mock heuristic diagnosis accuracy only confirms that the current controlled fault signatures can be detected; it is **not** a meaningful real-LLM research result and must not be presented as one. The cluster-ablation repair rate is `N/A` for the current no-fault baseline because there are no eligible failed baseline trajectories. Phase 0 should not be considered scientifically validated until trace review and the research protocol are approved.
