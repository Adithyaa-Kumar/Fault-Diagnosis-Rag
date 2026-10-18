# Phase 0 runbook (PowerShell, from repo root, venv active)

```powershell
python -m pytest -q
python -m src.evaluation.build_bundles_v2

# Step 3 (GPU): baseline + full leave-one-out + one claim-family removal, one question.
# Use a high-redundancy condition (B or D); A/C have no multi-document family and are refused.
python -m src.evaluation.run_multihop_v2 interventions --sample-id <ID> --condition D_highC_highP

# Step 5 (GPU): 4 conditions x 4 fault layers x reviewed samples. Checkpointed; rerun to resume.
python -m src.evaluation.run_multihop_v2 factorial --sample-manifest data/processed/multihoprag/phase0_sample.json
```

`phase0_sample.json` is written by hand after the evidence/filler review:

```json
{"reviewed": true, "reviewer": "<name>", "sample_ids": ["mh_...", "..."]}
```

Without it, `factorial` refuses to run unless `--allow-unreviewed --sample-ids ...` is passed (recorded as `reviewed: false`).

Outputs: `results/phase0/interventions.json`, `results/phase0/factorial.jsonl`, `results/phase0/factorial_coverage.json`.

Notes
- Prompts are versioned and hashed (`prompt_manifest()`); changing them changes the hash recorded in every output.
- The model sees neutral document aliases, never condition, copy, role or lineage strings.
- No timeout is configured for local generation; failures are recorded as `status: error` and listed in the coverage report, never dropped.
- Repair success is `null` (undefined) when the baseline is correct, not 0.
