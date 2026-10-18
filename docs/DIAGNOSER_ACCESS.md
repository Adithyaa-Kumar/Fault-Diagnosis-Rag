# Diagnoser access contract (MultiHop path)

`src.pipeline.multihop_faults.diagnose(observation)` is the only diagnosis entry point. It receives a `MultiHopObservation` (`extra="forbid"`), which contains exactly:

| Field | Content |
|---|---|
| `query` | question text |
| `specialists[i].retrieved_doc_ids` | document IDs retrieval returned for specialist i |
| `specialists[i].input_docs` | `doc_id` + `content` the specialist actually saw |
| `specialists[i].output` | specialist output |
| `arbiter_inputs` | `agent_id` + `output` of the reports the arbiter actually received |
| `final_verdict` | arbiter output |

Not accessible: gold answer, required claims, injected layer, injection record, condition name, lineage IDs, `role` (claim/filler), review status.

## Layer definitions (consistency violations)

| Layer | Injection (evaluator side) | Observable signature |
|---|---|---|
| retrieval | one claim document replaced by a pool filler before the specialist | `retrieved_doc_ids` != IDs of `input_docs` |
| specialist | one specialist output replaced by an ungrounded corpus sentence | output content-token novelty vs (query + its input docs) > 0.5; abstentions exempt |
| arbiter | arbiter receives all reports but one | `arbiter_inputs` != specialist outputs |
| none | nothing | all three checks pass |

Checks run in upstream-to-downstream order (retrieval, specialist, arbiter).

## Limits

- These signatures are defined by the injection mechanism, so accuracy near 100% on injected faults is expected and is a plumbing and leakage proof, not a research result.
- The specialist check is a lexical heuristic. Its false-positive rate on `none` runs with the real model is unmeasured until the real factorial runs; report it there.
- An answer change after deleting evidence is not used as evidence of the fault layer.
- `apply_injection=False` in `execute_fault_trial` is a test-only knob that declares a label without mutating anything.
