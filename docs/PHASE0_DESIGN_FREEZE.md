# Phase 0 design freeze

Status: **FROZEN v1**. Edit only by appending to the changelog at the bottom.
Scope: MultiHop-RAG only. Synthetic fixture (`data/seed_sample.json`) stays separate.

## D-1 Document count and claim coverage

- K = number of evidence items of a question (2-4 in `pilot_30.json`). All K are required claims.
- Every condition cell of a sample has exactly **N = 2K** documents.
- Every cell contains the same K claim passages (`copy0`), so claim coverage is identical across cells.
- Passage order is one seeded permutation of the 2K slots per sample, identical across the four cells.

## D-2 Redundancy (content)

| Level | Slots filled with |
|---|---|
| low  | K claim passages (`copy0`) + K grounded **filler** passages |
| high | K claim passages (`copy0`) + K overlapping claim passages (`copy1`) |

- Claim passages are verbatim windows of the raw corpus article body that contain the evidence `fact`. `copy0` and `copy1` are distinct, overlapping windows of the same article. No paraphrase, no invented text.
- Filler passages are verbatim windows of the same article that do not overlap any claim window, contain no evidence fact, and have the lowest lexical overlap with query + gold answer + facts. This is a lexical screen only: **every filler needs human review** (`filler_review_required: true`).

## D-3 Dependence (provenance, controlled)

- High: all claim passages of a sample share one lineage ID `synthetic_shared_parent_<qid>`.
- Low: every claim passage has its own lineage ID.
- Filler passages always have their own lineage ID.
- `original_provenance_status` stays `unknown` for every document. Lineage IDs are experimental assignments (`lineage_is_synthetic: true`), not source lineage.
- Known limitation: in low-dependence cells `copy0`/`copy1` come from the same article URL (recorded as `parent_source_url`). Their "independence" is a label only. Real cross-source independence cannot be established from this corpus without reviewed matches; Phase 1 must not claim it.

## D-4 Intervention groups (recorded separately)

- **Claim family** = all documents sharing `evidence_unit_id`. In B/D a family has 2 documents. Phase 0 cluster intervention = removal of one claim family in B or D (explicitly defined, >= 2 documents, never all evidence).
- **Lineage cluster** = all documents sharing `experimental_lineage_id`. In C/D this is all claim passages, so removal is expected to be degenerate. Implement and report it, do not use it as the Phase 0 cluster gate.

## D-5 Fault layers

Defined and injected independently of diagnosis; the injected label lives only in the evaluator record.

- `retrieval`: one required claim passage is replaced by a filler before specialists see it.
- `specialist`: one specialist output is replaced by a recorded wrong-claim output.
- `arbiter`: the arbiter receives the specialist outputs with one dropped; the drop is recorded.
- `none`: no injection.

## D-6 Scoring

`src.evaluation.multihop_scoring.score_multihop_answer` is the only scorer for MultiHop runs. Abstentions extract to `""`, score 0, and are tagged as abstentions in reports.

## D-7 Sample

- M = 6 reviewed questions, 2 per question type, drawn with the pilot seed.
- Excluded until full-source review clears them: `mh_0174`, `mh_0533`, `mh_0639`.
- A question enters the sample only if `build_bundles_v2` marks it feasible and the human review decision is `accept`.

## Changelog

- v1: initial freeze.
