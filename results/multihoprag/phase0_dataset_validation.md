# Phase 0 MultiHopRAG Dataset Validation

## Scope

Offline structural consistency and experimental-design risk checks only. This report does not verify article contents, factual correctness, or gold-answer validity.

## Counts

- Pilot samples: 30
- Annotation samples: 30
- Controlled-bundle samples: 30
- Review CSV rows / columns: 98 / 20
- Reviewer decisions: `{"revise_pending_verification": 98}`
- Canonical annotation statuses: `{"pending": 30}`
- Documents across four conditions: 504
- Documents with synthetic lineage IDs: 504
- Documents with unknown original provenance: 504

## Errors

- None found by structural checks.

## Warnings / Required follow-up

- Reviewer decisions are blank or pending; no human-validated evidence labels can be claimed.
- Canonical claim annotations still pending: 30 of 30.
- Gold-answer review required for 3 samples with at least one draft unsupported/contradictory claim: mh_0174, mh_0533, mh_0639.
- mh_0174: its first cited excerpt says sportsbooks earn profit built into odds/lines; it does not explicitly establish that the named article states sportsbooks adjust odds. Gold answer 'Yes' requires full-source review.
- Document-count confound: condition document counts differ for 30 samples: mh_0174, mh_0533, mh_0639, mh_0791, mh_0836, mh_1651, mh_1755, mh_1893, mh_2211, mh_2543, mh_0210, mh_0389, mh_0669, mh_1101, mh_1480, mh_1625, mh_1756, mh_2496, mh_2505, mh_2539, mh_0232, mh_0373, mh_0567, mh_0620, mh_0982, mh_1017, mh_1089, mh_2247, mh_2281, mh_2474.
- Synthetic lineage IDs: 504/504 documents; these encode experimental assignments, not verified real-world provenance.
- Original provenance marked unknown for 504/504 documents; do not interpret publisher/URL similarity as proof of dependence.

## Gate decision

**NOT CLEARED for the real factorial.** Human source/evidence review and experimental-design correction remain required. A clean structural audit is not evidence that claims or gold answers are true.
