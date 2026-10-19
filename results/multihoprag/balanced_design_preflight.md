# Phase 0 Balanced Experimental Design Preflight

- Samples: 30
- Document totals by condition: `{"A_lowC_lowP": 168, "B_highC_lowP": 168, "C_lowC_highP": 168, "D_highC_highP": 168}`
- Counts equal within every sample: `True`
- Counts equal globally: `False`
- Source-evidence count constant: `False`
- Candidate v2 bundle created: `No`

## Structural issues

- None detected by this preflight.

## Required safeguards

- This audit does not verify article contents, claim support, or gold-answer validity.
- Current lineage IDs are synthetic; original provenance is unknown.
- Do not equalize counts by blindly duplicating documents: that would change the redundancy manipulation.
- A defensible balanced factorial needs source-grounded distinct passages for low redundancy and repeated/overlapping evidence for high redundancy, with equal document counts and equivalent claim coverage in all four cells.
- Describe provenance dependence as a synthetic intervention unless real source lineage is verified.
- Gold-answer concerns for mh_0174, mh_0533, and mh_0639 remain unresolved pending full-source review.

## Gate decision

**NOT CLEARED.** This is intentional: arbitrary duplicate padding would invalidate the low-redundancy condition, and fabricated paraphrases would not be source-grounded.
