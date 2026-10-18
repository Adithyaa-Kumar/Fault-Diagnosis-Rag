import json
import random
from pathlib import Path
from collections import Counter

SEED = 20261018
SAMPLE_PER_TYPE = 10
TYPES = ["inference_query", "comparison_query", "temporal_query"]

source_path = Path("data/raw/multihoprag/MultiHopRAG.json")
out_dir = Path("data/processed/multihoprag")
out_dir.mkdir(parents=True, exist_ok=True)

with source_path.open("r", encoding="utf-8") as f:
    questions = json.load(f)

eligible = {
    kind: [
        (index, question)
        for index, question in enumerate(questions)
        if question.get("question_type") == kind
        and isinstance(question.get("evidence_list"), list)
        and len(question["evidence_list"]) > 0
        and str(question.get("query", "")).strip()
        and str(question.get("answer", "")).strip()
    ]
    for kind in TYPES
}

rng = random.Random(SEED)
selected = []

for kind in TYPES:
    pool = eligible[kind]
    if len(pool) < SAMPLE_PER_TYPE:
        raise SystemExit(
            f"Not enough eligible questions for {kind}: {len(pool)}"
        )

    chosen = rng.sample(pool, SAMPLE_PER_TYPE)

    for original_index, question in chosen:
        selected.append({
            "sample_id": f"mh_{original_index:04d}",
            "original_index": original_index,
            "question_type": kind,
            "query": question["query"],
            "answer": question["answer"],
            "evidence_list": question["evidence_list"],
            "selection_seed": SEED,
        })

# Stable order in output, independent of selection draw order.
selected.sort(key=lambda item: (item["question_type"], item["original_index"]))

output = {
    "dataset": "MultiHop-RAG",
    "sample_type": "controlled_evaluation_pipeline_pilot",
    "sample_size": len(selected),
    "selection_seed": SEED,
    "selection_policy": {
        "types": TYPES,
        "questions_per_type": SAMPLE_PER_TYPE,
        "requires_nonempty_query_answer_and_evidence": True,
        "null_query_included": False,
        "selection": "uniform random sample without replacement within each type",
    },
    "important_limitations": [
        "This pilot sample does not establish statistical power.",
        "Original provenance independence is unknown unless separately verified.",
        "The four experimental conditions must be constructed and validated separately.",
    ],
    "questions": selected,
}

output_path = out_dir / "pilot_30.json"
output_path.write_text(
    json.dumps(output, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("PILOT SAMPLE CREATED")
print("Output:", output_path)
print("Total selected:", len(selected))
print("Counts by type:", dict(Counter(q["question_type"] for q in selected)))
print("Seed:", SEED)
print("Raw dataset modified: False")
