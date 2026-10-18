
"""Real-data parallel-agent pilot using HotpotQA and local Qwen.

This is a baseline experiment, not yet a provenance-dependence experiment.
Each specialist sees a separate subset of context paragraphs.
"""
import json
import random
import time
from pathlib import Path

from src.evaluation.scoring import AnswerScorer
from src.models.local_qwen import LocalQwen


DATASET = Path("data/raw/hotpot_dev_distractor_v1.json")
OUTPUT = Path("results/hotpotqa_parallel_pilot.json")
SEED = 2026
LIMIT = 10


def get_documents(example):
    """Preserve paragraph titles and sentence text from the source dataset."""
    docs = []
    for idx, row in enumerate(example.get("context", [])):
        if len(row) < 2:
            continue
        title, sentences = row[0], row[1]
        if not isinstance(sentences, list):
            continue
        content = " ".join(str(s) for s in sentences).strip()
        if content:
            docs.append({
                "doc_id": f"doc_{idx}",
                "title": str(title),
                "content": content,
            })
    return docs


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    examples = json.loads(DATASET.read_text(encoding="utf-8"))
    valid = [
        ex for ex in examples
        if ex.get("_id") and ex.get("question") and ex.get("answer")
        and len(get_documents(ex)) >= 3
        and ex.get("supporting_facts")
    ]

    # Fixed, reproducible subset; does not alter the raw dataset.
    rng = random.Random(SEED)
    diagnostic_ids = {
        "5a7a33205542996a35c1712f",  # Province of New York
        "5abad3445542996cc5e49ea0",  # Portland Trail Blazers
        "5add3e2b5542997545bbbcf5",  # Kathryn Bigelow
    }

    selected = [ex for ex in valid if ex["_id"] in diagnostic_ids]

    model = LocalQwen()
    rows = []
    started = time.perf_counter()

    for i, ex in enumerate(selected, start=1):
        docs = get_documents(ex)

        # Keep supporting paragraphs ahead of distractors, without inventing
        # source provenance. Supporting facts identify title/sentence pairs.
        support_titles = {
            str(item[0]) for item in ex.get("supporting_facts", [])
            if isinstance(item, (list, tuple)) and len(item) >= 2
        }
        ordered_docs = (
            [d for d in docs if d["title"] in support_titles]
            + [d for d in docs if d["title"] not in support_titles]
        )

        # Control: answer once with the complete context.
        full_context = "\n\n".join(
            f"Title: {d['title']}\nEvidence: {d['content']}"
            for d in ordered_docs
        )

        full_result = model.generate_answer(
            context=full_context,
            question=ex["question"],
            system_prompt=(
                "Answer the question using only the supplied evidence. "
                "Combine facts across paragraphs when necessary. "
                "Resolve references using the context. "
                "Return the shortest answer that fully answers the question."
            ),
            max_new_tokens=64,
        )
        full_metrics = AnswerScorer.score_metrics(
            full_result["answer"], ex["answer"]
        )


        # Three isolated specialist contexts, assigned round-robin.
        chunks = [ordered_docs[j::3] for j in range(3)]
        reports = []
        measured_prompt_tokens = 0
        measured_generated_tokens = 0

        for specialist_idx, chunk in enumerate(chunks, start=1):
            context = "\n\n".join(
                f"Title: {d['title']}\nEvidence: {d['content']}"
                for d in chunk
            ) or "No evidence provided."

            result = model.generate_answer(
                context=context,
                question=ex["question"],
                system_prompt=(
                    "You are a specialist in a multi-agent evidence system. "
                    "Use only your assigned context. Report relevant facts "
                    "with their titles. If your context is insufficient, "
                    "say what is missing. Do not guess."
                ),
                max_new_tokens=96,
            )
            reports.append(
                f"Specialist {specialist_idx} report:\n{result['answer']}"
            )
            measured_prompt_tokens += result["prompt_tokens"]
            measured_generated_tokens += result["generated_tokens"]

        arbiter_result = model.generate_answer(
            context="\n\n".join(reports),
            question=ex["question"],
            system_prompt=(
                "You are the arbiter of a multi-agent evidence system. "
                "Answer using only the specialist reports. If they do not "
                "support an answer, state that clearly. Do not guess. "
                "Return a concise answer."
            ),
            max_new_tokens=96,
        )

        prediction = arbiter_result["answer"]
        metrics = AnswerScorer.score_metrics(prediction, ex["answer"])

        measured_prompt_tokens += arbiter_result["prompt_tokens"]
        measured_generated_tokens += arbiter_result["generated_tokens"]

        rows.append({
            "query_id": ex["_id"],
            "question": ex["question"],
            "gold_answer": ex["answer"],
            "prediction": prediction,
            "exact_match": metrics["exact_match"],
            "token_f1": metrics["token_f1"],
            "scorer_correct_at_f1_0_5": metrics["correct"],
            "context_paragraphs": len(docs),
            "specialist_count": len(chunks),
            "prompt_tokens_measured": measured_prompt_tokens,
            "generated_tokens_measured": measured_generated_tokens,
            "model_id": arbiter_result["model_id"],
            "supporting_titles": sorted(support_titles),
            "specialist_reports": reports,
            "provenance_status": "unverified",
            "full_context_prediction": full_result["answer"],
            "full_context_exact_match": full_metrics["exact_match"],
            "full_context_token_f1": full_metrics["token_f1"],
            "full_context_prompt_tokens_measured": full_result["prompt_tokens"],
            "full_context_generated_tokens_measured": full_result["generated_tokens"],
        })

        print(
            f"[{i}/{len(selected)}] "
            f"F1={metrics['token_f1']:.3f} "
            f"EM={metrics['exact_match']} "
            f"ID={ex['_id']}"
        )

    elapsed = time.perf_counter() - started
    report = {
        "experiment": "hotpotqa_real_data_parallel_baseline",
        "dataset": str(DATASET),
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "random_seed": SEED,
        "requested_examples": LIMIT,
        "evaluated_examples": len(rows),
        "elapsed_seconds_measured": round(elapsed, 3),
        "exact_match_rate": (
            sum(r["exact_match"] for r in rows) / len(rows) if rows else None
        ),
        "mean_token_f1": (
            sum(r["token_f1"] for r in rows) / len(rows) if rows else None
        ),
        "scorer_accuracy_at_f1_0_5": (
            sum(r["scorer_correct_at_f1_0_5"] for r in rows) / len(rows)
            if rows else None
        ),
        "total_prompt_tokens_measured": sum(
            r["prompt_tokens_measured"] for r in rows
        ),
        "total_generated_tokens_measured": sum(
            r["generated_tokens_measured"] for r in rows
        ),
        "limitations": [
            "Small pilot; results do not establish statistical significance.",
            "HotpotQA supporting facts are not proof of independent provenance.",
            "Specialists run sequentially on one shared local model due to hardware limits.",
            "This baseline has no injected faults or counterfactual interventions.",
            "The answer scorer is an automatic metric and may misjudge valid answers.",
        ],
        "runs": rows,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== PILOT SUMMARY ===")
    print(json.dumps(
        {k: v for k, v in report.items() if k != "runs"},
        indent=2,
    ))
    print(f"Detailed results: {OUTPUT}")


if __name__ == "__main__":
    main()
