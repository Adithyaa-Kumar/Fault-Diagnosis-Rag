"""Small, reproducible real-dataset smoke test using HotpotQA and local Qwen."""
import json
import time
from pathlib import Path
from typing import Any

from src.evaluation.scoring import AnswerScorer
from src.models.local_qwen import LocalQwen


def _context(example: dict[str, Any]) -> str:
    paragraphs = example.get("context", [])
    return "\n\n".join(f"Title: {row[0]}\n" + " ".join(row[1]) for row in paragraphs)


def run_hotpotqa_smoke(dataset_path: str = "data/raw/hotpot_dev_distractor_v1.json",
                       limit: int = 3, output_path: str = "results/hotpotqa_qwen_smoke.json") -> dict:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"HotpotQA file not found: {path}. Put hotpot_dev_distractor_v1.json in data/raw/. "
            "The dataset is not included in the code archive."
        )
    examples = json.loads(path.read_text(encoding="utf-8"))
    valid = [x for x in examples if x.get("question") and x.get("answer") and x.get("context")]
    model = LocalQwen()
    rows = []
    started = time.perf_counter()
    for ex in valid[:limit]:
        prompt_context = _context(ex)
        result = model.generate_answer(
            context=prompt_context,
            question=ex["question"],
            system_prompt=("Answer the question using only the supplied Wikipedia context. "
                           "Give the shortest answer that fully answers the question. "
                           "If the evidence is insufficient, say so."),
            max_new_tokens=64,
        )
        metrics = AnswerScorer.score_metrics(result["answer"], ex["answer"])
        rows.append({
            "query_id": ex.get("_id"), "question": ex["question"], "gold_answer": ex["answer"],
            "prediction": result["answer"], "exact_match": metrics["exact_match"],
            "token_f1": metrics["token_f1"], "scorer_correct_at_f1_0_5": metrics["correct"],
            "prompt_tokens_measured": result["prompt_tokens"],
            "generated_tokens_measured": result["generated_tokens"],
            "model_id": result["model_id"],
        })
    elapsed = time.perf_counter() - started
    report = {
        "experiment": "hotpotqa_local_qwen_smoke",
        "dataset_file": str(path), "dataset_total_examples": len(examples),
        "valid_examples": len(valid), "evaluated_examples": len(rows),
        "elapsed_seconds_measured": round(elapsed, 3),
        "exact_match_rate": sum(r["exact_match"] for r in rows) / len(rows) if rows else None,
        "mean_token_f1": sum(r["token_f1"] for r in rows) / len(rows) if rows else None,
        "rows": rows,
        "limitations": ["Smoke test only; tiny non-random sample, not a research result.",
                        "The full context is supplied; this does not test retrieval quality.",
                        "Token counts are reported only from the model wrapper's tokenizer measurements."],
    }
    out = Path(output_path); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    print(f"Detailed results: {out}")
    return report


if __name__ == "__main__":
    run_hotpotqa_smoke()
