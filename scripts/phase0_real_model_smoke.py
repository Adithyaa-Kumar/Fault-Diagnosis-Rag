
import json
from pathlib import Path

import torch

from src.models.local_qwen import LocalQwen


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# This is an engineering smoke test, not a research result.
# Replace this probe with verified benchmark cases in the
# subsequent Phase 0 experiment runner.
CONTEXT = (
    "The fictional Arbordale Research Board issued permit AR-17 "
    "on 12 March 2025."
)
QUESTION = "Which permit did the board issue?"
GOLD_ANSWER = "The board issued permit AR-17."


def main() -> None:
    model = LocalQwen(max_new_tokens=64)

    print("\nRunning actual-model generation...")
    generation = model.generate_answer(
        context=CONTEXT,
        question=QUESTION,
    )

    print("\nRunning gold-answer likelihood scoring...")
    likelihood = model.score_gold_answer(
        context=CONTEXT,
        question=QUESTION,
        gold_answer=GOLD_ANSWER,
    )

    result = {
        "phase": "phase_0_real_model_smoke",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "device": torch.cuda.get_device_name(0),
        "gpu_total_memory_gib": round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            3,
        ),
        "gpu_peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(0) / 1024**3,
            3,
        ),
        "generation": generation,
        "gold_answer_likelihood": likelihood,
        "status": "passed",
        "interpretation": (
            "Engineering smoke test only; not a benchmark "
            "or scientific result."
        ),
    }

    output_path = LOG_DIR / "phase0_real_model_smoke.json"
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\n--- REAL MODEL SMOKE TEST ---")
    print(json.dumps(result, indent=2))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
