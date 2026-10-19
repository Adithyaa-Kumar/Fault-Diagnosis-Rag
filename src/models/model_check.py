"""Phase 0 model gates -> results/phase0/model_manifest.json.

  python -m src.models.model_check

Gates: generate, chat template, determinism, realistic-size prompt with peak VRAM / OOM,
no truncation, answer extraction. Everything recorded is measured; nothing is estimated.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from src.evaluation.multihop_scoring import score_multihop_answer
from src.evaluation.run_multihop_v2 import (
    ARBITER_PROMPT, MAX_NEW_TOKENS, SPECIALIST_PROMPT, prompt_manifest, render_doc,
)

OUT = Path("results/phase0/model_manifest.json")
BUNDLE = Path("data/processed/multihoprag/controlled_bundles_v2.json")
# Unambiguous toys: every report names the same subject, so a correct arbiter can commit.
TOYS = (
    ("Specialist_1 report:\nThe city council approved the 2024 transit budget.\n\n"
     "Specialist_2 report:\nThe 2024 transit budget includes three new bus routes.",
     "Did the city council approve a 2024 transit budget that includes new bus routes?", "Yes"),
    ("Specialist_1 report:\nThe city council rejected the 2024 transit budget.\n\n"
     "Specialist_2 report:\nThe 2024 transit budget was sent back to committee.",
     "Did the city council approve the 2024 transit budget?", "No"),
)
TOY_CONTEXT, TOY_QUESTION = TOYS[0][0], TOYS[0][1]
LABEL_LINE = re.compile(r"(?im)^\s*[*`]*answer[*`]*\s*:")


def check_chat_template(tokenizer: Any, system: str, user: str) -> dict[str, Any]:
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    ok = ("<|im_start|>system" in text and "<|im_start|>user" in text
          and text.rstrip().endswith("<|im_start|>assistant"))
    return {"pass": bool(ok), "rendered_tail": text[-80:]}


def _is_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


def timed_generate(model: Any, probe: Any, **kwargs) -> dict[str, Any]:
    probe.reset()
    start = time.perf_counter()
    try:
        result = model.generate_answer(**kwargs)
    except Exception as exc:
        return {"ok": False, "oom": _is_oom(exc), "error": f"{type(exc).__name__}: {exc}",
                "peak_vram_bytes": probe.peak()}
    return {"ok": True, "oom": False, "answer": result.get("answer", ""),
            "prompt_tokens": result.get("prompt_tokens"),
            "generated_tokens": result.get("generated_tokens"),
            "latency_seconds": time.perf_counter() - start, "peak_vram_bytes": probe.peak()}


def largest_specialist_context(bundle: dict[str, Any]) -> tuple[str, str]:
    """Longest rendered specialist context over the bundle (all cells, round-robin split)."""
    best = ("", "")
    for q in bundle["questions"]:
        for c in q["conditions"].values():
            docs = c["documents"]
            for i in range(3):
                ctx = "\n\n".join(render_doc(d) for d in docs[i::3])
                if len(ctx) > len(best[0]):
                    best = (ctx, q["query"])
    return best


def run_checks(model: Any, bundle: Optional[dict[str, Any]], probe: Any,
               max_positions: Optional[int]) -> dict[str, Any]:
    gates: dict[str, Any] = {}

    gates["chat_template"] = check_chat_template(model.tokenizer, SPECIALIST_PROMPT, "Context:\nx\n\nQuestion:\ny")

    g = timed_generate(model, probe, context="The sky is blue.", question="What color is the sky? One word.",
                       system_prompt="Answer briefly.", max_new_tokens=16)
    gates["generate"] = {**g, "pass": bool(g["ok"] and g["answer"].strip())}

    runs = [timed_generate(model, probe, context=TOY_CONTEXT, question=TOY_QUESTION,
                           system_prompt=SPECIALIST_PROMPT, max_new_tokens=MAX_NEW_TOKENS["specialist"])
            for _ in range(3)]
    same = all(r["ok"] for r in runs) and len({r["answer"] for r in runs}) == 1
    gates["determinism"] = {
        "pass": bool(same), "runs": 3, "answers": [r.get("answer") for r in runs],
        "settings": {"do_sample": False, "temperature": None, "top_p": None, "seed_required": False},
        "note": "Greedy decoding; GPU kernels can still differ across hardware or library versions."}

    probes = []
    for context, question, gold in TOYS:
        run = timed_generate(model, probe, context=context, question=question,
                             system_prompt=ARBITER_PROMPT, max_new_tokens=MAX_NEW_TOKENS["arbiter"])
        scored = score_multihop_answer(run.get("answer", ""), gold) if run["ok"] else None
        probes.append({
            "gold": gold, "raw_output": run.get("answer"),
            "has_answer_line": bool(run["ok"] and LABEL_LINE.search(run["answer"])),
            "extracted": scored and scored["extracted_answer"],
            "parsed_yes_no": bool(scored and scored["extracted_answer"] in ("Yes", "No")),
            "correct_non_gating": scored and scored["correct"]})
    gates["answer_extraction"] = {
        "pass": all(p["has_answer_line"] and p["parsed_yes_no"] for p in probes),
        "probes": probes,
        "note": "Gates format compliance and parsing only (an 'Answer:' line that parses to Yes/No on "
                "unambiguous toys). Correctness is reported, not gated; abstaining on a toy fails the gate."}

    if bundle:
        ctx, question = largest_specialist_context(bundle)
        big = timed_generate(model, probe, context=ctx, question=question,
                             system_prompt=SPECIALIST_PROMPT, max_new_tokens=MAX_NEW_TOKENS["specialist"])
        pt = big.get("prompt_tokens")
        within = bool(pt is not None and (max_positions is None or pt + MAX_NEW_TOKENS["specialist"] < max_positions))
        gates["realistic_prompt"] = {
            **{k: v for k, v in big.items() if k != "answer"}, "context_chars": len(ctx),
            "max_position_embeddings": max_positions,
            "pass": bool(big["ok"] and not big["oom"] and within)}
        gates["no_truncation"] = {
            "pass": within, "prompt_tokens": pt, "max_position_embeddings": max_positions,
            "note": "The wrapper applies no truncation; this gate checks the longest prompt fits the window."}
    else:
        gates["realistic_prompt"] = {"pass": False, "error": "bundle not found"}
        gates["no_truncation"] = {"pass": False, "error": "bundle not found"}

    return {"gates": gates, "all_gates_pass": all(g["pass"] for g in gates.values())}


def fetch_license(model_id: str) -> dict[str, Any]:
    try:
        from huggingface_hub import hf_hub_download
        text = Path(hf_hub_download(model_id, "README.md")).read_text(encoding="utf-8")
        m = re.search(r"(?m)^license:\s*(.+)$", text)
        return {"license": m.group(1).strip() if m else None, "source": "model card README frontmatter"}
    except Exception as exc:  # offline, gated, etc.
        return {"license": None, "error": f"{type(exc).__name__}: {exc}",
                "note": "Not verified; check the model card manually."}


class TorchProbe:
    def __init__(self, torch_module: Any):
        self.torch = torch_module

    def reset(self) -> None:
        self.torch.cuda.reset_peak_memory_stats()

    def peak(self) -> int:
        return int(self.torch.cuda.max_memory_allocated())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(BUNDLE))
    parser.add_argument("--output", default=str(OUT))
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    import torch
    import transformers
    from src.models.local_qwen import LocalQwen

    props = torch.cuda.get_device_properties(0)
    start = time.perf_counter()
    model = LocalQwen(model_id=args.model_id)
    load_seconds = time.perf_counter() - start
    after_load = int(torch.cuda.memory_allocated())
    cfg = model.model.config
    bundle_path = Path(args.bundle)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8")) if bundle_path.is_file() else None

    try:
        import bitsandbytes
        bnb_version = bitsandbytes.__version__
    except Exception:
        bnb_version = None

    result = run_checks(model, bundle, TorchProbe(torch), getattr(cfg, "max_position_embeddings", None))
    manifest = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "revision_commit_hash": getattr(cfg, "_commit_hash", None),
        "revision_note": "Resolved snapshot hash of what was actually loaded; null means unavailable.",
        "quantization": {"load_in_4bit": True, "quant_type": "nf4", "compute_dtype": "float16",
                         "double_quant": True},
        "dtype": "float16",
        "inference": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "chat_template": "tokenizer default"},
        "generation_config_loaded": model.model.generation_config.to_dict(),
        "generation_config_note": "Applied by generate() even with do_sample=False "
                                  "(e.g. repetition_penalty); sampling fields are ignored when do_sample=False.",
        "prompts": prompt_manifest(),
        "license": fetch_license(args.model_id),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "transformers": transformers.__version__, "bitsandbytes": bnb_version},
        "gpu": {"name": props.name, "total_memory_bytes": int(props.total_memory)},
        "load_seconds_measured": load_seconds, "vram_after_load_bytes": after_load,
        **result,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    for name, gate in manifest["gates"].items():
        print(f"  {'PASS' if gate['pass'] else 'FAIL'}  {name}")
    if not manifest["all_gates_pass"]:
        raise SystemExit("One or more model gates failed; see manifest")


if __name__ == "__main__":
    main()
