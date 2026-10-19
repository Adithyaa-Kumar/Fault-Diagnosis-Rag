"""Phase 0 environment and dataset manifests (stdlib only; no torch import).

  python -m src.evaluation.phase0_manifests env      -> results/phase0/env_manifest.json
  python -m src.evaluation.phase0_manifests dataset  -> results/phase0/dataset_manifest.json
  python -m src.evaluation.phase0_manifests verify   -> exit 1 if any hashed input changed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

OUT_DIR = Path("results/phase0")
RAW_INPUTS = [
    "data/raw/multihoprag/MultiHopRAG.json",
    "data/raw/multihoprag/corpus.json",
]
PROCESSED_INPUTS = [
    "data/processed/multihoprag/pilot_30.json",
    "data/processed/multihoprag/claim_annotations_30.json",
    "data/processed/multihoprag/controlled_bundles_v2.json",
]
CONFIG_INPUTS = [
    "requirements.txt", "requirements-llm.txt", "docs/PHASE0_DESIGN_FREEZE.md",
    "src/evaluation/run_multihop_v2.py", "src/evaluation/multihop_scoring.py",
    "src/pipeline/multihop_faults.py",
]
KEY_PACKAGES = ["torch", "transformers", "bitsandbytes", "accelerate", "pydantic", "pytest", "numpy"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_paths(root: Path, paths: list[str]) -> dict[str, Optional[dict[str, Any]]]:
    out: dict[str, Optional[dict[str, Any]]] = {}
    for rel in paths:
        p = Path(root) / rel
        out[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size} if p.is_file() else None
    return out


def _git(root: Path, *args: str) -> Optional[str]:
    try:
        res = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def capture_env(root: Path = Path(".")) -> dict[str, Any]:
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    packages = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            packages[name.lower()] = dist.version
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(Path(root).resolve()),
        "git": {
            "is_repository": commit is not None, "commit": commit,
            "working_tree_clean": (status == "") if status is not None else None,
            "dirty_files": status.splitlines() if status else [],
            "note": None if commit else "Not a git repository (or git unavailable); commit unknown.",
        },
        "python": {"version": sys.version, "executable": sys.executable},
        "os": {"platform": platform.platform(), "machine": platform.machine()},
        "key_packages": {k: packages.get(k) for k in KEY_PACKAGES},
        "all_packages": dict(sorted(packages.items())),
        "config_hashes": hash_paths(Path(root), CONFIG_INPUTS),
    }


def validate_ids(bundle: dict[str, Any], annotations: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Unique question, document and claim IDs; bundle/annotation agreement."""
    errors: list[str] = []
    questions = bundle.get("questions", [])
    sample_ids = [q["sample_id"] for q in questions]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("duplicate sample_id in bundle")
    doc_ids: list[str] = []
    for q in questions:
        for cond, c in q["conditions"].items():
            ids = [d["doc_id"] for d in c["documents"]]
            doc_ids += ids
            if len(ids) != len(set(ids)):
                errors.append(f"{q['sample_id']}/{cond}: duplicate doc_id in cell")
    if len(doc_ids) != len(set(doc_ids)):
        errors.append("doc_id not globally unique")
    claim_ids = 0
    if annotations is not None:
        ann = annotations.get("annotations", [])
        ann_ids = [a["sample_id"] for a in ann]
        if len(ann_ids) != len(set(ann_ids)):
            errors.append("duplicate sample_id in annotations")
        for a in ann:
            cids = [c["claim_id"] for c in a.get("required_claims", [])]
            claim_ids += len(cids)
            if len(cids) != len(set(cids)):
                errors.append(f"{a['sample_id']}: duplicate claim_id")
        missing = sorted(set(sample_ids) - set(ann_ids))
        if missing:
            errors.append(f"bundle samples without annotations: {missing[:5]}")
    return {"questions": len(sample_ids), "documents": len(doc_ids), "claims": claim_ids,
            "errors": errors, "valid": not errors}


def dataset_manifest(root: Path = Path(".")) -> dict[str, Any]:
    root = Path(root)

    def load(rel):
        p = root / rel
        return json.loads(p.read_text(encoding="utf-8-sig")) if p.is_file() else None

    bundle, ann = load(PROCESSED_INPUTS[2]), load(PROCESSED_INPUTS[1])
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MultiHop-RAG",
        "source_hint": "https://github.com/yixuantt/MultiHop-RAG",
        "license": None, "version": None,
        "license_version_note": "Not asserted here. Copy license and version from the dataset card "
                                "you downloaded from, then re-run; null means unrecorded.",
        "raw_files": hash_paths(root, RAW_INPUTS),
        "processed_files": hash_paths(root, PROCESSED_INPUTS),
        "raw_files_immutable_policy": "Raw files are never written by project code; "
                                      "'verify' fails if a hash changes.",
        "id_validation": validate_ids(bundle, ann) if bundle else {"valid": False, "errors": ["bundle missing"]},
    }


def verify(root: Path = Path("."), manifest_path: Path = OUT_DIR / "dataset_manifest.json") -> list[str]:
    path = Path(root) / manifest_path
    if not path.is_file():
        return [f"{manifest_path} not found; run the 'dataset' command first"]
    saved = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    for group in ("raw_files", "processed_files"):
        now = hash_paths(Path(root), list(saved[group]))
        for rel, old in saved[group].items():
            cur = now[rel]
            if (old or {}).get("sha256") != (cur or {}).get("sha256"):
                problems.append(f"{group}: {rel} changed or missing")
    return problems


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["env", "dataset", "verify"])
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root)
    if args.command == "env":
        _write(root / OUT_DIR / "env_manifest.json", capture_env(root))
        print("Wrote", OUT_DIR / "env_manifest.json")
    elif args.command == "dataset":
        manifest = dataset_manifest(root)
        _write(root / OUT_DIR / "dataset_manifest.json", manifest)
        print("Wrote", OUT_DIR / "dataset_manifest.json", "| ids valid:", manifest["id_validation"]["valid"])
    else:
        problems = verify(root)
        print("\n".join(problems) if problems else "All hashed inputs match.")
        raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
