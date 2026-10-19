import json
import shutil
from pathlib import Path

import pytest

from src.evaluation import phase0_manifests as pm
from src.models import model_check as mc


# ------------------------------------------------------------ manifests
def test_hash_paths_and_verify_detects_change(tmp_path):
    raw = tmp_path / "data/raw/multihoprag"
    raw.mkdir(parents=True)
    (raw / "MultiHopRAG.json").write_text("[1]", encoding="utf-8")
    (raw / "corpus.json").write_text("[2]", encoding="utf-8")
    proc = tmp_path / "data/processed/multihoprag"
    proc.mkdir(parents=True)
    for name in ("pilot_30.json", "claim_annotations_30.json", "controlled_bundles_v2.json"):
        (proc / name).write_text(json.dumps({"questions": [], "annotations": []}), encoding="utf-8")
    manifest = pm.dataset_manifest(tmp_path)
    assert all(v and len(v["sha256"]) == 64 for v in manifest["raw_files"].values())
    out = tmp_path / pm.OUT_DIR
    out.mkdir(parents=True)
    (out / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert pm.verify(tmp_path) == []
    (raw / "corpus.json").write_text("[3]", encoding="utf-8")
    assert any("corpus.json" in p for p in pm.verify(tmp_path))


def test_verify_without_manifest_reports_problem(tmp_path):
    assert pm.verify(tmp_path)


def test_validate_ids_flags_duplicates_and_missing_annotations():
    doc = lambda i: {"doc_id": i}
    bundle = {"questions": [
        {"sample_id": "a", "conditions": {"A": {"documents": [doc("x"), doc("x")]}}},
        {"sample_id": "a", "conditions": {"A": {"documents": [doc("y")]}}},
        {"sample_id": "b", "conditions": {"A": {"documents": [doc("z")]}}}]}
    ann = {"annotations": [{"sample_id": "a", "required_claims": [{"claim_id": "C1"}, {"claim_id": "C1"}]}]}
    r = pm.validate_ids(bundle, ann)
    assert not r["valid"]
    text = " ".join(r["errors"])
    for needle in ("duplicate sample_id", "duplicate doc_id", "duplicate claim_id", "without annotations"):
        assert needle in text


def test_capture_env_structure_and_non_git_note(tmp_path):
    env = pm.capture_env(tmp_path)
    assert env["git"]["is_repository"] is False and env["git"]["note"]
    assert "pytest" in env["key_packages"] and env["python"]["version"]


REAL = Path("data/processed/multihoprag/controlled_bundles_v2.json")


@pytest.mark.skipif(not (REAL.exists() and Path("data/processed/multihoprag/claim_annotations_30.json").exists()),
                    reason="real data not present")
def test_real_ids_valid():
    manifest = pm.dataset_manifest(Path("."))
    assert manifest["id_validation"]["valid"], manifest["id_validation"]["errors"]
    assert manifest["id_validation"]["questions"] == 30


# ---------------------------------------------------------- model check
class FakeProbe:
    def reset(self): pass
    def peak(self): return 123


class FakeTokenizer:
    def __init__(self, good=True):
        self.good = good

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if not self.good:
            return "plain text"
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"


class FakeModel:
    def __init__(self, answers=None, good_template=True, fail=None, tokens=50):
        self.tokenizer = FakeTokenizer(good_template)
        self.answers, self.fail, self.tokens, self.n = answers, fail, tokens, 0

    def generate_answer(self, context, question, system_prompt, max_new_tokens=None):
        self.n += 1
        if self.fail:
            raise RuntimeError(self.fail)
        if "arbiter" in system_prompt:
            text = "Both facts are supported.\nAnswer: Yes"
        else:
            text = self.answers[self.n % len(self.answers)] if self.answers else "Facts: ok"
        return {"answer": text, "prompt_tokens": self.tokens, "generated_tokens": 5, "model_id": "fake"}


def bundle():
    d = {"doc_id": "x", "content": "Some evidence text.", "source_title": "T", "source_publisher": "P",
         "published_at": "2023"}
    return {"questions": [{"query": "Q?", "conditions": {"A": {"documents": [d, d, d, d]}}}]}


def test_all_gates_pass_with_well_behaved_model():
    r = mc.run_checks(FakeModel(), bundle(), FakeProbe(), 32768)
    assert r["all_gates_pass"], r["gates"]
    assert set(r["gates"]) == {"chat_template", "generate", "determinism", "answer_extraction",
                               "realistic_prompt", "no_truncation"}
    assert r["gates"]["realistic_prompt"]["peak_vram_bytes"] == 123


def test_nondeterminism_bad_template_and_oom_are_detected():
    r = mc.run_checks(FakeModel(answers=["a", "b", "c"]), bundle(), FakeProbe(), 32768)
    assert not r["gates"]["determinism"]["pass"] and not r["all_gates_pass"]
    assert not mc.run_checks(FakeModel(good_template=False), bundle(), FakeProbe(), 32768)["gates"]["chat_template"]["pass"]
    oom = mc.run_checks(FakeModel(fail="CUDA out of memory"), bundle(), FakeProbe(), 32768)
    assert oom["gates"]["realistic_prompt"]["oom"] is True and not oom["gates"]["realistic_prompt"]["pass"]


def test_prompt_exceeding_window_fails_truncation_gate():
    r = mc.run_checks(FakeModel(tokens=40000), bundle(), FakeProbe(), 32768)
    assert not r["gates"]["no_truncation"]["pass"]


def test_missing_bundle_fails_those_gates_not_crash():
    r = mc.run_checks(FakeModel(), None, FakeProbe(), 32768)
    assert not r["gates"]["realistic_prompt"]["pass"] and not r["all_gates_pass"]


def test_extraction_gate_fails_on_abstention_and_on_missing_answer_line():
    class Abstainer(FakeModel):
        def generate_answer(self, context, question, system_prompt, max_new_tokens=None):
            r = super().generate_answer(context, question, system_prompt, max_new_tokens)
            if "arbiter" in system_prompt:
                r["answer"] = "Answer: Insufficient evidence. No link between the documents."
            return r

    class NoLabel(FakeModel):
        def generate_answer(self, context, question, system_prompt, max_new_tokens=None):
            r = super().generate_answer(context, question, system_prompt, max_new_tokens)
            if "arbiter" in system_prompt:
                r["answer"] = "Yes, the council approved it."
            return r

    for model in (Abstainer(), NoLabel()):
        gate = mc.run_checks(model, bundle(), FakeProbe(), 32768)["gates"]["answer_extraction"]
        assert gate["pass"] is False
    ok = mc.run_checks(FakeModel(), bundle(), FakeProbe(), 32768)["gates"]["answer_extraction"]
    assert ok["pass"] and len(ok["probes"]) == 2


def test_markdown_bold_answer_line_counts_as_label():
    assert mc.LABEL_LINE.search("Reasoning.\n**Answer:** Yes")
