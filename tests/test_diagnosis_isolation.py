import ast
import inspect
import json
import re
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.evaluation.build_bundles_v2 import CONDITIONS, build_v2
from src.pipeline import multihop_faults as mf
from src.schema import FaultType

FAULTS = [FaultType.NONE, FaultType.RETRIEVAL, FaultType.SPECIALIST, FaultType.ARBITER]

ART_A = [
    "Alpha opening sentence introduces the regional transit authority and its mandate.",
    "The council approved the new transit budget on Monday after a long public debate.",
    "Officials said the budget includes funding for three additional suburban bus routes.",
    "Weather forecasters expect heavy rain across the northern valley during the weekend.",
    "Local farmers reported that the late harvest was delayed by persistent cold nights.",
    "A museum announced a new exhibit of regional pottery opening in the spring season.",
    "Closing remarks from the mayor thanked residents for attending the public meeting.",
]
ART_B = [
    "Beta opening sentence describes the harbor commission and its annual shipping report.",
    "Harbor regulators confirmed that cargo volumes rose sharply during the third quarter.",
    "The commission credited new crane equipment for faster container unloading times.",
    "Fishermen complained about restricted access to the eastern pier during construction.",
    "A tourism board survey found that cruise visitors prefer shorter guided walking tours.",
    "Engineers inspected the lighthouse foundation and recommended minor stone repairs soon.",
    "The harbor festival will feature music, food stalls, and a lantern parade on Saturday.",
]


def mini_question(tmp_path):
    arts = []
    for n, sents in enumerate((ART_A, ART_B)):
        arts.append({"title": f"T{n}", "author": "A", "source": "Pub", "published_at": "2023-01-01",
                     "category": "x", "url": f"https://example.test/{n}", "body": " ".join(sents)})
    ev = [{"title": a["title"], "author": "A", "url": a["url"], "source": "Pub", "category": "x",
           "published_at": "2023-01-01", "fact": s[2 if i == 0 else 2]}
          for i, (a, s) in enumerate(zip(arts, (ART_A, ART_B)))]
    pilot = {"selection_seed": 3, "questions": [{
        "sample_id": "q1", "original_index": 0, "question_type": "comparison_query",
        "query": "Did both bodies report budget funding and equipment improvements?",
        "answer": "Yes", "evidence_list": ev, "selection_seed": 3}]}
    pp, cp = tmp_path / "p.json", tmp_path / "c.json"
    pp.write_text(json.dumps(pilot), encoding="utf-8")
    cp.write_text(json.dumps(arts), encoding="utf-8")
    payload = build_v2(pp, cp)
    assert payload["excluded_questions"] == [], payload["excluded_questions"]
    return payload["questions"][0]


def fake_generate(role, context, question):
    """Deterministic stand-in for the model: grounded extraction, no label access."""
    if "No evidence provided" in context:
        return "The provided context does not contain relevant evidence."
    lines = [ln for ln in context.splitlines() if ln.strip()]
    if role == "specialist":
        body = re.sub(r"^\[[^\]]+\]\s*", "", lines[0])
        return "Facts: " + body.split(". ")[0]
    reports = [ln for ln in lines if not ln.endswith("report:")]
    return "Answer: " + " ".join(reports)


@pytest.fixture()
def question(tmp_path):
    return mini_question(tmp_path)


def test_observation_schema_cannot_carry_labels_or_ground_truth():
    names = set(mf.MultiHopObservation.model_fields) | set(mf.SpecialistObservation.model_fields)
    assert not any(re.search(r"inject|gold|required|label|fault|claim", n) for n in names)
    with pytest.raises(ValidationError):
        mf.MultiHopObservation(query="q", specialists=[], arbiter_inputs=[], final_verdict="v",
                               injected_fault_layer="retrieval")


def test_diagnoser_signature_and_source_do_not_reference_labels():
    assert list(inspect.signature(mf.diagnose).parameters) == ["observation"]
    for fn in (mf.diagnose, mf._novelty, mf._tokens):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        idents = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        idents |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        bad = [i for i in idents if re.search(r"inject|gold|required|InjectionRecord|TrialRecord|injected", i, re.I)]
        assert not bad, (fn.__name__, bad)


@pytest.mark.parametrize("condition", list(CONDITIONS))
@pytest.mark.parametrize("layer", FAULTS[1:])
def test_changing_only_the_hidden_label_changes_neither_input_nor_prediction(
        question, condition, layer, monkeypatch):
    seen = []
    real = mf.diagnose
    monkeypatch.setattr(mf, "diagnose", lambda obs: (seen.append(obs), real(obs))[1])
    base = mf.execute_fault_trial(question, condition, FaultType.NONE, fake_generate,
                                  apply_injection=False)
    relabelled = mf.execute_fault_trial(question, condition, layer, fake_generate,
                                        apply_injection=False)
    assert relabelled.injected_layer == layer and base.injected_layer == FaultType.NONE
    assert all(type(o) is mf.MultiHopObservation for o in seen)
    assert seen[0].model_dump_json() == seen[1].model_dump_json()
    assert base.predicted_layer == relabelled.predicted_layer == FaultType.NONE


@pytest.mark.parametrize("condition", list(CONDITIONS))
@pytest.mark.parametrize("layer", FAULTS)
def test_all_16_cells_inject_separately_and_diagnose_from_observation_only(
        question, condition, layer):
    rec = mf.execute_fault_trial(question, condition, layer, fake_generate)
    assert rec.predicted_layer == layer, (condition, layer, rec.predicted_layer, rec.injection)
    assert rec.injection.layer == layer
    n = len(question["conditions"][condition]["documents"])
    if layer == FaultType.RETRIEVAL:
        seen = sum(len(s.input_docs) for s in rec.observation.specialists)
        assert seen == n  # document count preserved
        assert rec.injection.details["replaced_doc_id"] != rec.injection.details["replacement_doc_id"]
    if layer == FaultType.ARBITER:
        assert len(rec.observation.arbiter_inputs) == len(rec.observation.specialists) - 1
    if layer == FaultType.NONE:
        assert rec.injection.transformation == "none"


def test_each_fault_changes_only_its_own_layer(question):
    cond = "B_highC_lowP"
    base = mf.execute_fault_trial(question, cond, FaultType.NONE, fake_generate).observation
    ret = mf.execute_fault_trial(question, cond, FaultType.RETRIEVAL, fake_generate).observation
    spec = mf.execute_fault_trial(question, cond, FaultType.SPECIALIST, fake_generate).observation
    arb = mf.execute_fault_trial(question, cond, FaultType.ARBITER, fake_generate).observation
    ids = lambda o: [s.retrieved_doc_ids for s in o.specialists]
    seen = lambda o: [[d["doc_id"] for d in s.input_docs] for s in o.specialists]
    assert ids(ret) == ids(base) and seen(ret) != seen(base)
    assert seen(spec) == seen(base) and ids(spec) == ids(base)
    assert seen(arb) == seen(base) and [s.output for s in arb.specialists] == [s.output for s in base.specialists]
    assert arb.arbiter_inputs != base.arbiter_inputs


def test_observation_text_contains_no_label_leaking_strings(question):
    for cond in CONDITIONS:
        for layer in FAULTS:
            blob = mf.execute_fault_trial(question, cond, layer, fake_generate) \
                .observation.model_dump_json().lower()
            for token in ("fault", "inject", "gold", "ground_truth", "synthetic"):
                assert token not in blob, (cond, layer, token)


def test_prediction_is_made_before_label_is_attached(question, monkeypatch):
    order = []
    real = mf.TrialRecord
    monkeypatch.setattr(mf, "TrialRecord", lambda **kw: (order.append("record"), real(**kw))[1])
    monkeypatch.setattr(mf, "diagnose", lambda obs: (order.append("diagnose"), FaultType.NONE)[1])
    mf.execute_fault_trial(question, "A_lowC_lowP", FaultType.ARBITER, fake_generate)
    assert order == ["diagnose", "record"]


def test_injection_is_deterministic(question):
    a = mf.execute_fault_trial(question, "D_highC_highP", FaultType.SPECIALIST, fake_generate)
    b = mf.execute_fault_trial(question, "D_highC_highP", FaultType.SPECIALIST, fake_generate)
    assert a.observation.model_dump_json() == b.observation.model_dump_json()
    assert a.injection == b.injection

def test_empty_specialist_context_abstains_without_model_call(question):
    condition = "B_highC_lowP"
    docs = question["conditions"][condition]["documents"]
    chunks = mf.assign_chunks(docs)

    # Remove all documents assigned to the first specialist.
    excluded = [d["doc_id"] for d in chunks[0]]
    calls = []

    def tracking_generate(role, context, query):
        calls.append((role, context))
        return fake_generate(role, context, query)

    rec = mf.execute_fault_trial(
        question,
        condition,
        FaultType.NONE,
        tracking_generate,
        apply_injection=False,
        exclude_doc_ids=excluded,
    )

    first = rec.observation.specialists[0]
    assert first.input_docs == []
    assert first.output == "Insufficient evidence: no documents provided."

    # Two remaining specialists plus the arbiter; no call for the empty one.
    assert len(calls) == 3
    assert all("No evidence provided." not in context for _, context in calls)
    
REAL_OK = Path("data/processed/multihoprag/pilot_30.json").exists() and \
    Path("data/raw/multihoprag/corpus.json").exists()


@pytest.mark.skipif(not REAL_OK, reason="real data not present")
def test_real_bundle_all_questions_all_cells_have_exact_coverage():
    payload = build_v2()
    cells = set()
    for q in payload["questions"]:
        for cond in CONDITIONS:
            for layer in FAULTS:
                rec = mf.execute_fault_trial(q, cond, layer, fake_generate)
                assert rec.predicted_layer == layer, (q["sample_id"], cond, layer)
                cells.add((q["sample_id"], cond, layer.value))
    assert len(cells) == len(payload["questions"]) * 16
