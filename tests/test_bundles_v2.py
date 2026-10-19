import json
from pathlib import Path

import pytest

from src.evaluation.build_bundles_v2 import (
    CONDITIONS,
    build_v2,
    locate_fact,
    sha256_file,
    split_sentences,
)

SENTS = [
    "Alpha sentence about the opening of the article with several plain words.",
    "The council approved the new transit budget on Monday after a long debate.",
    "Officials said the budget includes funding for three additional bus routes.",
    "Weather forecasters expect heavy rain across the northern valley this weekend.",
    "Local farmers reported that the late harvest was delayed by persistent cold nights.",
    "A museum announced a new exhibit of regional pottery opening in the spring season.",
    "Closing remarks from the mayor thanked residents for attending the public meeting.",
]


def make_fixture(tmp_path, fact_index=2):
    body = " ".join(SENTS)
    url = "https://example.test/a"
    corpus = [{"title": "T", "author": "A", "source": "Pub", "published_at": "2023-01-01",
               "category": "x", "url": url, "body": body}]
    pilot = {"selection_seed": 7, "questions": [{
        "sample_id": "q1", "original_index": 0, "question_type": "inference_query",
        "query": "Did the council approve funding for more bus routes?", "answer": "Yes",
        "evidence_list": [{"title": "T", "author": "A", "url": url, "source": "Pub",
                           "category": "x", "published_at": "2023-01-01",
                           "fact": SENTS[fact_index]}],
        "selection_seed": 7}]}
    pp, cp = tmp_path / "pilot.json", tmp_path / "corpus.json"
    pp.write_text(json.dumps(pilot), encoding="utf-8")
    cp.write_text(json.dumps(corpus), encoding="utf-8")
    return pp, cp, body


def test_equal_document_count_and_claim_coverage(tmp_path):
    pp, cp, _ = make_fixture(tmp_path)
    payload = build_v2(pp, cp)
    q = payload["questions"][0]
    counts = {c: len(v["documents"]) for c, v in q["conditions"].items()}
    assert set(counts) == set(CONDITIONS)
    assert set(counts.values()) == {2 * q["required_claim_count"]}
    for cond in CONDITIONS:
        claim0 = {d["evidence_unit_id"] for d in q["conditions"][cond]["documents"]
                  if d["role"] == "claim" and d["doc_id"].endswith("_c0")}
        assert claim0 == {"q1_ev_00"}


def test_only_intended_factor_changes(tmp_path):
    pp, cp, _ = make_fixture(tmp_path)
    c = build_v2(pp, cp)["questions"][0]["conditions"]

    def view(cond, drop):
        return [{k: v for k, v in d.items() if k not in drop}
                for d in c[cond]["documents"]]

    ignore_dep = {"doc_id", "experimental_lineage_id"}
    # A vs C and B vs D: identical content, order, hashes; only lineage differs.
    assert view("A_lowC_lowP", ignore_dep) == view("C_lowC_highP", ignore_dep)
    assert view("B_highC_lowP", ignore_dep) == view("D_highC_highP", ignore_dep)
    # A vs B: same slot positions; claim copy0 identical, slot_b content differs.
    a, b = c["A_lowC_lowP"]["documents"], c["B_highC_lowP"]["documents"]
    assert [d["slot_unit_id"] for d in a] == [d["slot_unit_id"] for d in b]
    assert {d["content_sha256"] for d in a} != {d["content_sha256"] for d in b}
    assert [d["role"] for d in b] == ["claim", "claim"]
    assert sorted(d["role"] for d in a) == ["claim", "filler"]


def test_lineage_definitions(tmp_path):
    pp, cp, _ = make_fixture(tmp_path)
    c = build_v2(pp, cp)["questions"][0]["conditions"]
    lin = lambda cond: [d["experimental_lineage_id"] for d in c[cond]["documents"]
                        if d["role"] == "claim"]
    assert len(set(lin("D_highC_highP"))) == 1 and len(lin("D_highC_highP")) == 2
    assert len(set(lin("B_highC_lowP"))) == 2
    assert len(set(lin("A_lowC_lowP"))) == 1 and len(set(lin("C_lowC_highP"))) == 1
    for cond in c:
        for d in c[cond]["documents"]:
            assert d["original_provenance_status"] == "unknown"
            assert d["lineage_is_synthetic"] is True


def test_passages_are_verbatim_corpus_windows_and_fillers_neutral(tmp_path):
    pp, cp, body = make_fixture(tmp_path)
    q = build_v2(pp, cp)["questions"][0]
    fact = q["source_evidence"][0]["fact"]
    for cond in q["conditions"].values():
        for d in cond["documents"]:
            assert body[d["source_char_start"]:d["source_char_end"]] == d["content"]
            if d["role"] == "claim":
                assert fact in d["content"]
            else:
                assert fact not in d["content"]
                assert d["filler_review_required"] is True


def test_unlocatable_fact_is_excluded_not_invented(tmp_path):
    pp, cp, _ = make_fixture(tmp_path)
    pilot = json.loads(pp.read_text(encoding="utf-8"))
    pilot["questions"][0]["evidence_list"][0]["fact"] = "This sentence is not in the article."
    pp.write_text(json.dumps(pilot), encoding="utf-8")
    payload = build_v2(pp, cp)
    assert payload["questions"] == []
    assert "not locatable" in payload["excluded_questions"][0]["reason"]


def test_builder_does_not_modify_inputs_and_is_deterministic(tmp_path):
    pp, cp, _ = make_fixture(tmp_path)
    before = (sha256_file(pp), sha256_file(cp))
    first, second = build_v2(pp, cp), build_v2(pp, cp)
    assert before == (sha256_file(pp), sha256_file(cp))
    assert first == second
    assert first["input_sha256"] == {"pilot": before[0], "corpus": before[1]}


def test_whitespace_tolerant_fact_location_and_splitter():
    body = "First sentence here.  The fact\nspans lines. Last one."
    span = locate_fact(body, "The fact spans lines.")
    assert span is not None
    assert len(split_sentences(body)) >= 2


REAL_PILOT = Path("data/processed/multihoprag/pilot_30.json")
REAL_CORPUS = Path("data/raw/multihoprag/corpus.json")


@pytest.mark.skipif(not (REAL_PILOT.exists() and REAL_CORPUS.exists()),
                    reason="real data not present")
def test_real_data_invariants():
    payload = build_v2()
    assert payload["sample_size"] + len(payload["excluded_questions"]) == 30
    for q in payload["questions"]:
        n = 2 * q["required_claim_count"]
        assert {len(v["documents"]) for v in q["conditions"].values()} == {n}
        ids = [d["doc_id"] for v in q["conditions"].values() for d in v["documents"]]
        assert len(ids) == len(set(ids))


def test_filler_pool_has_one_neutral_filler_per_claim(tmp_path):
    pp, cp, body = make_fixture(tmp_path)
    q = build_v2(pp, cp)["questions"][0]
    pool = q["filler_pool"]
    fact = q["source_evidence"][0]["fact"]
    assert len(pool) == q["required_claim_count"]
    for d in pool:
        assert d["role"] == "filler" and fact not in d["content"]
        assert d["doc_id"].startswith("pool_") and "fault" not in d["doc_id"]
        assert body[d["source_char_start"]:d["source_char_end"]] == d["content"]


def test_fillers_are_their_own_single_document_family(tmp_path):
    pp, cp, _ = make_fixture(tmp_path)
    c = build_v2(pp, cp)["questions"][0]["conditions"]
    for cond in c:
        fam = {}
        for d in c[cond]["documents"]:
            fam.setdefault(d["evidence_unit_id"], []).append(d["role"])
        for unit, roles in fam.items():
            assert len(set(roles)) == 1, (cond, unit, roles)
        sizes = sorted(len(v) for v in fam.values())
        if cond.startswith(("A", "C")):
            assert set(sizes) == {1}
        else:
            assert set(sizes) == {2}


def test_stale_bundle_is_rejected_with_regenerate_message(tmp_path):
    from src.evaluation.build_bundles_v2 import assert_current_bundle
    pp, cp, _ = make_fixture(tmp_path)
    fresh = build_v2(pp, cp)
    assert_current_bundle(fresh)  # current: no error
    stale = {**fresh, "builder_version": "v2.0"}
    with pytest.raises(SystemExit, match="Regenerate"):
        assert_current_bundle(stale)
    no_pool = json.loads(json.dumps(fresh))
    del no_pool["questions"][0]["filler_pool"]
    with pytest.raises(SystemExit, match="Regenerate"):
        assert_current_bundle(no_pool)
