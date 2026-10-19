import json
from collections import Counter
from pathlib import Path

import pytest

from src.evaluation import build_phase0_review_sheet as rs

BUNDLE = Path("data/processed/multihoprag/controlled_bundles_v2.json")
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="bundle not generated")


def load():
    return json.loads(BUNDLE.read_text(encoding="utf-8"))


def test_selection_is_balanced_deterministic_and_excludes_flagged():
    b = load()
    ids = rs.select_sample(b)
    assert ids == rs.select_sample(b) and len(ids) == 6
    types = Counter(q["question_type"] for q in b["questions"] if q["sample_id"] in ids)
    assert set(types.values()) == {2} and len(types) == 3
    assert not set(ids) & set(rs.EXCLUDED)
    assert all(q["required_claim_count"] <= rs.MAX_K for q in b["questions"] if q["sample_id"] in ids)


def test_sheet_contains_every_passage_and_filler_and_draft_is_unreviewed():
    b = load()
    ids = rs.select_sample(b)
    sheet = rs.render_sheet(b, ids)
    for q in b["questions"]:
        if q["sample_id"] in ids:
            assert q["answer"] in sheet and q["query"] in sheet
            for p in q["filler_pool"]:
                assert p["content"].strip() in sheet
    m = rs.draft_manifest(b, ids)
    assert m["reviewed"] is False and m["sample_ids"] == ids


def test_reject_replaces_only_that_question_with_same_type_next_ranked():
    b = load()
    base = rs.select_sample(b)
    by = {q["sample_id"]: q for q in b["questions"]}
    victim = base[0]
    new = rs.select_sample(b, rejected=[victim])
    assert victim not in new and len(new) == 6
    assert [i for i in new if i in base] == [i for i in base if i != victim]  # others unchanged
    added = [i for i in new if i not in base]
    assert len(added) == 1 and by[added[0]]["question_type"] == by[victim]["question_type"]
    assert rs.draft_manifest(b, new, [victim])["rejected_in_review"] == [victim]


def test_sample_never_contains_prompt_dev_or_flagged_questions_even_after_rejections():
    b = load()
    for rejected in ([], ["mh_0791"], ["mh_0791", "mh_0836"]):
        ids = rs.select_sample(b, rejected=rejected)
        assert not set(ids) & (set(rs.PROMPT_DEV_USED) | set(rs.EXCLUDED) | set(rejected))
        assert len(ids) == 6
