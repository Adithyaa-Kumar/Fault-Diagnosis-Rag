"""Build controlled_bundles_v2.json: fixed document count, source-grounded passages.

Reads pilot_30.json and the raw corpus read-only. See docs/PHASE0_DESIGN_FREEZE.md.
Every passage is a verbatim window of a corpus article body; nothing is paraphrased.
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

BUILDER_VERSION = "v2.1"  # v2.1: filler families + slot_unit_id + filler_pool
PILOT = Path("data/processed/multihoprag/pilot_30.json")
CORPUS = Path("data/raw/multihoprag/corpus.json")
OUTPUT = Path("data/processed/multihoprag/controlled_bundles_v2.json")

CONDITIONS = {
    "A_lowC_lowP": {"redundancy": "low", "dependence": "low"},
    "B_highC_lowP": {"redundancy": "high", "dependence": "low"},
    "C_lowC_highP": {"redundancy": "low", "dependence": "high"},
    "D_highC_highP": {"redundancy": "high", "dependence": "high"},
}

MAX_CLAIM_CHARS = 1200
FILLER_MIN_WORDS, FILLER_MAX_WORDS, FILLER_MAX_CHARS = 15, 80, 600
STOPWORDS = frozenset(
    "the and for that with this from have has had are was were will would could "
    "should their there been being into about which while when what who whom "
    "than then them they you your not but also more most such only over some "
    "any all can may its our out one two new said says".split()
)


def assert_current_bundle(payload):
    """Fail fast (clear message) when a bundle predates the current builder."""
    if payload.get("builder_version") != BUILDER_VERSION or not all(
            "filler_pool" in q for q in payload.get("questions", [])):
        raise SystemExit(
            f"Stale bundle (builder_version={payload.get('builder_version')!r}, "
            f"current={BUILDER_VERSION!r}). Regenerate: python -m src.evaluation.build_bundles_v2")


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def split_sentences(body):
    """Return [(start, end)] char spans. Spans are only used to cut verbatim windows."""
    spans, start = [], 0
    for match in re.finditer(r"[.!?][\"\u201d\u2019')]*(?=\s+|$)|\n{2,}", body):
        end = match.end()
        if body[start:end].strip():
            spans.append((start, end))
        start = end
        while start < len(body) and body[start].isspace():
            start += 1
    if start < len(body) and body[start:].strip():
        spans.append((start, len(body)))
    return spans


def locate_fact(body, fact):
    """Return (start, end) of fact in body (whitespace-tolerant) or None."""
    fact = (fact or "").strip()
    if not fact:
        return None
    idx = body.find(fact)
    if idx >= 0:
        return idx, idx + len(fact)
    pattern = r"\s+".join(re.escape(tok) for tok in fact.split())
    match = re.search(pattern, body)
    return (match.start(), match.end()) if match else None


def content_tokens(text):
    return {t for t in re.findall(r"[a-z]{3,}", text.lower()) if t not in STOPWORDS}


def window_text(body, spans, first, last):
    return body[spans[first][0]:spans[last][1]]


def claim_windows(body, spans, fact_span):
    """Return two distinct verbatim windows containing the fact, or None."""
    fs, fe = fact_span
    covered = [i for i, (s, e) in enumerate(spans) if s < fe and e > fs]
    if not covered:
        return None
    f, l = covered[0], covered[-1]
    n = len(spans)
    candidates = [(f - 1, l), (f, l + 1), (f - 1, l + 1), (f, l + 2), (f - 2, l)]
    out = []
    for a, b in candidates:
        if a < 0 or b >= n:
            continue
        text = window_text(body, spans, a, b)
        if len(text) > MAX_CLAIM_CHARS or text in [w["text"] for w in out]:
            continue
        out.append({"text": text, "start": spans[a][0], "end": spans[b][1],
                    "sentence_range": [a, b]})
        if len(out) == 2:
            return out
    return None


def pick_filler(body, spans, forbidden_ranges, forbidden_facts, reference_tokens, used):
    best = None
    for i in range(len(spans) - 1):
        start, end = spans[i][0], spans[i + 1][1]
        text = body[start:end]
        words = len(text.split())
        if not (FILLER_MIN_WORDS <= words <= FILLER_MAX_WORDS) or len(text) > FILLER_MAX_CHARS:
            continue
        if any(start < fe and end > fs for fs, fe in forbidden_ranges):
            continue
        if any(start < ue and end > us for us, ue in used):
            continue
        if any(f and f in text for f in forbidden_facts):
            continue
        tokens = content_tokens(text)
        if not tokens:
            continue
        score = len(tokens & reference_tokens) / len(tokens)
        key = (score, start)
        if best is None or key < best[0]:
            best = (key, {"text": text, "start": start, "end": end,
                          "sentence_range": [i, i + 1], "lexical_overlap": round(score, 4)})
    return best[1] if best else None


def make_doc(cond, qid, unit_id, role, copy, window, meta, body_sha, lineage,
             transformation, extra=None):
    suffix = f"c{copy}" if role == "claim" else "fill"
    # Fillers form their own one-document family; slot_unit_id keeps the slot pairing.
    family_id = unit_id if role == "claim" else unit_id.replace("_ev_", "_fill_")
    doc = {
        "doc_id": f"{cond}_{unit_id}_{suffix}",
        "question_id": qid,
        "evidence_unit_id": family_id,
        "slot_unit_id": unit_id,
        "role": role,
        "content": window["text"],
        "parent_source_url": meta["url"],
        "source_publisher": meta["source"],
        "source_title": meta["title"],
        "published_at": meta["published_at"],
        "original_provenance_status": "unknown",
        "experimental_lineage_id": lineage,
        "lineage_is_synthetic": True,
        "transformation": transformation,
        "parent_ids": [meta["url"]],
        "source_char_start": window["start"],
        "source_char_end": window["end"],
        "source_sentence_range": window["sentence_range"],
        "source_corpus_body_sha256": body_sha,
        "content_sha256": sha256_text(window["text"]),
    }
    if extra:
        doc.update(extra)
    return doc


def build_question(question, corpus_by_url, seed):
    qid = question["sample_id"]
    evidence = question["evidence_list"]
    k = len(evidence)
    reference = content_tokens(question["query"] + " " + str(question["answer"]) +
                               " " + " ".join(e.get("fact", "") for e in evidence))
    facts = [e.get("fact", "").strip() for e in evidence]
    units, ranges_by_url = [], {}

    for index, item in enumerate(evidence):
        unit_id = f"{qid}_ev_{index:02d}"
        article = corpus_by_url.get(item.get("url"))
        if article is None:
            return None, f"{unit_id}: evidence URL not in corpus"
        body = article["body"]
        fact_span = locate_fact(body, item.get("fact"))
        if fact_span is None:
            return None, f"{unit_id}: fact not locatable in corpus body"
        spans = split_sentences(body)
        windows = claim_windows(body, spans, fact_span)
        if windows is None:
            return None, f"{unit_id}: fewer than two distinct claim windows"
        for w in windows:
            ranges_by_url.setdefault(item["url"], []).append((w["start"], w["end"]))
        ranges_by_url[item["url"]].append(fact_span)
        units.append({"unit_id": unit_id, "item": item, "body": body, "spans": spans,
                      "windows": windows, "body_sha": sha256_text(body)})

    used_by_url = {}
    for unit in units:
        url = unit["item"]["url"]
        used = used_by_url.setdefault(url, [])
        filler = pick_filler(unit["body"], unit["spans"], ranges_by_url[url], facts,
                             reference, used)
        if filler is None:
            return None, f"{unit['unit_id']}: no eligible filler window"
        used.append((filler["start"], filler["end"]))
        unit["filler"] = filler

    slots = [(u["unit_id"], slot) for u in units for slot in ("claim0", "slot_b")]
    random.Random(f"{seed}:{qid}").shuffle(slots)
    unit_by_id = {u["unit_id"]: u for u in units}

    conditions = {}
    for cond, design in CONDITIONS.items():
        high_red = design["redundancy"] == "high"
        high_dep = design["dependence"] == "high"
        docs = []
        for unit_id, slot in slots:
            unit = unit_by_id[unit_id]
            meta = unit["item"]
            if slot == "claim0":
                role, copy, win = "claim", 0, unit["windows"][0]
                transformation = "verbatim_corpus_window"
            elif high_red:
                role, copy, win = "claim", 1, unit["windows"][1]
                transformation = "verbatim_corpus_window_overlapping_copy0"
            else:
                role, copy, win = "filler", None, unit["filler"]
                transformation = "verbatim_corpus_window_filler"
            if role == "claim":
                lineage = (f"synthetic_shared_parent_{qid}" if high_dep
                           else f"synthetic_independent_{unit_id}_c{copy}")
            else:
                lineage = f"synthetic_filler_{unit_id}"
            extra = {"filler_review_required": True,
                     "neutrality_check": "lexical_overlap_only_unreviewed",
                     "lexical_overlap": win["lexical_overlap"]} if role == "filler" else None
            docs.append(make_doc(cond, qid, unit_id, role, copy, win, meta,
                                 unit["body_sha"], lineage, transformation, extra))
        conditions[cond] = {**design, "document_count": len(docs),
                            "lineage_semantics": ("synthetic_shared_parent" if high_dep
                                                  else "synthetic_distinct_assignments"),
                            "documents": docs}

    filler_pool = [
        make_doc("pool", qid, u["unit_id"], "filler", None, u["filler"], u["item"],
                 u["body_sha"], f"synthetic_filler_{u['unit_id']}",
                 "verbatim_corpus_window_filler",
                 {"filler_review_required": True,
                  "neutrality_check": "lexical_overlap_only_unreviewed",
                  "lexical_overlap": u["filler"]["lexical_overlap"]})
        for u in units
    ]

    return {
        "sample_id": qid,
        "original_index": question["original_index"],
        "question_type": question["question_type"],
        "query": question["query"],
        "answer": question["answer"],
        "source_evidence": evidence,
        "required_claim_count": k,
        "documents_per_condition": 2 * k,
        "review_status": "pending",
        "filler_pool": filler_pool,
        "conditions": conditions,
    }, None


def build_v2(pilot_path=PILOT, corpus_path=CORPUS, seed=None):
    pilot = load_json(pilot_path)
    corpus = load_json(corpus_path)
    seed = pilot.get("selection_seed") if seed is None else seed
    corpus_by_url = {a["url"]: a for a in corpus}
    questions, excluded = [], []
    for question in pilot["questions"]:
        built, reason = build_question(question, corpus_by_url, seed)
        if built is None:
            excluded.append({"sample_id": question["sample_id"], "reason": reason})
        else:
            questions.append(built)
    return {
        "dataset": "MultiHop-RAG",
        "artifact_type": "controlled_grounded_passage_intervention_v2",
        "builder_version": BUILDER_VERSION,
        "design_doc": "docs/PHASE0_DESIGN_FREEZE.md",
        "seed": seed,
        "input_sha256": {"pilot": sha256_file(pilot_path), "corpus": sha256_file(corpus_path)},
        "sample_size": len(questions),
        "excluded_questions": excluded,
        "conditions": CONDITIONS,
        "important_limitations": [
            "Lineage IDs are synthetic experimental assignments, not verified provenance.",
            "Low-dependence copy0/copy1 come from the same article URL; independence is a label only.",
            "Filler passages passed a lexical screen only and require human review.",
            "Claim support and gold answers are not validated by this builder.",
        ],
        "questions": questions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", default=str(PILOT))
    parser.add_argument("--corpus", default=str(CORPUS))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    payload = build_v2(args.pilot, args.corpus)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}: {payload['sample_size']} questions, "
          f"{len(payload['excluded_questions'])} excluded")
    for item in payload["excluded_questions"]:
        print("  excluded", item["sample_id"], "-", item["reason"])


if __name__ == "__main__":
    main()
