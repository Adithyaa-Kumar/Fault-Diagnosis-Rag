
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "multihoprag"
OUT_DIR = ROOT / "data" / "processed"

QUESTIONS_FILE = RAW_DIR / "MultiHopRAG.json"
CORPUS_FILE = RAW_DIR / "corpus.json"

OUTPUT_CSV = OUT_DIR / "multihoprag_consolidated_manifest.csv"
OUTPUT_JSON = OUT_DIR / "multihoprag_consolidated_manifest_summary.json"

SEMANTIC_FILE = OUT_DIR / "multihoprag_semantic_redundancy_pairs.csv"
REPEATED_FILE = OUT_DIR / "multihoprag_repeated_evidence_sets.json"


def clean(value):
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def get_question_id(question, index):
    return clean(question.get("_id", question.get("id", index)))


def make_evidence_signature(evidence_list):
    pairs = []
    for evidence in evidence_list or []:
        if isinstance(evidence, dict):
            pairs.append((
                clean(evidence.get("url", "")).lower(),
                clean(evidence.get("fact", "")).lower(),
            ))

    payload = json.dumps(sorted(pairs), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_json(QUESTIONS_FILE)
    corpus = load_json(CORPUS_FILE)

    # Index corpus articles by normalized URL.
    corpus_by_url = defaultdict(list)

    for corpus_index, article in enumerate(corpus):
        url = clean(article.get("url", "")).rstrip("/").lower()
        if url:
            corpus_by_url[url].append((corpus_index, article))

    # Load existing audit outputs without modifying them.
    semantic_rows = load_csv(SEMANTIC_FILE)

    repeated_data = (
        load_json(REPEATED_FILE) if REPEATED_FILE.exists() else {}
    )

    repeated_group_by_question = {}

    for group in repeated_data.get("repeated_groups", []):
        for member in group.get("members", []):
            qid = clean(member.get("question_id", ""))
            repeated_group_by_question[qid] = group.get("group_id", "")

    # Index existing TF-IDF pair scores by question and fact pair.
    similarity_index = {}

    for row in semantic_rows:
        qid = clean(row.get("question_id", ""))
        left = clean(row.get("left_fact", ""))
        right = clean(row.get("right_fact", ""))

        try:
            score = float(row["tfidf_cosine_similarity"])
        except (KeyError, TypeError, ValueError):
            continue

        key = (qid, tuple(sorted((left, right))))
        similarity_index[key] = max(
            score, similarity_index.get(key, -1.0)
        )

    manifest = []
    signature_to_questions = defaultdict(list)

    empty_questions = 0
    evidence_count = 0
    unique_url_matches = 0
    ambiguous_url_matches = 0
    unmatched_urls = 0

    for question_index, question in enumerate(questions):
        qid = get_question_id(question, question_index)
        evidence_list = question.get("evidence_list") or []

        if not evidence_list:
            empty_questions += 1
            continue

        signature = make_evidence_signature(evidence_list)
        signature_to_questions[signature].append(qid)

        for evidence_index, evidence in enumerate(evidence_list):
            if not isinstance(evidence, dict):
                continue

            evidence_count += 1

            url = clean(evidence.get("url", ""))
            normalized_url = url.rstrip("/").lower()
            matches = corpus_by_url.get(normalized_url, [])

            corpus_index = ""
            corpus_article = {}

            if len(matches) == 1:
                corpus_index, corpus_article = matches[0]
                match_status = "unique_url_match"
                unique_url_matches += 1
            elif len(matches) > 1:
                corpus_index, corpus_article = matches[0]
                match_status = "ambiguous_url_match"
                ambiguous_url_matches += 1
            else:
                match_status = "no_url_match"
                unmatched_urls += 1

            fact = clean(evidence.get("fact", ""))
            pair_scores = []

            for other_index, other in enumerate(evidence_list):
                if other_index == evidence_index or not isinstance(other, dict):
                    continue

                other_fact = clean(other.get("fact", ""))
                key = (qid, tuple(sorted((fact, other_fact))))

                if key in similarity_index:
                    pair_scores.append(similarity_index[key])

            group_id = repeated_group_by_question.get(qid, "")

            manifest.append({
                "question_id": qid,
                "question_index": question_index,
                "query": clean(question.get("query", "")),
                "gold_answer": clean(question.get("answer", "")),
                "question_type": clean(question.get("question_type", "")),
                "evidence_signature": signature,
                "evidence_set_group_id": group_id,
                "evidence_set_group_status": (
                    "repeated" if group_id else "not_in_repeated_group_audit"
                ),
                "evidence_count_for_question": len(evidence_list),
                "evidence_index": evidence_index,
                "evidence_id": clean(evidence.get("id", "")),
                "evidence_title": clean(evidence.get("title", "")),
                "evidence_fact": fact,
                "evidence_author": clean(evidence.get("author", "")),
                "evidence_publisher": clean(evidence.get("source", "")),
                "evidence_category": clean(evidence.get("category", "")),
                "evidence_published_at": clean(evidence.get("published_at", "")),
                "evidence_url": url,
                "corpus_match_status": match_status,
                "corpus_index": corpus_index,
                "corpus_title": clean(corpus_article.get("title", "")),
                "corpus_author": clean(corpus_article.get("author", "")),
                "corpus_publisher": clean(corpus_article.get("source", "")),
                "corpus_published_at": clean(
                    corpus_article.get("published_at", "")
                ),
                "corpus_url": clean(corpus_article.get("url", "")),
                "corpus_body_present": bool(
                    clean(corpus_article.get("body", ""))
                ),
                "max_pair_tfidf_similarity_in_question": (
                    max(pair_scores) if pair_scores else ""
                ),
                "lineage_status": "unknown",
                "lineage_cluster_id": "",
                "lineage_evidence_note": "",
                "redundancy_label": "unlabeled",
                "redundancy_evidence_note": "",
                "condition_assignment": "",
                "fault_assignment": "",
                "eligible_for_experiment": "no",
                "eligibility_blocker": (
                    "Provenance and redundancy require validation; "
                    "experimental condition not assigned."
                ),
            })

    # Count distinct evidence sets and repeated-set group sizes.
    group_sizes = Counter(
        len(question_ids)
        for question_ids in signature_to_questions.values()
    )

    repeated_group_count = sum(
        count for size, count in group_sizes.items() if size > 1
    )

    questions_in_repeated_groups = sum(
        size * count
        for size, count in group_sizes.items()
        if size > 1
    )

    questions_in_singleton_groups = sum(
        count for size, count in group_sizes.items() if size == 1
    )

    # Write consolidated CSV.
    fieldnames = list(manifest[0].keys()) if manifest else []

    with OUTPUT_CSV.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    summary = {
        "dataset": "MultiHop-RAG",
        "raw_question_count": len(questions),
        "questions_with_evidence": len(questions) - empty_questions,
        "questions_without_evidence": empty_questions,
        "corpus_article_count": len(corpus),
        "manifest_evidence_rows": len(manifest),
        "unique_evidence_set_count": len(signature_to_questions),
        "repeated_evidence_set_group_count": repeated_group_count,
        "questions_in_repeated_evidence_groups": (
            questions_in_repeated_groups
        ),
        "questions_in_singleton_evidence_groups": (
            questions_in_singleton_groups
        ),
        "evidence_set_group_size_distribution": {
            str(size): count
            for size, count in sorted(group_sizes.items())
        },
        "corpus_url_matches": {
            "unique": unique_url_matches,
            "ambiguous": ambiguous_url_matches,
            "unmatched": unmatched_urls,
        },
        "existing_semantic_pair_rows_loaded": len(semantic_rows),
        "provenance_status": (
            "unknown unless documented evidence establishes lineage"
        ),
        "redundancy_status": (
            "unlabeled; TF-IDF scores are screening signals only"
        ),
        "experimental_conditions_assigned": False,
        "faults_assigned": False,
        "eligible_manifest_rows": 0,
        "limitations": [
            "Repeated evidence sets are not provenance clusters.",
            "Different publishers do not prove independent source lineage.",
            "Lexical similarity does not prove claim-level redundancy.",
            "No experimental condition has been assigned.",
        ],
        "output_csv": str(OUTPUT_CSV.relative_to(ROOT)),
        "output_summary": str(OUTPUT_JSON.relative_to(ROOT)),
    }

    OUTPUT_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nConsolidated manifest build completed.")
    print("Raw questions:", len(questions))
    print("Questions with evidence:", len(questions) - empty_questions)
    print("Questions without evidence:", empty_questions)
    print("Corpus articles:", len(corpus))
    print("Manifest evidence rows:", len(manifest))
    print("Unique evidence sets:", len(signature_to_questions))
    print("Repeated evidence-set groups:", repeated_group_count)
    print("Questions in repeated groups:", questions_in_repeated_groups)
    print("Questions in singleton groups:", questions_in_singleton_groups)
    print("Unique corpus URL matches:", unique_url_matches)
    print("Ambiguous corpus URL matches:", ambiguous_url_matches)
    print("Unmatched corpus URLs:", unmatched_urls)
    print("\nNo provenance labels, redundancy labels, conditions, or faults assigned.")
    print("Manifest:", OUTPUT_CSV)
    print("Summary: ", OUTPUT_JSON)


if __name__ == "__main__":
    main()
