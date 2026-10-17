
import csv
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "multihoprag"
PROCESSED_DIR = ROOT / "data" / "processed"

QUESTIONS_FILE = RAW_DIR / "MultiHopRAG.json"
CORPUS_FILE = RAW_DIR / "corpus.json"

MANIFEST_FILE = PROCESSED_DIR / "multihoprag_intervention_manifest.csv"
SUMMARY_FILE = PROCESSED_DIR / "multihoprag_manifest_summary.json"


def normalize_url(url):
    """Normalize superficial URL differences without inferring lineage."""
    if not url:
        return ""

    try:
        parts = urlsplit(url.strip())
        host = parts.netloc.lower().removeprefix("www.")
        path = parts.path.rstrip("/")
        return urlunsplit(
            (parts.scheme.lower(), host, path, "", "")
        )
    except ValueError:
        return url.strip()


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list in {path.name}, "
            f"got {type(data).__name__}"
        )

    return data


def main():
    questions = load_json(QUESTIONS_FILE)
    articles = load_json(CORPUS_FILE)

    # Preserve every article record; do not change the raw corpus.
    articles_by_url = {}
    for article in articles:
        url = normalize_url(article.get("url", ""))
        if url:
            articles_by_url.setdefault(url, []).append(article)

    rows = []
    issue_counts = Counter()
    question_ids_seen = set()

    for q_index, question in enumerate(questions):
        query = question.get("query", "")
        answer = question.get("answer", "")
        evidence_list = question.get("evidence_list", [])

        question_id = str(
            question.get("id")
            or question.get("_id")
            or f"multihoprag_{q_index:05d}"
        )

        if question_id in question_ids_seen:
            issue_counts["duplicate_question_id"] += 1
        question_ids_seen.add(question_id)

        question_issues = []
        if not isinstance(query, str) or not query.strip():
            question_issues.append("missing_query")

        if answer is None or (
            isinstance(answer, str) and not answer.strip()
        ):
            question_issues.append("missing_answer")

        if not isinstance(evidence_list, list) or not evidence_list:
            issue_counts["question_without_evidence"] += 1
            continue

        for e_index, evidence in enumerate(evidence_list):
            if not isinstance(evidence, dict):
                issue_counts["malformed_evidence_record"] += 1
                continue

            url = evidence.get("url", "")
            normalized_url = normalize_url(url)
            matching_articles = articles_by_url.get(
                normalized_url, []
            )

            if len(matching_articles) == 1:
                corpus_match = "unique_url_match"
                article = matching_articles[0]
            elif len(matching_articles) > 1:
                corpus_match = "ambiguous_url_match"
                article = matching_articles[0]
                issue_counts["ambiguous_url_match"] += 1
            else:
                corpus_match = "no_url_match"
                article = {}
                issue_counts["no_url_match"] += 1

            if not url:
                issue_counts["missing_evidence_url"] += 1

            if not evidence.get("fact"):
                issue_counts["missing_evidence_fact"] += 1

            if not evidence.get("title"):
                issue_counts["missing_evidence_title"] += 1

            row_issues = list(question_issues)
            if corpus_match != "unique_url_match":
                row_issues.append(corpus_match)

            # Deliberately do not assign lineage clusters from publisher,
            # URL, title, text overlap, or shared topic alone.
            rows.append({
                "question_id": question_id,
                "question_index": q_index,
                "query": query if isinstance(query, str) else "",
                "gold_answer": (
                    answer if isinstance(answer, str)
                    else json.dumps(answer, ensure_ascii=False)
                ),
                "question_type": question.get("question_type", ""),
                "evidence_index": e_index,
                "evidence_id": f"{question_id}_e{e_index:02d}",
                "evidence_title": evidence.get("title", ""),
                "evidence_fact": evidence.get("fact", ""),
                "evidence_author": evidence.get("author", ""),
                "evidence_publisher": evidence.get("source", ""),
                "evidence_category": evidence.get("category", ""),
                "evidence_published_at": evidence.get(
                    "published_at", ""
                ),
                "evidence_url": url,
                "corpus_match_status": corpus_match,
                "corpus_title": article.get("title", ""),
                "corpus_publisher": article.get("source", ""),
                "corpus_url": article.get("url", ""),
                "lineage_status": "unknown",
                "lineage_cluster_id": "",
                "lineage_evidence": "",
                "eligible_for_experiment": "no",
                "condition": "",
                "injected_fault_layer": "",
                "validation_issues": ";".join(row_issues),
            })

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if rows:
        with MANIFEST_FILE.open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=list(rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(rows)

    unique_questions = len({r["question_id"] for r in rows})
    summary = {
        "dataset": "MultiHop-RAG",
        "raw_question_count": len(questions),
        "raw_corpus_article_count": len(articles),
        "manifest_evidence_row_count": len(rows),
        "questions_with_manifest_rows": unique_questions,
        "unique_corpus_urls": len(articles_by_url),
        "corpus_url_match_counts": dict(
            Counter(r["corpus_match_status"] for r in rows)
        ),
        "lineage_status_counts": dict(
            Counter(r["lineage_status"] for r in rows)
        ),
        "experiment_eligibility_counts": dict(
            Counter(r["eligible_for_experiment"] for r in rows)
        ),
        "issue_counts": dict(issue_counts),
        "condition_assignment_performed": False,
        "fault_injection_performed": False,
        "warning": (
            "URL matching does not establish provenance dependence "
            "or independence. All lineage labels remain unknown; "
            "no experimental conditions or faults were assigned."
        ),
        "manifest_path": str(MANIFEST_FILE),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== MULTIHOP-RAG INTERVENTION MANIFEST ===")
    print("Raw questions:", len(questions))
    print("Raw corpus articles:", len(articles))
    print("Evidence rows:", len(rows))
    print("Questions represented:", unique_questions)
    print("URL match counts:", summary["corpus_url_match_counts"])
    print("Lineage labels:", summary["lineage_status_counts"])
    print("Issues:", dict(issue_counts))
    print("Conditions assigned: No")
    print("Faults injected: No")
    print("Manifest:", MANIFEST_FILE)
    print("Summary:", SUMMARY_FILE)


if __name__ == "__main__":
    main()
