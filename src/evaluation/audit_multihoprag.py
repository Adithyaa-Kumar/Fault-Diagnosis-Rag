import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

BASE = Path("data/raw/multihoprag")
OUT = Path("results/multihoprag")
QUESTION_FILE = BASE / "MultiHopRAG.json"
CORPUS_FILE = BASE / "corpus.json"


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_url(url):
    """Conservative URL normalization; do not merge different URL paths."""
    if not isinstance(url, str) or not url.strip():
        return ""
    parts = urlsplit(url.strip())
    return urlunsplit((
        parts.scheme.lower(),
        parts.netloc.lower(),
        parts.path.rstrip("/"),
        parts.query,
        "",  # URL fragments do not identify a different fetched article
    ))


def fingerprint(text):
    normalized = " ".join((text or "").split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def main():
    questions = read_json(QUESTION_FILE)
    corpus = read_json(CORPUS_FILE)

    # URL -> article. The audit stops if URLs are ambiguous.
    corpus_by_url = {}
    duplicate_corpus_urls = Counter()

    for article in corpus:
        url = normalize_url(article.get("url"))
        if not url:
            continue
        if url in corpus_by_url:
            duplicate_corpus_urls[url] += 1
        else:
            corpus_by_url[url] = article

    if duplicate_corpus_urls:
        raise SystemExit(
            f"Ambiguous corpus URLs found: {len(duplicate_corpus_urls)}. "
            "Resolve them before auditing evidence."
        )

    rows = []
    evidence_count = 0
    empty_evidence_questions = 0
    unmatched_evidence = 0
    missing_urls = 0
    matched_count = 0
    source_counts = Counter()
    question_type_counts = Counter()

    for question_index, question in enumerate(questions):
        query = question.get("query", "")
        qid = f"q_{question_index:05d}"
        question_type = question.get("question_type") or "MISSING"
        question_type_counts[question_type] += 1

        evidence_list = question.get("evidence_list")
        if not isinstance(evidence_list, list) or not evidence_list:
            empty_evidence_questions += 1
            continue

        for evidence_index, evidence in enumerate(evidence_list):
            if not isinstance(evidence, dict):
                continue

            evidence_count += 1
            url = normalize_url(evidence.get("url"))
            source = str(evidence.get("source") or "").strip()
            source_counts[source or "MISSING"] += 1

            article = corpus_by_url.get(url) if url else None
            if not url:
                missing_urls += 1
            elif article is None:
                unmatched_evidence += 1
            else:
                matched_count += 1

            fact = str(evidence.get("fact") or "")
            body = str(article.get("body") or "") if article else ""

            # Fingerprints support exact normalized-text comparisons only.
            # They do not establish semantic equivalence or source lineage.
            fact_hash = fingerprint(fact) if fact else ""
            body_hash = fingerprint(body) if body else ""

            rows.append({
                "question_id": qid,
                "question_type": question_type,
                "evidence_index": evidence_index,
                "query": query,
                "evidence_title": evidence.get("title") or "",
                "evidence_url": url,
                "evidence_source": source,
                "evidence_published_at": evidence.get("published_at") or "",
                "evidence_fact": fact,
                "fact_sha256_normalized": fact_hash,
                "corpus_match": "matched" if article else "unmatched",
                "corpus_title": article.get("title", "") if article else "",
                "corpus_source": article.get("source", "") if article else "",
                "corpus_published_at": article.get("published_at", "") if article else "",
                "corpus_body_sha256_normalized": body_hash,
                "provenance_status": "unknown",
                "provenance_basis": "Dataset metadata does not establish editorial/source lineage",
            })

    OUT.mkdir(parents=True, exist_ok=True)
    audit_path = OUT / "evidence_audit.csv"
    summary_path = OUT / "audit_summary.json"

    columns = list(rows[0].keys()) if rows else []
    with audit_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": "MultiHop-RAG",
        "raw_files_modified": False,
        "question_count": len(questions),
        "corpus_article_count": len(corpus),
        "question_types": dict(question_type_counts),
        "evidence_record_count": evidence_count,
        "questions_without_evidence": empty_evidence_questions,
        "matched_evidence_records": matched_count,
        "unmatched_evidence_records": unmatched_evidence,
        "evidence_records_without_url": missing_urls,
        "distinct_normalized_corpus_urls": len(corpus_by_url),
        "duplicate_corpus_url_groups": len(duplicate_corpus_urls),
        "evidence_source_counts": dict(source_counts),
        "provenance_status_policy": {
            "unknown": "Default for every record until source lineage is documented",
            "verified_dependent": "Requires documented common origin or derivation",
            "verified_independent": "Requires evidence of independent origins",
        },
        "limitations": [
            "URL matching establishes metadata linkage, not factual support.",
            "Different publishers do not prove independent provenance.",
            "Text hashes detect exact normalized-text equality, not semantic similarity.",
            "Question IDs are stable only while the raw question order remains unchanged.",
        ],
        "outputs": [str(audit_path), str(summary_path)],
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("MULTIHOP-RAG EVIDENCE AUDIT COMPLETE")
    print(json.dumps({
        "questions": len(questions),
        "corpus_articles": len(corpus),
        "evidence_records": evidence_count,
        "questions_without_evidence": empty_evidence_questions,
        "matched_evidence_records": matched_count,
        "unmatched_evidence_records": unmatched_evidence,
        "evidence_records_without_url": missing_urls,
        "provenance_labels": "unknown for all records pending lineage evidence",
        "audit_csv": str(audit_path),
        "summary_json": str(summary_path),
        "raw_files_modified": False,
    }, indent=2))


if __name__ == "__main__":
    main()
