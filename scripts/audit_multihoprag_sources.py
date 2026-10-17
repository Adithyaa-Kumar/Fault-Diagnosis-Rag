
import csv
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "multihoprag"
OUT_DIR = ROOT / "data" / "processed"

QUESTIONS_FILE = DATA_DIR / "MultiHopRAG.json"
CORPUS_FILE = DATA_DIR / "corpus.json"
REPORT_FILE = OUT_DIR / "multihoprag_source_audit.csv"


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_url(url):
    if not url:
        return ""

    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    path = re.sub(r"/+$", "", parts.path) or "/"

    # Remove fragments and query parameters for comparison.
    return urlunsplit(
        (parts.scheme.lower(), host, path, "", "")
    )


def domain(url):
    if not url:
        return ""
    return urlsplit(url).netloc.lower().removeprefix("www.")


def main():
    questions = load_json(QUESTIONS_FILE)
    corpus = load_json(CORPUS_FILE)

    corpus_by_url = {}
    corpus_domains = Counter()
    corpus_publishers = Counter()

    for article in corpus:
        key = normalize_url(article.get("url", ""))
        if key:
            corpus_by_url.setdefault(key, []).append(article)

        corpus_domains[domain(article.get("url", ""))] += 1
        if article.get("source"):
            corpus_publishers[article["source"]] += 1

    total_evidence = 0
    matched_evidence = 0
    questions_with_multiple_domains = 0
    questions_with_multiple_publishers = 0
    rows = []

    for index, question in enumerate(questions):
        evidence = question.get("evidence_list", [])
        evidence_domains = set()
        evidence_publishers = set()
        evidence_urls = set()
        matched_count = 0

        for item in evidence:
            url = item.get("url", "")
            key = normalize_url(url)
            matches = corpus_by_url.get(key, []) if key else []

            total_evidence += 1
            if matches:
                matched_evidence += 1
                matched_count += 1

            d = domain(url)
            if d:
                evidence_domains.add(d)

            if item.get("source"):
                evidence_publishers.add(item["source"])

            if key:
                evidence_urls.add(key)

            rows.append({
                "question_index": index,
                "query": question.get("query", ""),
                "answer": question.get("answer", ""),
                "question_type": question.get("question_type", ""),
                "evidence_title": item.get("title", ""),
                "evidence_url": url,
                "publisher": item.get("source", ""),
                "domain": d,
                "normalized_url": key,
                "matched_corpus_url": bool(matches),
                "corpus_match_count": len(matches),
                "distinct_domains_in_question": "",
                "distinct_publishers_in_question": "",
                "lineage_status": "unknown",
                "lineage_notes": "",
            })

        if len(evidence_domains) >= 2:
            questions_with_multiple_domains += 1

        if len(evidence_publishers) >= 2:
            questions_with_multiple_publishers += 1

        # Add question-level counts to its evidence rows.
        for row in rows:
            if row["question_index"] == index:
                row["distinct_domains_in_question"] = len(evidence_domains)
                row["distinct_publishers_in_question"] = len(
                    evidence_publishers
                )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if rows:
        with REPORT_FILE.open(
            "w", newline="", encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print("=== MULTIHOP-RAG SOURCE AUDIT ===")
    print("Questions:", len(questions))
    print("Corpus articles:", len(corpus))
    print("Evidence records:", total_evidence)
    print("Evidence URLs matching corpus URLs:", matched_evidence)
    print(
        "Evidence URL match rate:",
        f"{matched_evidence / total_evidence:.1%}"
        if total_evidence else "N/A",
    )
    print(
        "Questions with multiple evidence domains:",
        questions_with_multiple_domains,
    )
    print(
        "Questions with multiple publisher labels:",
        questions_with_multiple_publishers,
    )
    print("Distinct corpus domains:", len(corpus_domains))
    print("Distinct corpus publishers:", len(corpus_publishers))
    print("Report:", REPORT_FILE)
    print()
    print("IMPORTANT: Domain or publisher differences do not prove")
    print("independent provenance. All lineage labels remain unknown.")


if __name__ == "__main__":
    main()
