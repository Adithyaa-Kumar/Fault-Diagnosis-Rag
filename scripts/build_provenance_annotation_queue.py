
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "multihoprag" / "corpus.json"
OUT_DIR = ROOT / "data" / "processed"

QUEUE_FILE = OUT_DIR / "multihoprag_provenance_annotation_queue.csv"
SUMMARY_FILE = OUT_DIR / "multihoprag_provenance_annotation_summary.json"

# These expressions identify passages for review, not confirmed lineage.
ATTRIBUTION_PATTERNS = [
    re.compile(
        r"\baccording to\b.{0,180}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\breported by\b.{0,180}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfirst reported by\b.{0,180}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\boriginally reported by\b.{0,180}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\boriginally published by\b.{0,180}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:via|per)\s+[A-Z][A-Za-z0-9&.' -]{1,70}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:Reuters|Associated Press|AP News|AFP|Bloomberg)"
        r"\s+(?:reported|said|wrote|confirmed|according to)\b.{0,180}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:press release|statement|filing|court document)"
        r"\b.{0,180}",
        re.IGNORECASE,
    ),
]

# Explicitly preserve source attribution as a candidate, not a label.
LINEAGE_STATUS = "unknown"


def clean(value):
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def extract_contexts(body):
    contexts = []

    for pattern in ATTRIBUTION_PATTERNS:
        for match in pattern.finditer(body):
            start = max(0, match.start() - 100)
            end = min(len(body), match.end() + 100)

            snippet = clean(body[start:end])
            if snippet and snippet not in contexts:
                contexts.append(snippet)

    return contexts


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with RAW.open("r", encoding="utf-8") as f:
        articles = json.load(f)

    rows = []
    attribution_article_count = 0
    total_contexts = 0
    missing_body_count = 0
    publisher_counts = Counter()

    for index, article in enumerate(articles):
        body = clean(article.get("body", ""))

        if not body:
            missing_body_count += 1

        contexts = extract_contexts(body)
        publisher = clean(article.get("source", ""))
        publisher_counts[publisher or "(unknown)"] += 1

        if contexts:
            attribution_article_count += 1

        total_contexts += len(contexts)

        # One row per article so even articles with no attribution
        # passages remain available for manual annotation.
        rows.append({
            "article_id": f"article_{index:04d}",
            "corpus_index": index,
            "title": clean(article.get("title", "")),
            "publisher": publisher,
            "author": clean(article.get("author", "")),
            "published_at": clean(article.get("published_at", "")),
            "category": clean(article.get("category", "")),
            "url": clean(article.get("url", "")),
            "body_present": bool(body),
            "body_length_chars": len(body),
            "attribution_context_count": len(contexts),
            "attribution_contexts": " || ".join(contexts),
            "lineage_status": LINEAGE_STATUS,
            "candidate_parent_article_id": "",
            "lineage_evidence_url_or_quote": "",
            "reviewer_decision": "pending",
            "reviewer_notes": "",
        })

    fields = list(rows[0].keys()) if rows else []

    with QUEUE_FILE.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "corpus_articles": len(articles),
        "articles_with_body": len(articles) - missing_body_count,
        "articles_without_body": missing_body_count,
        "articles_with_attribution_candidates": attribution_article_count,
        "total_attribution_contexts": total_contexts,
        "publisher_count": len(publisher_counts),
        "publisher_distribution": dict(
            publisher_counts.most_common()
        ),
        "annotation_policy": {
            "initial_lineage_status": "unknown",
            "candidate_passages_are_proof": False,
            "different_publishers_prove_independence": False,
            "same_publisher_proves_dependence": False,
            "required_for_confirmed_dependence": (
                "A verifiable citation, explicit syndication relationship, "
                "identified common upstream document, or equivalent "
                "documented source-lineage evidence."
            ),
            "required_for_independence": (
                "Documented evidence sufficient to support independent "
                "origins; absence of a citation is not proof."
            ),
        },
        "outputs": [
            str(QUEUE_FILE.relative_to(ROOT)),
            str(SUMMARY_FILE.relative_to(ROOT)),
        ],
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nProvenance annotation queue created.")
    print("Corpus articles:", len(articles))
    print("Articles with body:", len(articles) - missing_body_count)
    print("Articles without body:", missing_body_count)
    print("Articles with attribution candidates:", attribution_article_count)
    print("Candidate attribution passages:", total_contexts)
    print("Publishers:", len(publisher_counts))
    print("\nNo lineage relationships were automatically confirmed.")
    print("Queue:", QUEUE_FILE)
    print("Summary:", SUMMARY_FILE)


if __name__ == "__main__":
    main()
