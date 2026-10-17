import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "multihoprag" / "corpus.json"
QUEUE = ROOT / "data" / "processed" / "multihoprag_provenance_annotation_queue.csv"
OUT = ROOT / "data" / "processed"

OUTPUT_CSV = OUT / "multihoprag_provenance_candidate_graph.csv"
OUTPUT_JSON = OUT / "multihoprag_provenance_candidate_graph_summary.json"


def clean(value):
    return " ".join(str(value or "").split()).strip()


def publisher_pattern(name):
    # Match publisher names as phrases, not arbitrary substrings.
    escaped = re.escape(name.strip())
    return re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)


def main():
    with RAW.open("r", encoding="utf-8") as f:
        articles = json.load(f)

    with QUEUE.open("r", encoding="utf-8-sig", newline="") as f:
        queue = list(csv.DictReader(f))

    # Stable article identifiers follow the corpus order.
    publishers = {}
    article_info = {}

    for i, article in enumerate(articles):
        article_id = f"article_{i:04d}"
        publisher = clean(article.get("source", ""))
        publishers[article_id] = publisher
        article_info[article_id] = {
            "title": clean(article.get("title", "")),
            "publisher": publisher,
            "url": clean(article.get("url", "")),
        }

    # Build patterns only for distinct publisher names.
    distinct_publishers = sorted(
        {p for p in publishers.values() if len(p) >= 4},
        key=len,
        reverse=True,
    )
    patterns = {
        name: publisher_pattern(name)
        for name in distinct_publishers
    }

    rows = []
    seen = set()
    article_candidate_counts = Counter()

    for row in queue:
        source_id = clean(row.get("article_id", ""))
        source_title = clean(row.get("title", ""))
        source_publisher = clean(row.get("publisher", ""))
        source_url = clean(row.get("url", ""))
        contexts = clean(row.get("attribution_contexts", ""))

        if not source_id or not contexts:
            continue

        # Contexts were separated by the queue builder with " || ".
        snippets = [s.strip() for s in contexts.split(" || ") if s.strip()]

        for snippet in snippets:
            for mentioned_publisher, pattern in patterns.items():
                if not pattern.search(snippet):
                    continue

                # A publisher mention is only a candidate. It may refer
                # to a different story, an unrelated entity, or boilerplate.
                if mentioned_publisher.casefold() == source_publisher.casefold():
                    continue

                target_ids = [
                    aid for aid, publisher in publishers.items()
                    if publisher.casefold() == mentioned_publisher.casefold()
                ]

                for target_id in target_ids:
                    if target_id == source_id:
                        continue

                    target = article_info[target_id]
                    key = (source_id, target_id, snippet)

                    if key in seen:
                        continue
                    seen.add(key)

                    rows.append({
                        "source_article_id": source_id,
                        "source_title": source_title,
                        "source_publisher": source_publisher,
                        "source_url": source_url,
                        "mentioned_publisher": mentioned_publisher,
                        "candidate_target_article_id": target_id,
                        "candidate_target_title": target["title"],
                        "candidate_target_url": target["url"],
                        "matched_context": snippet,
                        "candidate_basis": "publisher_name_in_attribution_context",
                        "lineage_status": "unknown",
                        "verified_relationship": "no",
                        "review_status": "needs_manual_verification",
                        "review_notes": "",
                    })
                    article_candidate_counts[source_id] += 1

    rows.sort(
        key=lambda r: (
            r["source_article_id"],
            r["mentioned_publisher"],
            r["candidate_target_article_id"],
        )
    )

    fields = list(rows[0].keys()) if rows else [
        "source_article_id",
        "source_title",
        "source_publisher",
        "source_url",
        "mentioned_publisher",
        "candidate_target_article_id",
        "candidate_target_title",
        "candidate_target_url",
        "matched_context",
        "candidate_basis",
        "lineage_status",
        "verified_relationship",
        "review_status",
        "review_notes",
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "corpus_articles": len(articles),
        "candidate_relationship_rows": len(rows),
        "source_articles_with_candidates": len(article_candidate_counts),
        "candidate_publisher_mentions": len({
            r["mentioned_publisher"] for r in rows
        }),
        "verified_relationships": 0,
        "lineage_labels_assigned": 0,
        "interpretation": (
            "These are candidate links generated from publisher names "
            "appearing in attribution-like contexts. They do not establish "
            "that a particular target article is the upstream source. "
            "All relationships remain unknown until verified."
        ),
        "outputs": [
            str(OUTPUT_CSV.relative_to(ROOT)),
            str(OUTPUT_JSON.relative_to(ROOT)),
        ],
    }

    OUTPUT_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nCandidate provenance graph created.")
    print("Corpus articles:", len(articles))
    print("Candidate relationship rows:", len(rows))
    print("Source articles with candidates:", len(article_candidate_counts))
    print("Publisher names matched:", summary["candidate_publisher_mentions"])
    print("Verified relationships: 0 (manual verification required)")
    print("CSV:", OUTPUT_CSV)
    print("Summary:", OUTPUT_JSON)


if __name__ == "__main__":
    main()