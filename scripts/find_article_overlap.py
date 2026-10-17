
import csv
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_FILE = ROOT / "data" / "raw" / "multihoprag" / "corpus.json"
OUTPUT_FILE = (
    ROOT / "data" / "processed" / "multihoprag_article_overlap.csv"
)

SHINGLE_SIZE = 8
MAX_DOCUMENT_FREQUENCY = 20
MIN_SHARED_SHINGLES = 3


def tokenize(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def shingles(text):
    tokens = tokenize(text)
    if len(tokens) < SHINGLE_SIZE:
        return set()
    return {
        tuple(tokens[i:i + SHINGLE_SIZE])
        for i in range(len(tokens) - SHINGLE_SIZE + 1)
    }


def main():
    articles = json.loads(CORPUS_FILE.read_text(encoding="utf-8"))

    docs = []
    for index, article in enumerate(articles):
        docs.append({
            "index": index,
            "title": article.get("title", ""),
            "source": article.get("source", ""),
            "url": article.get("url", ""),
            "shingles": shingles(article.get("body", "")),
        })

    # Ignore very common phrases that occur in many articles.
    document_frequency = Counter()
    for doc in docs:
        document_frequency.update(doc["shingles"])

    filtered = []
    for doc in docs:
        filtered.append({
            **doc,
            "shingles": {
                shingle for shingle in doc["shingles"]
                if document_frequency[shingle] <= MAX_DOCUMENT_FREQUENCY
            },
        })

    postings = defaultdict(list)
    for i, doc in enumerate(filtered):
        for shingle in doc["shingles"]:
            postings[shingle].append(i)

    shared_counts = Counter()
    for indices in postings.values():
        if len(indices) < 2:
            continue
        for left, right in combinations(indices, 2):
            shared_counts[(left, right)] += 1

    candidates = []

    for (left, right), shared in shared_counts.items():
        if shared < MIN_SHARED_SHINGLES:
            continue

        a = filtered[left]
        b = filtered[right]
        size_a = len(a["shingles"])
        size_b = len(b["shingles"])
        union_size = size_a + size_b - shared

        jaccard = shared / union_size if union_size else 0
        containment = shared / min(size_a, size_b) if min(size_a, size_b) else 0

        candidates.append({
            "left_title": a["title"],
            "left_publisher": a["source"],
            "left_url": a["url"],
            "right_title": b["title"],
            "right_publisher": b["source"],
            "right_url": b["url"],
            "shared_8_word_shingles": shared,
            "jaccard_similarity": round(jaccard, 5),
            "shorter_article_containment": round(containment, 5),
            "lineage_status": "unknown",
            "review_notes": "",
        })

    candidates.sort(
        key=lambda row: (
            row["shorter_article_containment"],
            row["jaccard_similarity"],
            row["shared_8_word_shingles"],
        ),
        reverse=True,
    )

    columns = [
        "left_title", "left_publisher", "left_url",
        "right_title", "right_publisher", "right_url",
        "shared_8_word_shingles", "jaccard_similarity",
        "shorter_article_containment", "lineage_status",
        "review_notes",
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(candidates)

    print("=== FULL-CORPUS TEXT-OVERLAP AUDIT ===")
    print("Articles:", len(articles))
    print("Article pairs with at least", MIN_SHARED_SHINGLES,
          "shared distinctive shingles:", len(candidates))
    print("Output:", OUTPUT_FILE)
    print()
    print("Overlap is a screening signal, not proof of common lineage.")
    print("All lineage labels remain unknown.")


if __name__ == "__main__":
    main()
