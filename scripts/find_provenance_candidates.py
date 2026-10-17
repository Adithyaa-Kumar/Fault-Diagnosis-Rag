
import csv
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "multihoprag"
OUT_DIR = ROOT / "data" / "processed"

QUESTIONS_FILE = DATA_DIR / "MultiHopRAG.json"
CORPUS_FILE = DATA_DIR / "corpus.json"
OUTPUT_FILE = OUT_DIR / "multihoprag_provenance_candidates.csv"

ATTRIBUTION_PATTERNS = [
    r"\baccording to\b",
    r"\breported by\b",
    r"\bfirst reported by\b",
    r"\boriginally published by\b",
    r"\bpublished by\b",
    r"\bvia Reuters\b",
    r"\bReuters reported\b",
    r"\bAssociated Press\b",
    r"\bAP News\b",
    r"\bsyndicat(?:ed|ion)\b",
    r"\bpress release\b",
    r"\bsaid in a statement\b",
]

def normalize_url(url):
    if not url:
        return ""
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    path = re.sub(r"/+$", "", parts.path) or "/"
    return f"{host}{path}".lower()

def normalize_text(text):
    text = (text or "").lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())

def attribution_hits(body):
    hits = []
    for pattern in ATTRIBUTION_PATTERNS:
        match = re.search(pattern, body or "", flags=re.IGNORECASE)
        if match:
            start = max(0, match.start() - 100)
            end = min(len(body), match.end() + 180)
            hits.append(body[start:end].replace("\n", " "))
    return hits[:4]

def main():
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    corpus = json.loads(CORPUS_FILE.read_text(encoding="utf-8"))

    corpus_by_url = {
        normalize_url(article.get("url", "")): article
        for article in corpus
        if article.get("url")
    }

    candidates = []
    seen_pairs = set()

    for question_index, question in enumerate(questions):
        evidence = question.get("evidence_list", [])

        for left, right in combinations(evidence, 2):
            left_url = left.get("url", "")
            right_url = right.get("url", "")

            left_key = normalize_url(left_url)
            right_key = normalize_url(right_url)

            if not left_key or not right_key or left_key == right_key:
                continue

            left_source = left.get("source", "")
            right_source = right.get("source", "")

            # Focus on cross-publisher pairs for this initial screen.
            if left_source and right_source and left_source == right_source:
                continue

            pair_key = tuple(sorted((left_key, right_key)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            left_article = corpus_by_url.get(left_key, {})
            right_article = corpus_by_url.get(right_key, {})

            left_body = left_article.get("body", "")
            right_body = right_article.get("body", "")

            left_fact = left.get("fact", "")
            right_fact = right.get("fact", "")

            left_norm = normalize_text(left_fact)
            right_norm = normalize_text(right_fact)

            similarity = (
                SequenceMatcher(None, left_norm, right_norm).ratio()
                if left_norm and right_norm else 0.0
            )

            left_hits = attribution_hits(left_body)
            right_hits = attribution_hits(right_body)

            if similarity < 0.75 and not left_hits and not right_hits:
                continue

            if similarity >= 0.90:
                screening_flag = "high_fact_text_similarity"
            elif similarity >= 0.75:
                screening_flag = "possible_fact_text_similarity"
            else:
                screening_flag = "attribution_language_found"

            candidates.append({
                "question_index": question_index,
                "query": question.get("query", ""),
                "answer": question.get("answer", ""),
                "left_title": left.get("title", ""),
                "left_publisher": left_source,
                "left_url": left_url,
                "left_fact": left_fact,
                "left_attribution_snippets": " || ".join(left_hits),
                "right_title": right.get("title", ""),
                "right_publisher": right_source,
                "right_url": right_url,
                "right_fact": right_fact,
                "right_attribution_snippets": " || ".join(right_hits),
                "fact_text_similarity": round(similarity, 3),
                "screening_flag": screening_flag,
                "lineage_status": "unknown",
                "review_notes": "",
            })

    candidates.sort(
        key=lambda row: (
            row["fact_text_similarity"],
            bool(row["left_attribution_snippets"])
            + bool(row["right_attribution_snippets"]),
        ),
        reverse=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "question_index", "query", "answer",
        "left_title", "left_publisher", "left_url",
        "left_fact", "left_attribution_snippets",
        "right_title", "right_publisher", "right_url",
        "right_fact", "right_attribution_snippets",
        "fact_text_similarity", "screening_flag",
        "lineage_status", "review_notes",
    ]

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(candidates)

    print("=== PROVENANCE CANDIDATE SCREEN ===")
    print("Questions inspected:", len(questions))
    print("Corpus articles:", len(corpus))
    print("Candidate pairs:", len(candidates))
    print(
        "High fact-text similarity (>= 0.90):",
        sum(c["fact_text_similarity"] >= 0.90 for c in candidates),
    )
    print(
        "Pairs with attribution snippets:",
        sum(
            bool(c["left_attribution_snippets"])
            or bool(c["right_attribution_snippets"])
            for c in candidates
        ),
    )
    print("Output:", OUTPUT_FILE)
    print()
    print("All candidate lineage labels remain 'unknown'.")
    print("Text similarity and attribution phrases require manual verification.")

if __name__ == "__main__":
    main()
