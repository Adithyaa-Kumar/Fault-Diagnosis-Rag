
import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_FILE = ROOT / "data" / "raw" / "multihoprag" / "MultiHopRAG.json"
OUTPUT_DIR = ROOT / "data" / "processed"

PAIR_OUTPUT = OUTPUT_DIR / "multihoprag_semantic_redundancy_pairs.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "multihoprag_semantic_redundancy_summary.json"

# These are screening thresholds, not scientifically validated labels.
THRESHOLDS = (0.30, 0.50, 0.70, 0.90)


def normalize_text(value):
    """Convert a field to clean text without modifying the source data."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    # Collect all non-empty evidence facts.
    evidence_rows = []
    question_counts = Counter()
    skipped_questions = 0

    for question_index, question in enumerate(questions):
        evidence_list = question.get("evidence_list") or []

        if not evidence_list:
            skipped_questions += 1
            continue

        question_id = normalize_text(
            question.get("_id", question.get("id", question_index))
        )

        for evidence_index, evidence in enumerate(evidence_list):
            if not isinstance(evidence, dict):
                continue

            fact = normalize_text(evidence.get("fact", ""))

            if not fact:
                continue

            evidence_rows.append({
                "question_id": question_id,
                "question_index": question_index,
                "evidence_index": evidence_index,
                "evidence_id": normalize_text(evidence.get("id", "")),
                "fact": fact,
                "title": normalize_text(evidence.get("title", "")),
                "publisher": normalize_text(evidence.get("source", "")),
                "url": normalize_text(evidence.get("url", "")),
            })

            question_counts[question_id] += 1

    if not evidence_rows:
        raise RuntimeError("No non-empty evidence facts were found.")

    # Fit one vocabulary across all facts, then compare facts only
    # when they belong to the same question.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(
        [row["fact"] for row in evidence_rows]
    )

    rows_by_question = {}
    for row_index, row in enumerate(evidence_rows):
        rows_by_question.setdefault(row["question_id"], []).append(row_index)

    pair_rows = []
    pair_count = 0
    questions_with_pairs = 0
    scores_by_threshold = Counter()
    qualifying_questions = {threshold: set() for threshold in THRESHOLDS}
    maximum_scores = []

    for question_id, indices in rows_by_question.items():
        if len(indices) < 2:
            continue

        questions_with_pairs += 1

        # Each question has only a few evidence items, so pairwise
        # comparison is small and computationally manageable.
        for left_index, right_index in combinations(indices, 2):
            left = evidence_rows[left_index]
            right = evidence_rows[right_index]

            score = float(
                cosine_similarity(
                    matrix[left_index],
                    matrix[right_index],
                )[0, 0]
            )

            pair_count += 1
            maximum_scores.append(score)

            record = {
                "question_id": question_id,
                "left_evidence_index": left["evidence_index"],
                "right_evidence_index": right["evidence_index"],
                "left_evidence_id": left["evidence_id"],
                "right_evidence_id": right["evidence_id"],
                "left_publisher": left["publisher"],
                "right_publisher": right["publisher"],
                "left_title": left["title"],
                "right_title": right["title"],
                "left_url": left["url"],
                "right_url": right["url"],
                "left_fact": left["fact"],
                "right_fact": right["fact"],
                "tfidf_cosine_similarity": round(score, 6),
            }

            # Record threshold flags for inspection only.
            for threshold in THRESHOLDS:
                record[f"at_least_{threshold:.2f}"] = score >= threshold

                if score >= threshold:
                    scores_by_threshold[threshold] += 1
                    qualifying_questions[threshold].add(question_id)

            pair_rows.append(record)

    # Sort most textually similar pairs to the top.
    pair_rows.sort(
        key=lambda row: row["tfidf_cosine_similarity"],
        reverse=True,
    )

    fieldnames = list(pair_rows[0].keys()) if pair_rows else [
        "question_id",
        "tfidf_cosine_similarity",
    ]

    with PAIR_OUTPUT.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pair_rows)

    summary = {
        "dataset": "MultiHop-RAG",
        "method": "TF-IDF cosine similarity over evidence fact text",
        "interpretation": (
            "Similarity is a lexical screening signal, not proof of "
            "semantic equivalence, factual duplication, or provenance lineage."
        ),
        "raw_question_count": len(questions),
        "questions_without_evidence_list": skipped_questions,
        "nonempty_evidence_fact_count": len(evidence_rows),
        "questions_with_at_least_two_nonempty_facts": questions_with_pairs,
        "within_question_pair_count": pair_count,
        "thresholds": {
            f"{threshold:.2f}": {
                "pair_count_at_or_above_threshold":
                    scores_by_threshold[threshold],
                "question_count_with_at_least_one_pair":
                    len(qualifying_questions[threshold]),
            }
            for threshold in THRESHOLDS
        },
        "maximum_pair_similarity": (
            round(max(maximum_scores), 6) if maximum_scores else None
        ),
        "mean_pair_similarity": (
            round(sum(maximum_scores) / len(maximum_scores), 6)
            if maximum_scores else None
        ),
        "outputs": [
            str(PAIR_OUTPUT.relative_to(ROOT)),
            str(SUMMARY_OUTPUT.relative_to(ROOT)),
        ],
        "experimental_conditions_assigned": False,
        "provenance_labels_assigned": False,
    }

    with SUMMARY_OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\nSemantic redundancy screening completed.")
    print(f"Raw questions: {len(questions)}")
    print(f"Evidence facts compared: {len(evidence_rows)}")
    print(f"Questions with comparable pairs: {questions_with_pairs}")
    print(f"Within-question pairs: {pair_count}")
    print("\nThreshold summary:")

    for threshold in THRESHOLDS:
        print(
            f"  Similarity >= {threshold:.2f}: "
            f"{scores_by_threshold[threshold]} pairs across "
            f"{len(qualifying_questions[threshold])} questions"
        )

    print(f"\nPair results: {PAIR_OUTPUT}")
    print(f"Summary:      {SUMMARY_OUTPUT}")
    print("\nNo provenance labels or experimental conditions were assigned.")


if __name__ == "__main__":
    main()
