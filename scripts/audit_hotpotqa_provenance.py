
import csv
import json
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "hotpot_dev_distractor_v1.json"
OUTPUT_FILE = (
    ROOT / "data" / "processed" / "hotpotqa_provenance_audit.csv"
)

def valid_example(example):
    contexts = {
        title: sentences
        for title, sentences in example.get("context", [])
    }
    facts = example.get("supporting_facts", [])

    if not facts:
        return False

    valid_titles = set()

    for fact in facts:
        if len(fact) != 2:
            return False

        title, sentence_index = fact

        if title not in contexts:
            return False

        if not isinstance(sentence_index, int):
            return False

        if not 0 <= sentence_index < len(contexts[title]):
            return False

        valid_titles.add(title)

    return len(valid_titles) >= 2

def wikipedia_url(title):
    slug = quote(title.replace(" ", "_"), safe="()_,:'-")
    return f"https://en.wikipedia.org/wiki/{slug}"

def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}"
        )

    with RAW_FILE.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    selected = [
        example
        for example in dataset
        if valid_example(example)
    ][:10]

    if len(selected) < 10:
        raise ValueError(
            f"Only {len(selected)} valid examples found; "
            "expected at least 10."
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "question_id",
        "question",
        "gold_answer",
        "supporting_title",
        "supporting_sentence",
        "wikipedia_page_url",
        "other_supporting_titles",
        "lineage_status",
        "lineage_evidence_url",
        "review_notes",
    ]

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for example in selected:
            contexts = {
                title: sentences
                for title, sentences in example["context"]
            }

            support_titles = list(dict.fromkeys(
                title
                for title, _ in example["supporting_facts"]
            ))

            for title, sentence_index in example["supporting_facts"]:
                writer.writerow({
                    "question_id": example["_id"],
                    "question": example["question"],
                    "gold_answer": example["answer"],
                    "supporting_title": title,
                    "supporting_sentence": (
                        contexts[title][sentence_index]
                    ),
                    "wikipedia_page_url": wikipedia_url(title),
                    "other_supporting_titles": " | ".join(
                        other
                        for other in support_titles
                        if other != title
                    ),
                    "lineage_status": "unknown",
                    "lineage_evidence_url": "",
                    "review_notes": "",
                })

    print(f"Dataset examples: {len(dataset)}")
    print(f"Audit questions: {len(selected)}")
    print(f"Audit rows: {sum(len(e['supporting_facts']) for e in selected)}")
    print(f"Created: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
