
import json
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "multihoprag"

BASE_URL = (
    "https://huggingface.co/datasets/yixuantt/"
    "MultiHopRAG/resolve/main/"
)

FILES = ["MultiHopRAG.json", "corpus.json"]


def download_file(filename):
    destination = DATA_DIR / filename
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"Already downloaded: {destination}")
        return destination

    url = BASE_URL + filename
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MultiHopRAG-Research-Audit/1.0"},
    )

    print(f"Downloading {filename}...")
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()

    if not data:
        raise ValueError(f"Downloaded file is empty: {filename}")

    destination.write_bytes(data)
    print(f"Saved {destination} ({len(data):,} bytes)")
    return destination


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def records(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "records", "documents", "corpus"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    raise TypeError(f"Unexpected JSON root type: {type(data).__name__}")


def inspect_file(filename):
    path = DATA_DIR / filename
    data = load_json(path)
    rows = records(data)

    print(f"\n--- {filename} ---")
    print("Root type:", type(data).__name__)
    print("Record count:", len(rows))

    if not rows:
        print("No records found.")
        return

    first = rows[0]
    if not isinstance(first, dict):
        print("First record type:", type(first).__name__)
        print("First record:", repr(first)[:1000])
        return

    print("First-record fields:", list(first.keys()))
    print("First record preview:")
    print(json.dumps(first, ensure_ascii=False, indent=2)[:2500])

    # Summarize source metadata when present.
    for field in ("source", "publisher", "category"):
        values = [
            row.get(field)
            for row in rows
            if isinstance(row, dict) and row.get(field)
        ]
        if values:
            counts = Counter(values)
            print(f"\n{field}: {len(counts)} distinct values")
            print("Most common:", counts.most_common(10))


def main():
    for filename in FILES:
        download_file(filename)

    for filename in FILES:
        inspect_file(filename)

    print("\nInspection complete.")
    print("Raw HotpotQA data has not been changed.")


if __name__ == "__main__":
    main()
