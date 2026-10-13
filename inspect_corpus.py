
import json
from pathlib import Path

corpus_path = Path("data/processed/corpus.json")
corpus = json.loads(corpus_path.read_text(encoding="utf-8"))

lengths = [
    (len(record["text"]), record["id"], record["title"])
    for record in corpus
]

lengths.sort(reverse=True)

print(f"Total records: {len(corpus)}")
print(f"Shortest record: {min(lengths)[0]} characters")
print(f"Longest record: {max(lengths)[0]} characters")
print(
    "Records longer than 3,000 characters:",
    sum(length > 3000 for length, _, _ in lengths),
)

print("\n10 longest records:")
for length, record_id, title in lengths[:10]:
    print(f"{length:>6} characters | {record_id} | {title}")
