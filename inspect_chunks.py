
import json
from pathlib import Path

chunks_path = Path("data/processed/chunks.json")

with chunks_path.open("r", encoding="utf-8") as file:
    chunks = json.load(file)

print("Data type:", type(chunks).__name__)

if isinstance(chunks, list):
    print("Number of chunks:", len(chunks))

    if chunks:
        print("\nFirst chunk's field names:")
        print(list(chunks[0].keys()))

        print("\nFirst chunk:")
        print(json.dumps(chunks[0], indent=2, ensure_ascii=False))

elif isinstance(chunks, dict):
    print("Top-level keys:", list(chunks.keys()))
    print("\nFirst part of the data:")
    print(json.dumps(chunks, indent=2, ensure_ascii=False)[:3000])
