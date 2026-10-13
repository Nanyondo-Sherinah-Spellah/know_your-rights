
import json
from pathlib import Path

files = [
    Path("data/processed/articles.json"),
    Path("data/processed/other_sections.json"),
]

for file_path in files:
    data = json.loads(file_path.read_text(encoding="utf-8"))

    print(f"\nFILE: {file_path}")
    print(f"Records: {len(data)}")
    print("First record:")

    print(json.dumps(data[0], ensure_ascii=False, indent=2)[:1800])
