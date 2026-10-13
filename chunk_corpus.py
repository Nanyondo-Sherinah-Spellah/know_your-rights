
import json
import re
from pathlib import Path

INPUT_PATH = Path("data/processed/corpus.json")
OUTPUT_PATH = Path("data/processed/chunks.json")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Recognise standalone constitutional clause labels such as (1) and (a).
CLAUSE_LABEL = re.compile(r"^\s*(?:\(\d+\)|\([a-z]\))\s*$", re.IGNORECASE)


def clean_source_lines(text: str) -> list[str]:
    """Remove blank lines and formatting-only underscore lines."""
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not re.fullmatch(r"_+", line.strip())
    ]


def attach_clause_labels(lines: list[str]) -> list[str]:
    """Attach a standalone clause label to the following line."""
    result = []
    index = 0

    while index < len(lines):
        line = lines[index]

        if CLAUSE_LABEL.fullmatch(line) and index + 1 < len(lines):
            result.append(f"{line} {lines[index + 1]}")
            index += 2
        else:
            result.append(line)
            index += 1

    return result


def split_long_line(line: str) -> list[str]:
    """Split long lines at word boundaries."""
    words = line.split()
    parts = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip() if current else word

        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            if current:
                parts.append(current)

            # Keep an unusually long single token intact rather than
            # splitting it in the middle of a word.
            if len(word) > CHUNK_SIZE:
                parts.append(word)
                current = ""
            else:
                current = word

    if current:
        parts.append(current)

    return parts


def prepare_units(text: str) -> list[str]:
    """Prepare clean text units for chunking."""
    lines = clean_source_lines(text)
    lines = attach_clause_labels(lines)

    units = []
    for line in lines:
        units.extend(split_long_line(line))

    return units


def choose_overlap(
    previous_units: list[str],
    next_unit: str,
) -> list[str]:
    """Reuse whole previous units without exceeding the chunk limit."""
    selected = []
    selected_length = 0

    for unit in reversed(previous_units):
        proposed_length = (
            selected_length + len(unit) + (1 if selected else 0)
        )

        if proposed_length > CHUNK_OVERLAP:
            break

        # Leave enough room for the next unit and its separating newline.
        next_length = len(next_unit) + 1 + proposed_length

        if next_length > CHUNK_SIZE:
            break

        selected.insert(0, unit)
        selected_length = proposed_length

    return selected


def make_chunks(text: str) -> list[str]:
    """Pack text units into chunks with safe, whole-unit overlap."""
    units = prepare_units(text)

    if not units:
        return []

    chunks = []
    current_units = []

    for unit in units:
        candidate_units = current_units + [unit]
        candidate_text = "\n".join(candidate_units)

        if len(candidate_text) <= CHUNK_SIZE:
            current_units.append(unit)
            continue

        if current_units:
            chunks.append("\n".join(current_units))

        overlap_units = choose_overlap(current_units, unit)
        current_units = overlap_units + [unit]

        if len("\n".join(current_units)) > CHUNK_SIZE:
            current_units = [unit]

    if current_units:
        chunks.append("\n".join(current_units))

    return chunks


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {INPUT_PATH.resolve()}"
        )

    corpus = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    all_chunks = []

    for record in corpus:
        source_text = record.get("text", "").strip()

        if not source_text:
            continue

        pieces = make_chunks(source_text)

        for index, piece in enumerate(pieces):
            chunk = {
                **record,
                "id": f"{record['id']}_chunk_{index + 1}",
                "source_record_id": record["id"],
                "chunk_index": index,
                "chunk_count": len(pieces),
                "text": piece,
            }
            all_chunks.append(chunk)

    lengths = [len(chunk["text"]) for chunk in all_chunks]
    unique_ids = len({chunk["id"] for chunk in all_chunks})
    empty_count = sum(not chunk["text"].strip() for chunk in all_chunks)
    oversized_count = sum(
        len(chunk["text"]) > CHUNK_SIZE for chunk in all_chunks
    )

    if unique_ids != len(all_chunks):
        raise ValueError("Duplicate chunk IDs detected.")

    if empty_count:
        raise ValueError("Empty chunks detected.")

    if oversized_count:
        raise ValueError(
            "Some chunks exceed CHUNK_SIZE. Check unusually long tokens."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Chunking complete.")
    print(f"Source records: {len(corpus)}")
    print(f"Chunks created: {len(all_chunks)}")
    print(f"Longest chunk: {max(lengths, default=0)} characters")
    print(f"Shortest chunk: {min(lengths, default=0)} characters")
    print(f"Empty chunks: {empty_count}")
    print(f"Oversized chunks: {oversized_count}")
    print(f"Unique IDs: {unique_ids}")
    print(f"Saved to: {OUTPUT_PATH.resolve()}")

    print("\nBoundary samples:")

    for source_id in [
        "constitution_article_23",
        "constitution_article_28",
        "constitution_second_schedule",
    ]:
        matching = [
            chunk
            for chunk in all_chunks
            if chunk["source_record_id"] == source_id
        ]

        print(f"\n{source_id}: {len(matching)} chunks")

        for chunk in matching[:2]:
            print(f"\nChunk {chunk['chunk_index'] + 1}")
            print("START:", repr(chunk["text"][:160]))
            print("END:  ", repr(chunk["text"][-160:]))


if __name__ == "__main__":
    main()
