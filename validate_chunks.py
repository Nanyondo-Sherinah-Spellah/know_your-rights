
import json
from collections import defaultdict
from pathlib import Path

from chunk_corpus import CHUNK_SIZE, prepare_units

CORPUS_PATH = Path("data/processed/corpus.json")
CHUNKS_PATH = Path("data/processed/chunks.json")


def normalize(text: str) -> str:
    """Normalize whitespace for text comparisons."""
    return " ".join(text.split())


def main() -> None:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    errors = []

    source_by_id = {
        record["id"]: record
        for record in corpus
    }

    chunks_by_source = defaultdict(list)

    # Check IDs, chunk sizes, and empty content.
    chunk_ids = [chunk.get("id") for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        errors.append("Duplicate chunk IDs found.")

    for chunk in chunks:
        chunk_id = chunk.get("id", "<missing ID>")
        source_id = chunk.get("source_record_id")

        if not source_id:
            errors.append(f"{chunk_id}: missing source_record_id.")
            continue

        if source_id not in source_by_id:
            errors.append(
                f"{chunk_id}: refers to unknown source {source_id}."
            )
            continue

        chunks_by_source[source_id].append(chunk)

        text = chunk.get("text", "")

        if not text.strip():
            errors.append(f"{chunk_id}: empty text.")

        if len(text) > CHUNK_SIZE:
            errors.append(
                f"{chunk_id}: {len(text)} characters "
                f"(limit {CHUNK_SIZE})."
            )

        if not chunk.get("id"):
            errors.append("A chunk has no ID.")

    # Check record coverage, chunk numbering, metadata, and text order.
    missing_records = []
    checked_units = 0
    missing_units = []

    for record in corpus:
        source_id = record["id"]
        record_chunks = sorted(
            chunks_by_source.get(source_id, []),
            key=lambda item: item.get("chunk_index", -1),
        )

        if not record_chunks:
            missing_records.append(source_id)
            continue

        expected_indices = list(range(len(record_chunks)))
        actual_indices = [
            chunk.get("chunk_index")
            for chunk in record_chunks
        ]

        if actual_indices != expected_indices:
            errors.append(
                f"{source_id}: chunk indices are missing, duplicated, "
                "or out of order."
            )

        for chunk in record_chunks:
            if chunk.get("chunk_count") != len(record_chunks):
                errors.append(
                    f"{chunk['id']}: incorrect chunk_count metadata."
                )

        source_units = prepare_units(record.get("text", ""))
        combined_chunks = " ".join(
            normalize(chunk.get("text", ""))
            for chunk in record_chunks
        )

        # Confirm each prepared source unit appears in order.
        # Chunk overlap may duplicate text, which is expected.
        search_from = 0

        for unit_number, unit in enumerate(source_units, start=1):
            normalized_unit = normalize(unit)
            position = combined_chunks.find(
                normalized_unit,
                search_from,
            )

            checked_units += 1

            if position == -1:
                missing_units.append(
                    f"{source_id}: prepared unit {unit_number} "
                    f"not found in order: {normalized_unit[:100]!r}"
                )
                continue

            search_from = position + len(normalized_unit)

    if missing_records:
        errors.append(
            f"{len(missing_records)} source records have no chunks: "
            f"{missing_records[:10]}"
        )

    print("CHUNK VALIDATION REPORT")
    print("=" * 40)
    print(f"Source records: {len(corpus)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Records with chunks: {len(chunks_by_source)}")
    print(f"Unique chunk IDs: {len(set(chunk_ids))}")
    print(f"Prepared text units checked: {checked_units}")
    print(f"Missing or out-of-order units: {len(missing_units)}")
    print(f"Validation errors: {len(errors)}")

    if missing_units:
        print("\nText coverage issues (first 10):")
        for issue in missing_units[:10]:
            print("-", issue)

    if errors:
        print("\nValidation errors (first 20):")
        for error in errors[:20]:
            print("-", error)

    if errors or missing_units:
        print("\nSTRUCTURAL/TEXT COVERAGE CHECK: FAIL")
    else:
        print("\nSTRUCTURAL/TEXT COVERAGE CHECK: PASS")
        print(
            "All prepared source units were found in order within "
            "their respective chunk sequences."
        )
        print(
            "Note: this checks consistency with the extracted corpus, "
            "not accuracy against the official Constitution."
        )


if __name__ == "__main__":
    main()
