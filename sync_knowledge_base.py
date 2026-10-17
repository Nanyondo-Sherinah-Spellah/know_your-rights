
import argparse
import json
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("data/processed/chunks.json")
DB_PATH = Path("data/chroma_db")
COLLECTION_NAME = "uganda_constitution"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


def load_chunks(source_document: str) -> list[dict[str, Any]]:
    """Load and validate chunks for one explicitly selected source."""

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunk file not found: {CHUNKS_PATH}")

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        all_chunks = json.load(file)

    if not isinstance(all_chunks, list) or not all_chunks:
        raise ValueError("chunks.json must contain a non-empty list.")

    chunks = [
        chunk
        for chunk in all_chunks
        if chunk.get("source_document") == source_document
    ]

    if not chunks:
        raise ValueError(
            f"No chunks found for source {source_document!r}. "
            "Nothing was changed."
        )

    seen_ids: set[str] = set()

    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Every chunk must be a JSON object.")

        chunk_id = chunk.get("id")
        text = chunk.get("text")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("A chunk has a missing or invalid ID.")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk {chunk_id!r} has empty text.")

        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk ID: {chunk_id}")

        if not isinstance(chunk.get("source_record_id"), str):
            raise ValueError(
                f"Chunk {chunk_id!r} is missing source_record_id."
            )

        seen_ids.add(chunk_id)

    return chunks


def safe_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Keep only metadata types supported by ChromaDB."""

    return {
        key: value
        for key, value in chunk.items()
        if key not in {"id", "text"}
        and value is not None
        and isinstance(value, (str, int, float, bool))
    }


def sync_source(source_document: str, apply_changes: bool) -> None:
    """Synchronize one source without touching other sources."""

    chunks = load_chunks(source_document)

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)

    # Read existing records belonging only to this source.
    existing = collection.get(
        where={"source_document": source_document},
        include=["documents", "metadatas"],
    )

    existing_ids = set(existing["ids"])
    incoming_by_id = {chunk["id"]: chunk for chunk in chunks}
    incoming_ids = set(incoming_by_id)

    ids_to_add = sorted(incoming_ids - existing_ids)
    ids_to_delete = sorted(existing_ids - incoming_ids)

    existing_by_id = {
        chunk_id: (document, metadata or {})
        for chunk_id, document, metadata in zip(
            existing["ids"],
            existing["documents"],
            existing["metadatas"],
        )
    }

    ids_to_update = []

    for chunk_id in sorted(incoming_ids & existing_ids):
        chunk = incoming_by_id[chunk_id]
        old_document, old_metadata = existing_by_id[chunk_id]

        if (
            old_document != chunk["text"]
            or old_metadata != safe_metadata(chunk)
        ):
            ids_to_update.append(chunk_id)

    print(f"\nSource: {source_document}")
    print(f"Incoming chunks: {len(chunks)}")
    print(f"Existing source chunks: {len(existing_ids)}")
    print(f"New chunks: {len(ids_to_add)}")
    print(f"Changed chunks: {len(ids_to_update)}")
    print(f"Obsolete chunks: {len(ids_to_delete)}")

    if not apply_changes:
        print("\nDRY RUN: no database changes were made.")
        print("Review the counts before applying synchronization.")
        return

    if ids_to_add or ids_to_update:
        print("\nLoading embedding model...")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        changed_ids = ids_to_add + ids_to_update

        for start in range(0, len(changed_ids), BATCH_SIZE):
            batch_ids = changed_ids[start : start + BATCH_SIZE]
            batch = [incoming_by_id[item] for item in batch_ids]

            documents = [item["text"] for item in batch]
            metadata = [safe_metadata(item) for item in batch]

            embeddings = model.encode(
                documents,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            collection.upsert(
                ids=batch_ids,
                documents=documents,
                metadatas=metadata,
                embeddings=embeddings.tolist(),
            )

            print(
                f"Processed {min(start + len(batch), len(changed_ids))}"
                f"/{len(changed_ids)} changed chunks."
            )

    # Delete only obsolete IDs from this explicitly selected source.
    for start in range(0, len(ids_to_delete), BATCH_SIZE):
        batch_ids = ids_to_delete[start : start + BATCH_SIZE]
        collection.delete(ids=batch_ids)

    print("\nSynchronization complete.")
    print("Collection:", collection.name)
    print("Total records now:", collection.count())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely synchronize one source into ChromaDB."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Exact source_document value to synchronize.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, only preview changes.",
    )
    args = parser.parse_args()

    sync_source(args.source, args.apply)


if __name__ == "__main__":
    main()
