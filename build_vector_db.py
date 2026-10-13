
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# 1. Define project paths.
CHUNKS_PATH = Path("data/processed/chunks.json")
DB_PATH = Path("data/chroma_db")

# 2. Load the validated constitutional chunks.
with CHUNKS_PATH.open("r", encoding="utf-8") as file:
    chunks = json.load(file)

if not isinstance(chunks, list) or not chunks:
    raise ValueError("chunks.json must contain a non-empty list.")

# 3. Validate the essential fields before processing.
for chunk in chunks:
    if not chunk.get("id") or not chunk.get("text", "").strip():
        raise ValueError(f"Chunk has a missing ID or empty text: {chunk}")

chunk_ids = [chunk["id"] for chunk in chunks]
if len(chunk_ids) != len(set(chunk_ids)):
    raise ValueError("Duplicate chunk IDs found.")

print(f"Loaded {len(chunks)} constitutional chunks.")

# 4. Load the embedding model.
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 5. Create a persistent ChromaDB database.
client = chromadb.PersistentClient(path=str(DB_PATH))

collection = client.get_or_create_collection(
    name="uganda_constitution",
    metadata={"hnsw:space": "cosine"},
)

# 6. Embed and store chunks in batches.
batch_size = 32

for start in range(0, len(chunks), batch_size):
    batch = chunks[start : start + batch_size]

    documents = [chunk["text"] for chunk in batch]
    ids = [chunk["id"] for chunk in batch]

    # ChromaDB metadata must contain supported scalar values.
    metadatas = []
    for chunk in batch:
        metadata = {
            key: value
            for key, value in chunk.items()
            if key not in {"id", "text"}
            and value is not None
            and isinstance(value, (str, int, float, bool))
        }
        metadatas.append(metadata)

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Upsert makes rerunning the script safe for the same chunk IDs.
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )

    completed = min(start + len(batch), len(chunks))
    print(f"Processed {completed}/{len(chunks)} chunks.")

# 7. Verify the stored records.
print("\nVECTOR DATABASE BUILD COMPLETE")
print("Collection:", collection.name)
print("Stored chunks:", collection.count())
print("Database location:", DB_PATH.resolve())

sample = collection.get(
    ids=[chunks[0]["id"]],
    include=["documents", "metadatas"],
)

if sample["ids"]:
    print("\nSample stored chunk:")
    print("ID:", sample["ids"][0])
    print("Title:", sample["metadatas"][0].get("title"))
    print("Text preview:", sample["documents"][0][:200])
