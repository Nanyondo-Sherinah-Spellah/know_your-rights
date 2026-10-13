
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

DB_PATH = Path("data/chroma_db")
COLLECTION_NAME = "uganda_constitution"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5


# ============================================================
# 2. EVALUATION DATASET
# ============================================================

TEST_CASES = [
    {
        "question": "Who holds sovereignty in Uganda?",
        "expected_articles": [1],
    },
    {
        "question": "What is the supreme law of Uganda?",
        "expected_articles": [2],
    },
    {
        "question": "Can the police detain me without telling me why?",
        "expected_articles": [23],
    },
    {
        "question": "What does the Constitution say about a fair trial?",
        "expected_articles": [28],
    },
    {
        "question": "What does the Constitution say about the right to education?",
        "expected_articles": [30],
    },
    {
        "question": "What does the Constitution say about access to information?",
        "expected_articles": [41],
    },
    {
        "question": "Can someone invade my privacy at home?",
        "expected_articles": [27],
    },
    {
        "question": "Who is responsible for a child's basic education?",
        "expected_articles": [34],
    },
]


# ============================================================
# 3. CONNECT TO THE VECTOR DATABASE
# ============================================================

def load_collection():
    """Connect to the existing constitutional collection."""

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH.resolve()}\n"
            "Run build_vector_db.py first."
        )

    client = chromadb.PersistentClient(path=str(DB_PATH))

    collection = client.get_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        raise RuntimeError("The constitutional collection is empty.")

    return collection


# ============================================================
# 4. RETRIEVE RESULTS FOR A QUESTION
# ============================================================

def retrieve_results(
    question: str,
    embedding_model: SentenceTransformer,
    collection,
) -> list[dict]:
    """Return the top matching chunks and their metadata."""

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K, collection.count()),
        include=["metadatas", "distances"],
    )

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []

    for rank, (metadata, distance) in enumerate(
        zip(metadatas, distances),
        start=1,
    ):
        metadata = metadata or {}

        article_number = metadata.get("article_number")

        if article_number is not None:
            try:
                article_number = int(article_number)
            except (TypeError, ValueError):
                article_number = None

        retrieved.append(
            {
                "rank": rank,
                "article_number": article_number,
                "title": metadata.get("title") or "Untitled section",
                "distance": float(distance),
            }
        )

    return retrieved


# ============================================================
# 5. EVALUATE RETRIEVAL QUALITY
# ============================================================

def evaluate_case(
    test_case: dict,
    embedding_model: SentenceTransformer,
    collection,
) -> dict:
    """Check whether an expected article appears in the top results."""

    question = test_case["question"]
    expected_articles = test_case["expected_articles"]

    retrieved = retrieve_results(
        question=question,
        embedding_model=embedding_model,
        collection=collection,
    )

    matching_ranks = [
        result["rank"]
        for result in retrieved
        if result["article_number"] in expected_articles
    ]

    first_match_rank = (
        min(matching_ranks) if matching_ranks else None
    )

    hit_at_1 = first_match_rank is not None and first_match_rank <= 1
    hit_at_3 = first_match_rank is not None and first_match_rank <= 3
    hit_at_5 = first_match_rank is not None and first_match_rank <= 5

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print(f"EXPECTED ARTICLE(S): {expected_articles}")
    print("-" * 70)

    for result in retrieved:
        article = result["article_number"]
        title = result["title"]
        distance = result["distance"]

        if article is not None:
            reference = f"Article {article}: {title}"
        else:
            reference = title

        marker = (
            "  <-- EXPECTED ARTICLE"
            if article in expected_articles
            else ""
        )

        print(
            f"Rank {result['rank']}: {reference}"
            f" | Distance: {distance:.4f}{marker}"
        )

    if first_match_rank is None:
        print("RESULT: FAIL — expected article not found in top 5.")
    else:
        print(f"RESULT: PASS — expected article found at rank {first_match_rank}.")

    return {
        "question": question,
        "expected_articles": expected_articles,
        "first_match_rank": first_match_rank,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5,
    }


# ============================================================
# 6. SUMMARIZE THE EVALUATION
# ============================================================

def print_summary(results: list[dict]) -> None:
    """Calculate and print Hit@1, Hit@3 and Hit@5."""

    total = len(results)

    if total == 0:
        print("No test cases were evaluated.")
        return

    hit_1 = sum(result["hit_at_1"] for result in results)
    hit_3 = sum(result["hit_at_3"] for result in results)
    hit_5 = sum(result["hit_at_5"] for result in results)

    print("\n\n" + "=" * 70)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Total test questions: {total}")
    print(f"Hit@1: {hit_1}/{total} ({hit_1 / total:.1%})")
    print(f"Hit@3: {hit_3}/{total} ({hit_3 / total:.1%})")
    print(f"Hit@5: {hit_5}/{total} ({hit_5 / total:.1%})")

    print("\nINTERPRETATION")
    print("Hit@1 measures whether the correct article ranks first.")
    print("Hit@3 measures whether it appears in the top three chunks.")
    print("Hit@5 measures whether it appears in the top five chunks.")

    print(
        "\nThese scores measure retrieval against this small, "
        "manually labelled test set. They do not prove that the "
        "generated answers or legal interpretations are correct."
    )


# ============================================================
# 7. RUN THE EVALUATION
# ============================================================

def main() -> None:
    print("=" * 70)
    print("KNOW YOUR RIGHTS AI — RETRIEVAL EVALUATION")
    print("=" * 70)

    print("\nLoading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Connecting to ChromaDB...")
    collection = load_collection()

    print(f"Collection contains {collection.count()} chunks.")
    print(f"Evaluating {len(TEST_CASES)} questions...")

    results = []

    for test_case in TEST_CASES:
        result = evaluate_case(
            test_case=test_case,
            embedding_model=embedding_model,
            collection=collection,
        )
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()
