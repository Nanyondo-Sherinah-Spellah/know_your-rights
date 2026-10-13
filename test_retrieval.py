
import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "data/chroma_db"
COLLECTION_NAME = "uganda_constitution"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

test_cases = [
    {
        "question": "Can someone be detained without a lawful reason?",
        "expected_article": 23,
    },
    {
        "question": "What does the Constitution say about a fair trial?",
        "expected_article": 28,
    },
    {
        "question": "Who holds sovereignty in Uganda?",
        "expected_article": 1,
    },
    {
        "question": "What is the supreme law of Uganda?",
        "expected_article": 2,
    },
]

for case in test_cases:
    question = case["question"]
    expected_article = case["expected_article"]

    query_vector = model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=5,
        include=["metadatas", "distances"],
    )

    articles = [
        metadata.get("article_number")
        for metadata in results["metadatas"][0]
    ]

    found = expected_article in articles
    rank = (
        articles.index(expected_article) + 1
        if found
        else None
    )

    print(f"\nQuestion: {question}")
    print(f"Expected Article: {expected_article}")
    print(f"Retrieved Articles: {articles}")

    if found:
        print(f"Result: FOUND at rank {rank}")
    else:
        print("Result: NOT FOUND in top 5")
