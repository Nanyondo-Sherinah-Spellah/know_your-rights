
from sentence_transformers import SentenceTransformer

# Three example passages from your constitutional corpus.
passages = [
    {
        "id": "article_23",
        "text": (
            "Protection of personal liberty. A person arrested, "
            "restricted or detained shall be kept in a place "
            "authorised by law."
        ),
    },
    {
        "id": "article_28",
        "text": (
            "Right to a fair hearing. A person shall be entitled "
            "to a fair, speedy and public hearing before an "
            "independent and impartial court or tribunal."
        ),
    },
    {
        "id": "article_2",
        "text": (
            "Supremacy of the Constitution. This Constitution "
            "is the supreme law of Uganda and shall have binding "
            "force on all authorities and persons throughout Uganda."
        ),
    },
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts = [passage["text"] for passage in passages]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
)

print(f"Passages embedded: {len(embeddings)}")
print(f"Vector dimensions: {embeddings.shape[1]}")
print(f"Vector for first passage (first 5 values):")
print(embeddings[0][:5])
