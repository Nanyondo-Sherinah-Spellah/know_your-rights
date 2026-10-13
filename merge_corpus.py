
import json
import re
from pathlib import Path

ARTICLES_PATH = Path("data/processed/articles.json")
OTHER_SECTIONS_PATH = Path("data/processed/other_sections.json")
OUTPUT_PATH = Path("data/processed/corpus.json")
SOURCE_DOCUMENT = "Uganda Constitution 1995.pdf"


def make_id(value: str) -> str:
    """Convert a section title into a stable, readable identifier."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug


def main() -> None:
    # Load the two extracted datasets.
    articles = json.loads(
        ARTICLES_PATH.read_text(encoding="utf-8")
    )
    other_sections = json.loads(
        OTHER_SECTIONS_PATH.read_text(encoding="utf-8")
    )

    corpus = []

    # Standardise Article records without losing existing fields.
    for article in articles:
        record = dict(article)
        record["id"] = f"constitution_article_{article['article_number']}"
        record["section_type"] = "article"
        record["source_document"] = SOURCE_DOCUMENT
        corpus.append(record)

    # Add stable IDs and consistent source metadata to other sections.
    for section in other_sections:
        record = dict(section)
        record["id"] = (
            f"constitution_{make_id(section['title'])}"
        )
        corpus.append(record)

    # Validate the combined dataset before saving it.
    ids = [record["id"] for record in corpus]
    duplicate_ids = sorted({
        record_id for record_id in ids
        if ids.count(record_id) > 1
    })

    empty_text_ids = [
        record["id"]
        for record in corpus
        if not record.get("text", "").strip()
    ]

    article_numbers = sorted(
        record["article_number"]
        for record in corpus
        if record["section_type"] == "article"
    )

    expected_numbers = list(range(1, 289))

    if duplicate_ids:
        raise ValueError(f"Duplicate IDs found: {duplicate_ids}")

    if empty_text_ids:
        raise ValueError(f"Records with empty text: {empty_text_ids}")

    if article_numbers != expected_numbers:
        raise ValueError(
            "Article numbering check failed. "
            f"Found {len(article_numbers)} Articles."
        )

    if len(other_sections) != 9:
        raise ValueError(
            f"Expected 9 supplementary records; found {len(other_sections)}."
        )

    # Save only after the checks pass.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Corpus created successfully.")
    print(f"Articles: {len(articles)}")
    print(f"Supplementary sections: {len(other_sections)}")
    print(f"Total records: {len(corpus)}")
    print(f"Unique IDs: {len(set(ids))}")
    print(f"Empty text records: {len(empty_text_ids)}")
    print(f"Saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
