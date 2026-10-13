
import json
from pathlib import Path

JSON_PATH = Path("data/processed/articles.json")


def main() -> None:
    articles = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    print(f"Total records: {len(articles)}")

    empty_text = [
        item["article_number"]
        for item in articles
        if not item.get("text", "").strip()
    ]
    print(f"Articles with empty text: {empty_text}")

    duplicated_titles = [
        item["article_number"]
        for item in articles
        if (
            item.get("text", "").splitlines()
            and item["text"].splitlines()[0].strip() == item["title"].strip()
            and len(item["text"].splitlines()) > 1
            and item["text"].splitlines()[1].strip() == item["title"].strip()
        )
    ]
    print(f"Articles with duplicated opening titles: {duplicated_titles}")

    for number in (1, 4, 5, 99, 100, 288):
        article = next(
            (
                item for item in articles
                if item["article_number"] == number
            ),
            None,
        )

        if article:
            print(f"\n{'=' * 40}")
            print(f"Article {number}: {article['title']}")
            print(f"Chapter: {article['chapter']}")
            print(f"Start page: {article['start_page']}")
            print("--- Beginning ---")
            print(article["text"][:250])
            print("--- Ending ---")
            print(article["text"][-250:])


if __name__ == "__main__":
    main()

