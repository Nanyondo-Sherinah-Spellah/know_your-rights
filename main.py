
from pathlib import Path
import json
import re

import pymupdf


PDF_PATH = Path("data/raw/Uganda Constitution 1995.pdf")
OUTPUT_PATH = Path("data/processed/articles.json")

FIRST_ARTICLE_PAGE = 27
LAST_ARTICLE_PAGE = 172


def extract_articles(pdf_path: Path) -> list[dict]:
    """Extract Articles 1–288 from the Constitution PDF."""

    articles = []
    current_article = None
    current_chapter = None
    expected_number = 1

    def finish_article() -> None:
        """Save the current article after collecting its text."""
        nonlocal current_article

        if current_article is not None:
            current_article["text"] = "\n".join(
                current_article.pop("text_lines")
            ).strip()

            articles.append(current_article)
            current_article = None

    with pymupdf.open(pdf_path) as pdf:

        for page_index in range(
            FIRST_ARTICLE_PAGE - 1,
            LAST_ARTICLE_PAGE,
        ):
            page = pdf[page_index]
            page_number = page_index + 1
            lines = page.get_text("text").splitlines()

            index = 0

            while index < len(lines):
                line = lines[index].strip()

                # Skip empty lines.
                if not line:
                    index += 1
                    continue

                # Detect chapter headings and their titles.
                chapter_match = re.match(
                    r"^Chapter\s+([A-Z0-9]+)\s*$",
                    line,
                    re.IGNORECASE,
                )

                if chapter_match:
                    next_index = index + 1

                    # Find the next non-empty line, which is the
                    # chapter title.
                    while (
                        next_index < len(lines)
                        and not lines[next_index].strip()
                    ):
                        next_index += 1

                    if next_index < len(lines):
                        chapter_title = lines[next_index].strip()

                        current_chapter = (
                            f"Chapter {chapter_match.group(1)}: "
                            f"{chapter_title}"
                        )

                        # Skip both the chapter heading and its title.
                        index = next_index + 1
                    else:
                        index += 1

                    continue

                # Detect the next expected article number.
                article_match = re.match(
                    r"^(\d{1,3})\.\s*(.*)$",
                    line,
                )

                if (
                    article_match
                    and int(article_match.group(1)) == expected_number
                ):
                    title = article_match.group(2).strip()

                    # Some article numbers and titles appear on
                    # separate lines in the PDF.
                    if not title:
                        next_index = index + 1

                        while (
                            next_index < len(lines)
                            and not lines[next_index].strip()
                        ):
                            next_index += 1

                        if next_index >= len(lines):
                            index += 1
                            continue

                        title = lines[next_index].strip()

                        # Avoid treating clause text or another
                        # chapter heading as an article title.
                        if (
                            title.startswith("(")
                            or title.lower().startswith("chapter ")
                        ):
                            index += 1
                            continue

                        index = next_index

                    # Save the previous article before starting
                    # the next one.
                    finish_article()

                    current_article = {
                        "article_number": expected_number,
                        "title": title,
                        "chapter": current_chapter,
                        "start_page": page_number,
                        "text_lines": [title],
                    }

                    expected_number += 1
                    index += 1
                    continue

                # Add ordinary text to the current article.
                if current_article is not None:
                     if not re.fullmatch(r"_+", line):
                         current_article["text_lines"].append(line)

                index += 1

    # Save the final article.
    finish_article()

    return articles


def main() -> None:
    """Extract articles, save JSON, and verify numbering."""

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Constitution PDF not found: {PDF_PATH.resolve()}"
        )

    articles = extract_articles(PDF_PATH)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            articles,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Articles extracted: {len(articles)}")
    print(f"JSON saved to: {OUTPUT_PATH.resolve()}")

    # Verify that Articles 1–288 were extracted in order.
    numbers = [
        article["article_number"]
        for article in articles
    ]

    expected_numbers = list(range(1, 289))

    if numbers == expected_numbers:
        print(
            "Numbering check: PASS "
            "(Articles 1–288 found in order)"
        )
    else:
        print("Numbering check: WARNING")

        missing = sorted(
            set(expected_numbers) - set(numbers)
        )

        duplicates = sorted(
            number
            for number in set(numbers)
            if numbers.count(number) > 1
        )

        print(f"Missing Article numbers: {missing}")
        print(f"Duplicate Article numbers: {duplicates}")

    # Display selected articles for a quick inspection.
    for number in (1, 2, 4, 5, 99, 100, 288):
        article = next(
            (
                item
                for item in articles
                if item["article_number"] == number
            ),
            None,
        )

        if article:
            print(
                f"Article {number}: {article['title']} "
                f"(PDF page {article['start_page']})"
            )

            print(f"Chapter: {article['chapter']}")

            if number == 4:
                print("Article 4 ending:")
                print(article["text"][-200:])


if __name__ == "__main__":
    main()
