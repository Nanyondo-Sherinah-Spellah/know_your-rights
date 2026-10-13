
from pathlib import Path
import json
import re

import pymupdf


PDF_PATH = Path("data/raw/Uganda Constitution 1995.pdf")
OUTPUT_PATH = Path("data/processed/other_sections.json")

PREAMBLE_PAGE = 18
OBJECTIVES_START_PAGE = 19
OBJECTIVES_END_PAGE = 26

SCHEDULES_START_PAGE = 173
SCHEDULES_END_PAGE = 191

SCHEDULE_NAMES = [
    "First",
    "Second",
    "Third",
    "Fourth",
    "Fifth",
    "Sixth",
    "Seventh",
    "Eighth",
    "Ninth",
    "Tenth",
    "Eleventh",
    "Twelfth",
    "Thirteenth",
    "Fourteenth",
    "Fifteenth",
    "Sixteenth",
    "Seventeenth",
    "Eighteenth",
    "Nineteenth",
    "Twentieth",
]

SCHEDULE_PATTERN = re.compile(
    r"^\s*("
    + "|".join(SCHEDULE_NAMES)
    + r")\s+Schedule\.?\s*$",
    re.IGNORECASE,
)


def clean_lines(text: str) -> str:
    """Remove blank and underscore-only formatting lines."""

    cleaned_lines = []

    for line in text.splitlines():
        stripped_line = line.strip()

        if not stripped_line:
            continue

        # Remove PDF formatting artifacts such as _____.
        if re.fullmatch(r"_+", stripped_line):
            continue

        cleaned_lines.append(stripped_line)

    return "\n".join(cleaned_lines).strip()


def extract_page_text(
    pdf: pymupdf.Document,
    start_page: int,
    end_page: int,
) -> str:
    """Extract text from a range of PDF pages, inclusive."""

    page_texts = []

    for page_number in range(start_page, end_page + 1):
        page_text = pdf[page_number - 1].get_text("text")
        page_texts.append(page_text)

    return clean_lines("\n".join(page_texts))


def extract_preamble(pdf: pymupdf.Document) -> dict:
    """Extract the Preamble as a separate source record."""

    return {
        "section_type": "preamble",
        "title": "Preamble",
        "source_document": PDF_PATH.name,
        "start_page": PREAMBLE_PAGE,
        "end_page": PREAMBLE_PAGE,
        "text": extract_page_text(
            pdf,
            PREAMBLE_PAGE,
            PREAMBLE_PAGE,
        ),
    }


def extract_national_objectives(
    pdf: pymupdf.Document,
) -> dict:
    """Extract National Objectives and Directive Principles."""

    return {
        "section_type": "national_objectives",
        "title": (
            "National Objectives and Directive Principles "
            "of State Policy"
        ),
        "source_document": PDF_PATH.name,
        "start_page": OBJECTIVES_START_PAGE,
        "end_page": OBJECTIVES_END_PAGE,
        "text": extract_page_text(
            pdf,
            OBJECTIVES_START_PAGE,
            OBJECTIVES_END_PAGE,
        ),
    }


def extract_schedules(
    pdf: pymupdf.Document,
) -> list[dict]:
    """Extract each Schedule into a separate source record."""

    schedules = []
    current_schedule = None

    for page_number in range(
        SCHEDULES_START_PAGE,
        SCHEDULES_END_PAGE + 1,
    ):
        lines = pdf[page_number - 1].get_text(
            "text"
        ).splitlines()

        for line in lines:
            stripped_line = line.strip()

            if not stripped_line:
                continue

            match = SCHEDULE_PATTERN.match(stripped_line)

            if match:
                # Save the previous Schedule before starting
                # the next one.
                if current_schedule is not None:
                    current_schedule["text"] = clean_lines(
                        "\n".join(
                            current_schedule.pop("text_lines")
                        )
                    )

                    schedules.append(current_schedule)

                schedule_name = match.group(1).title()

                current_schedule = {
                    "section_type": "schedule",
                    "title": f"{schedule_name} Schedule",
                    "source_document": PDF_PATH.name,
                    "start_page": page_number,
                    "end_page": page_number,
                    "text_lines": [stripped_line],
                }

            elif current_schedule is not None:
                # Skip underscore-only PDF formatting artifacts.
                if not re.fullmatch(r"_+", stripped_line):
                    current_schedule["text_lines"].append(
                        stripped_line
                    )

                current_schedule["end_page"] = page_number

    # Save the final Schedule.
    if current_schedule is not None:
        current_schedule["text"] = clean_lines(
            "\n".join(current_schedule.pop("text_lines"))
        )

        schedules.append(current_schedule)

    return schedules


def main() -> None:
    """Extract and save supplementary constitutional sections."""

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Constitution PDF not found: {PDF_PATH.resolve()}"
        )

    with pymupdf.open(PDF_PATH) as pdf:
        records = [
            extract_preamble(pdf),
            extract_national_objectives(pdf),
        ]

        schedules = extract_schedules(pdf)
        records.extend(schedules)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Total supplementary records: {len(records)}")
    print(f"JSON saved to: {OUTPUT_PATH.resolve()}")

    print("\nExtracted sections:")

    for record in records:
        print(
            f"- {record['title']} "
            f"(PDF pages {record['start_page']}"
            f"–{record['end_page']})"
        )

        if not record["text"].strip():
            print("  WARNING: This section has empty text.")

        if record["text"].rstrip().endswith("_"):
            print("  WARNING: Possible underscore artifact remains.")

    schedule_count = sum(
        record["section_type"] == "schedule"
        for record in records
    )

    print(f"\nSchedules found: {schedule_count}")

    if schedule_count == 7:
        print("Schedule count check: PASS")
    else:
        print("Schedule count check: WARNING")


if __name__ == "__main__":
    main()
