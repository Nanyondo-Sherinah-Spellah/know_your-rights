
import json
from pathlib import Path


JSON_PATH = Path("data/processed/other_sections.json")


def main() -> None:
    records = json.loads(
        JSON_PATH.read_text(encoding="utf-8")
    )

    print(f"Total records: {len(records)}")

    empty_records = [
        record["title"]
        for record in records
        if not record.get("text", "").strip()
    ]

    print(f"Records with empty text: {empty_records}")

    print("\nRecord details:")

    for record in records:
        text = record.get("text", "")

        print(f"\n{'=' * 50}")
        print(f"Title: {record['title']}")
        print(f"Type: {record['section_type']}")
        print(
            f"Pages: {record['start_page']}"
            f"–{record['end_page']}"
        )
        print(f"Characters: {len(text)}")
        print(f"Beginning: {text[:180]!r}")
        print(f"Ending: {text[-180:]!r}")

    schedules = [
        record
        for record in records
        if record["section_type"] == "schedule"
    ]

    expected_schedule_titles = [
        "First Schedule",
        "Second Schedule",
        "Third Schedule",
        "Fourth Schedule",
        "Fifth Schedule",
        "Sixth Schedule",
        "Seventh Schedule",
    ]

    actual_schedule_titles = [
        record["title"]
        for record in schedules
    ]

    print("\nSchedule titles check:")

    if actual_schedule_titles == expected_schedule_titles:
        print("PASS")
    else:
        print("WARNING")
        print(f"Expected: {expected_schedule_titles}")
        print(f"Actual: {actual_schedule_titles}")


if __name__ == "__main__":
    main()
