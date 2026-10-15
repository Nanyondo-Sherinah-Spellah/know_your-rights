
from rag import retrieve_passages, TOP_K


# ============================================================
# 1. EVALUATION DATASET
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
# 2. EVALUATE ONE QUESTION
# ============================================================

def evaluate_case(test_case: dict) -> dict:
    """Evaluate the same source retrieval pipeline used by rag.py."""

    question = test_case["question"]
    expected_articles = test_case["expected_articles"]

    # Reuse production retrieval instead of querying ChromaDB
    # independently. This includes production deduplication and
    # source reconstruction.
    retrieved = retrieve_passages(
        question=question,
        top_k=TOP_K,
    )

    matching_ranks = []

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print(f"EXPECTED ARTICLE(S): {expected_articles}")
    print("-" * 70)

    for rank, passage in enumerate(retrieved, start=1):
        article = passage.get("article_number")
        title = passage.get("title") or "Untitled section"
        distance = passage.get("distance")

        # Normalize the article number for reliable comparison.
        try:
            article_number = (
                int(article) if article is not None else None
            )
        except (TypeError, ValueError):
            article_number = None

        if article_number in expected_articles:
            matching_ranks.append(rank)

        if article_number is not None:
            reference = f"Article {article_number}: {title}"
        else:
            reference = title

        distance_text = (
            f"{float(distance):.4f}"
            if distance is not None
            else "N/A"
        )

        marker = (
            "  <-- EXPECTED ARTICLE"
            if article_number in expected_articles
            else ""
        )

        print(
            f"Rank {rank}: {reference}"
            f" | Distance: {distance_text}{marker}"
        )

    first_match_rank = (
        min(matching_ranks) if matching_ranks else None
    )

    hit_at_1 = first_match_rank is not None and first_match_rank <= 1
    hit_at_3 = first_match_rank is not None and first_match_rank <= 3
    hit_at_5 = first_match_rank is not None and first_match_rank <= 5

    reciprocal_rank = (
        1 / first_match_rank
        if first_match_rank is not None
        else 0.0
    )

    if first_match_rank is None:
        print("RESULT: FAIL — expected article not found.")
    else:
        print(
            "RESULT: PASS — expected article found "
            f"at rank {first_match_rank}."
        )

    return {
        "question": question,
        "expected_articles": expected_articles,
        "first_match_rank": first_match_rank,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5,
        "reciprocal_rank": reciprocal_rank,
    }


# ============================================================
# 3. SUMMARIZE THE EVALUATION
# ============================================================

def print_summary(results: list[dict]) -> None:
    """Print retrieval coverage, ranking quality, and MRR."""

    total = len(results)

    if total == 0:
        print("No test cases were evaluated.")
        return

    hit_1 = sum(result["hit_at_1"] for result in results)
    hit_3 = sum(result["hit_at_3"] for result in results)
    hit_5 = sum(result["hit_at_5"] for result in results)

    mrr = (
        sum(result["reciprocal_rank"] for result in results)
        / total
    )

    print("\n\n" + "=" * 70)
    print("KNOW YOUR RIGHTS AI — RETRIEVAL EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Total test questions: {total}")
    print(f"Hit@1: {hit_1}/{total} ({hit_1 / total:.1%})")
    print(f"Hit@3: {hit_3}/{total} ({hit_3 / total:.1%})")
    print(f"Hit@5: {hit_5}/{total} ({hit_5 / total:.1%})")
    print(f"MRR:   {mrr:.4f}")

    print("\nINTERPRETATION")
    print("Hit@1: expected article appears first.")
    print("Hit@3: expected article appears in the top three.")
    print("Hit@5: expected article appears in the top five.")
    print(
        "MRR: Mean Reciprocal Rank; higher values indicate "
        "that expected articles tend to rank closer to first."
    )

    print(
        "\nIMPORTANT LIMITATIONS\n"
        "1. Results evaluate source retrieval, not legal accuracy.\n"
        "2. The dataset contains only manually labelled test cases.\n"
        "3. Expected article labels must be reviewed for completeness.\n"
        "4. This evaluator uses the production retrieval function,\n"
        "   so changes to that function also affect these results."
    )


# ============================================================
# 4. RUN THE EVALUATION
# ============================================================

def main() -> None:
    print("=" * 70)
    print("KNOW YOUR RIGHTS AI — PRODUCTION RETRIEVAL EVALUATION")
    print("=" * 70)

    print(
        "\nUsing retrieve_passages() from rag.py."
        "\nNo answer-generation API calls are made by this script."
    )

    results = []

    for test_case in TEST_CASES:
        result = evaluate_case(test_case)
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()
