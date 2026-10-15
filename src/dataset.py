"""Dataset loading, validation, and schema consistency checking."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "golden_dataset.json"

REQUIRED_FIELDS = {"id", "question", "expected_sql", "category", "difficulty"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_CATEGORIES = {"filtering", "sorting", "aggregation", "joins", "subquery"}


def infer_order_matters(question: str, expected_sql: str) -> bool:
    """Determine if query result ordering is semantically significant."""
    sql_upper = expected_sql.upper()
    q_lower = question.lower()

    # Queries with LIMIT inherently require deterministic ordering
    if "LIMIT " in sql_upper:
        return True

    # Questions that explicitly demand ordering in natural language
    ordering_phrases = [
        "sorted",
        "ordered",
        "cheapest to most expensive",
        "most expensive to cheapest",
        "highest to lowest",
        "lowest to highest",
        "from cheapest",
    ]
    return any(phrase in q_lower for phrase in ordering_phrases)


def load_dataset(dataset_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load and perform structure validation on the golden benchmark dataset."""
    target_path = Path(dataset_path) if dataset_path else DATASET_PATH

    if not target_path.exists():
        raise FileNotFoundError(f"Golden dataset file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError("The golden dataset must be a JSON array of examples.")

    seen_ids = set()
    for index, example in enumerate(dataset):
        if not isinstance(example, dict):
            raise ValueError(f"Example at index {index} is not a JSON object.")

        missing = REQUIRED_FIELDS - example.keys()
        if missing:
            raise ValueError(
                f"Example {example.get('id', f'at index {index}')} is missing fields: {missing}"
            )

        cid = example["id"]
        if not isinstance(cid, int):
            raise ValueError(f"Example ID must be an integer, got {type(cid)}: {cid}")

        if cid in seen_ids:
            raise ValueError(f"Duplicate example ID detected in dataset: {cid}")
        seen_ids.add(cid)

        # Set or infer order_matters
        if "order_matters" not in example:
            example["order_matters"] = infer_order_matters(
                example["question"], example["expected_sql"]
            )

    return dataset


def validate_dataset(
    dataset_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Validate all reference queries against the actual SQLite database.

    Returns:
        (is_valid, failures_list)
    """
    from .sql_executor import execute_query_detailed

    dataset = load_dataset(dataset_path)
    failures = []

    for example in dataset:
        sql = example["expected_sql"]
        rows, error = execute_query_detailed(sql, db_path=db_path)

        if error or rows is None:
            failures.append({
                "id": example["id"],
                "question": example["question"],
                "expected_sql": sql,
                "error": error or "Unknown execution failure",
            })

    is_valid = len(failures) == 0
    return is_valid, failures


if __name__ == "__main__":
    import sys
    print("Loading and validating golden dataset...")
    try:
        data = load_dataset()
        print(f"Loaded {len(data)} examples successfully.")
        valid, failed_cases = validate_dataset()
        if valid:
            print(f"Validation successful: All {len(data)} reference queries executed cleanly.")
        else:
            print(f"Validation failed: {len(failed_cases)} reference queries failed:")
            for f in failed_cases:
                print(f"  [ID {f['id']}] {f['error']}")
            sys.exit(1)
    except Exception as exc:
        print(f"Dataset error: {exc}")
        sys.exit(1)
