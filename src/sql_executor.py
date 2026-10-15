"""Safe read-only execution of SQLite queries against the evaluation database."""

import re
import sqlite3
from pathlib import Path
from typing import Any, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "ecommerce.db"


def clean_sql_text(sql_query: str) -> str:
    """Strip markdown fences, leading/trailing comments and whitespace."""
    if not isinstance(sql_query, str):
        return ""
    text = sql_query.strip()

    # Remove markdown code fences
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Remove leading comments (e.g., -- comment or /* comment */)
    while True:
        if text.startswith("--"):
            newline_idx = text.find("\n")
            if newline_idx == -1:
                return ""
            text = text[newline_idx + 1:].strip()
        elif text.startswith("/*"):
            end_idx = text.find("*/")
            if end_idx == -1:
                return ""
            text = text[end_idx + 2:].strip()
        else:
            break

    return text.strip()


def _has_multiple_statements(query: str) -> bool:
    """Check if query contains multiple SQL statements outside string literals."""
    # Remove string literals to inspect semicolons safely
    without_strings = re.sub(r"'(?:''|[^'])*'", "", query)
    without_strings = re.sub(r'"(?:""|[^"])*"', "", without_strings)

    # Allow one optional trailing semicolon
    without_strings = without_strings.strip()
    if without_strings.endswith(";"):
        without_strings = without_strings[:-1].strip()

    return ";" in without_strings


def execute_query_detailed(
    sql_query: str,
    db_path: Optional[Path] = None,
) -> Tuple[Optional[List[Tuple[Any, ...]]], Optional[str]]:
    """Execute a single read-only SQL query and return (rows, error).

    Returns:
        (rows, None) on success, where rows is a list of tuples.
        (None, error_message) on failure.
    """
    if not isinstance(sql_query, str) or not sql_query.strip():
        return None, "Query is empty or is not a string."

    query = clean_sql_text(sql_query)
    if not query:
        return None, "Query is empty after removing comments and formatting."

    # Allow one trailing semicolon
    if query.endswith(";"):
        query = query[:-1].strip()

    # Check for multiple statements
    if _has_multiple_statements(query):
        return None, "Multiple SQL statements are not allowed."

    # Verify query begins with a read-only keyword (SELECT or WITH)
    first_word = query.split(None, 1)[0].upper() if query else ""
    if first_word not in {"SELECT", "WITH"}:
        return None, "Only read-only SELECT or WITH queries are allowed."

    target_db = Path(db_path) if db_path else DB_PATH
    if not target_db.exists():
        return None, f"Database file not found: {target_db}"

    connection = None
    try:
        # Open in read-only mode via URI
        db_uri = f"{target_db.resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(db_uri, uri=True, timeout=5)

        # Restrict SQLite operations at the authorizer level to prevent write/ddl actions
        allowed_actions = {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION,
            sqlite3.SQLITE_RECURSIVE,
        }

        def authorizer(action, arg1, arg2, db_name, trigger_name):
            if action in allowed_actions:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        connection.set_authorizer(authorizer)

        cursor = connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        return rows, None

    except sqlite3.Error as error:
        return None, f"{type(error).__name__}: {error}"
    except Exception as error:
        return None, f"ExecutionError: {error}"
    finally:
        if connection is not None:
            connection.close()


def execute_query(sql_query: str, db_path: Optional[Path] = None) -> Optional[List[Tuple[Any, ...]]]:
    """Backward-compatible wrapper: return rows or None."""
    rows, _error = execute_query_detailed(sql_query, db_path=db_path)
    return rows


if __name__ == "__main__":
    test_query = """
        SELECT name, price
        FROM products
        WHERE price BETWEEN 500 AND 2000
        ORDER BY price, product_id;
    """
    rows, err = execute_query_detailed(test_query)
    if err:
        print("Query failed:", err)
    else:
        print(f"Query succeeded ({len(rows)} rows):")
        for row in rows:
            print(row)
