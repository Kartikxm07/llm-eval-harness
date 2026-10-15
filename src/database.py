"""Database initialization and schema management for the e-commerce benchmark."""

import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "ecommerce.db"


def create_database(db_path=None):
    """Initialize SQLite database with schema and sample data."""
    target_path = Path(db_path) if db_path else DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(target_path) as connection:
        cursor = connection.cursor()

        # Enforce foreign key constraints
        cursor.execute("PRAGMA foreign_keys = ON;")

        # Products available in the store
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL CHECK(price >= 0),
                stock INTEGER NOT NULL CHECK(stock >= 0)
            )
        """)

        # Customers registered in the store
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT NOT NULL
            )
        """)

        # Orders placed by customers
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                order_date TEXT NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
            )
        """)

        # Products contained in each order
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS order_items (
                order_item_id INTEGER PRIMARY KEY,
                order_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL CHECK(quantity > 0),
                unit_price REAL NOT NULL CHECK(unit_price >= 0),
                FOREIGN KEY(order_id) REFERENCES orders(order_id),
                FOREIGN KEY(product_id) REFERENCES products(product_id)
            )
        """)

        # Insert sample data
        cursor.executemany("""
            INSERT OR IGNORE INTO products
            VALUES (?, ?, ?, ?, ?)
        """, [
            (1, "Wireless Mouse", "Electronics", 799.0, 40),
            (2, "Mechanical Keyboard", "Electronics", 2499.0, 20),
            (3, "USB-C Cable", "Accessories", 299.0, 100),
            (4, "Running Shoes", "Fashion", 1999.0, 15),
            (5, "Cotton T-Shirt", "Fashion", 599.0, 50),
            (6, "Bluetooth Headphones", "Electronics", 1799.0, 25),
            (7, "Laptop Stand", "Accessories", 999.0, 30),
            (8, "Water Bottle", "Home", 399.0, 60)
        ])

        cursor.executemany("""
            INSERT OR IGNORE INTO customers
            VALUES (?, ?, ?)
        """, [
            (1, "Aarav", "Mumbai"),
            (2, "Diya", "Pune"),
            (3, "Kabir", "Delhi"),
            (4, "Anaya", "Mumbai"),
            (5, "Ishaan", "Nagpur")
        ])

        cursor.executemany("""
            INSERT OR IGNORE INTO orders
            VALUES (?, ?, ?, ?)
        """, [
            (101, 1, "2026-09-01", "delivered"),
            (102, 2, "2026-09-03", "shipped"),
            (103, 1, "2026-09-05", "delivered"),
            (104, 3, "2026-09-07", "cancelled"),
            (105, 4, "2026-09-10", "delivered"),
            (106, 5, "2026-09-12", "processing")
        ])

        cursor.executemany("""
            INSERT OR IGNORE INTO order_items
            VALUES (?, ?, ?, ?, ?)
        """, [
            (1, 101, 1, 2, 799.0),
            (2, 101, 3, 1, 299.0),
            (3, 102, 2, 1, 2499.0),
            (4, 103, 6, 1, 1799.0),
            (5, 103, 8, 2, 399.0),
            (6, 104, 4, 1, 1999.0),
            (7, 105, 5, 3, 599.0),
            (8, 105, 7, 1, 999.0),
            (9, 106, 1, 1, 799.0)
        ])

        connection.commit()

    return target_path


if __name__ == "__main__":
    path = create_database()
    print(f"E-commerce database created: {path}")
