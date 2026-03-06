import sqlite3
from pathlib import Path

DB_DIR = Path.home() / ".cache" / "brahe-mcp"
DB_PATH = DB_DIR / "brahe_mcp.db"


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize the SQLite database, creating the directory and schema if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)"
    )
    conn.commit()
    return conn


def get_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Get a database connection, initializing if needed."""
    return init_db(db_path)
