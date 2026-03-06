from brahe_mcp.db import init_db


def test_init_db_creates_file(tmp_db_path):
    conn = init_db(tmp_db_path)
    conn.close()
    assert tmp_db_path.exists()


def test_init_db_creates_metadata_table(tmp_db_path):
    conn = init_db(tmp_db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
    assert cursor.fetchone() is not None
    conn.close()


def test_init_db_is_idempotent(tmp_db_path):
    conn1 = init_db(tmp_db_path)
    conn1.close()
    conn2 = init_db(tmp_db_path)
    cursor = conn2.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'")
    assert cursor.fetchone() is not None
    conn2.close()
