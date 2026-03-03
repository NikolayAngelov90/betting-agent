"""Migrate local SQLite database to Neon PostgreSQL.

Uses PostgreSQL COPY protocol for fast bulk loading (~20x faster than INSERT).

Usage:
    DATABASE_URL="postgresql://..." python scripts/migrate_to_neon.py
"""

import os
import sys
import io
import sqlite3
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg2
from sqlalchemy import create_engine, text, inspect
from src.data.models import Base


SQLITE_PATH = Path("data/football_betting.db")
TABLE_ORDER = ["teams", "matches", "players", "injuries", "odds", "news", "saved_picks", "predictions"]

BOOL_COLUMNS = {
    "matches": {"is_fixture"},
    "players": {"is_key_player"},
    "saved_picks": {"used_fallback_odds"},
}


def coerce_value(table, col, val):
    """Fix SQLite→PostgreSQL type mismatches for COPY format."""
    if val is None:
        return "\\N"  # PostgreSQL COPY NULL marker
    if col in BOOL_COLUMNS.get(table, set()):
        return "t" if val else "f"
    s = str(val)
    # Escape tabs and newlines for COPY format
    s = s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
    return s


def get_pg_conn(db_url):
    """Get raw psycopg2 connection from DATABASE_URL."""
    return psycopg2.connect(db_url)


def migrate():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: Set DATABASE_URL env var")
        sys.exit(1)

    if not SQLITE_PATH.exists():
        print(f"ERROR: SQLite DB not found at {SQLITE_PATH}")
        sys.exit(1)

    sqlite_conn = sqlite3.connect(str(SQLITE_PATH))
    sqlite_conn.row_factory = sqlite3.Row

    # Create tables via SQLAlchemy (handles model schema)
    print("Creating tables in Neon...")
    sa_engine = create_engine(db_url, pool_pre_ping=True)
    Base.metadata.create_all(sa_engine)

    # Get existing counts
    pg_counts = {}
    with sa_engine.connect() as conn:
        for table in TABLE_ORDER:
            try:
                pg_counts[table] = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            except Exception:
                pg_counts[table] = 0
    sa_engine.dispose()

    # Use raw psycopg2 for COPY
    pg_conn = get_pg_conn(db_url)

    for table in TABLE_ORDER:
        sqlite_count = sqlite_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if sqlite_count == 0:
            print(f"  {table}: empty — skipping")
            continue

        pg_existing = pg_counts.get(table, 0)
        if pg_existing >= sqlite_count:
            print(f"  {table}: already migrated ({pg_existing:,} rows) — skipping")
            continue

        # Get column info
        cols_info = sqlite_conn.execute(f"PRAGMA table_info({table})").fetchall()
        col_names = [c["name"] for c in cols_info]
        cols_str = ", ".join(col_names)

        # If partial migration, we need a temp table approach to avoid duplicates
        if pg_existing > 0:
            print(f"  {table}: resuming ({pg_existing:,}/{sqlite_count:,} done)...", end="", flush=True)
            # Use INSERT ... ON CONFLICT for remaining rows via COPY to temp table
            use_temp = True
        else:
            print(f"  {table}: loading {sqlite_count:,} rows via COPY...", end="", flush=True)
            use_temp = False

        # Build TSV data in memory (stream it in chunks for large tables)
        cursor = sqlite_conn.execute(f"SELECT {cols_str} FROM {table}")
        pg_cursor = pg_conn.cursor()

        if use_temp:
            # Create temp table, COPY into it, then INSERT ... ON CONFLICT
            pg_cursor.execute(f"CREATE TEMP TABLE _tmp_{table} (LIKE {table} INCLUDING ALL)")
            target_table = f"_tmp_{table}"
        else:
            target_table = table

        # Stream data in chunks to avoid memory issues
        CHUNK_SIZE = 50000
        total_copied = 0
        while True:
            rows = cursor.fetchmany(CHUNK_SIZE)
            if not rows:
                break

            buf = io.StringIO()
            for row in rows:
                line = "\t".join(coerce_value(table, col_names[i], row[i]) for i in range(len(col_names)))
                buf.write(line + "\n")

            buf.seek(0)
            pg_cursor.copy_from(buf, target_table, columns=col_names)
            pg_conn.commit()
            total_copied += len(rows)
            print(f"\r  {table}: {total_copied:,}/{sqlite_count:,}", end="", flush=True)

        if use_temp:
            # Merge temp into main table
            pg_cursor.execute(f"""
                INSERT INTO {table} ({cols_str})
                SELECT {cols_str} FROM _tmp_{table}
                ON CONFLICT DO NOTHING
            """)
            pg_conn.commit()
            pg_cursor.execute(f"DROP TABLE _tmp_{table}")
            pg_conn.commit()

        # Verify
        pg_cursor.execute(f"SELECT COUNT(*) FROM {table}")
        final_count = pg_cursor.fetchone()[0]
        print(f"\r  {table}: {final_count:,} rows in Neon (source: {sqlite_count:,})        ")

        # Reset sequence
        if "id" in col_names:
            try:
                pg_cursor.execute(
                    f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                    f"COALESCE((SELECT MAX(id) FROM {table}), 1))"
                )
                pg_conn.commit()
            except Exception:
                pg_conn.rollback()

        pg_cursor.close()

    pg_conn.close()
    sqlite_conn.close()
    print("\nMigration complete!")


if __name__ == "__main__":
    migrate()
