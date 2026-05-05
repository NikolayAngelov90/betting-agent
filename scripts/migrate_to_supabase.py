"""Migrate Neon PostgreSQL to Supabase PostgreSQL.

Uses psycopg2 COPY TO STDOUT / COPY FROM STDIN for fast pg->pg bulk transfer.
No pg_dump required -- runs entirely in Python.

Usage:
    python scripts/migrate_to_supabase.py
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg2
from sqlalchemy import create_engine

from src.data.models import Base

# ---------------------------------------------------------------------------
# Connection strings
# ---------------------------------------------------------------------------
SOURCE_URL = (
    "postgresql://neondb_owner:npg_z6yAW0jthPHg"
    "@ep-bold-field-al1me8dx-pooler.c-3.eu-central-1.aws.neon.tech"
    "/neondb?sslmode=require"
)

# Session pooler at port 5432 (session mode) -- supports COPY, DDL, and transactions.
# Also used at runtime by the app and CI.
TARGET_URL_POOLER = (
    "postgresql://postgres.nhlurscyrlvpjzapmqcr:ofA5FEPTmjHzEtkQ"
    "@aws-1-eu-central-1.pooler.supabase.com:5432/postgres?sslmode=require"
)

# Alias used throughout the script
TARGET_URL_DIRECT = TARGET_URL_POOLER

# Migration order respects FK dependencies
TABLE_ORDER = [
    "teams", "matches", "players", "injuries",
    "odds", "news", "saved_picks", "predictions",
]

CHUNK = 50_000  # rows per COPY batch


def row_count(conn, table):
    with conn.cursor() as cur:
        cur.execute(f'SELECT COUNT(*) FROM "{table}"')
        return cur.fetchone()[0]


def get_columns(conn, table):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s "
            "ORDER BY ordinal_position",
            (table,),
        )
        return [r[0] for r in cur.fetchall()]


def copy_table(src_conn, dst_conn, table):
    src_count = row_count(src_conn, table)
    dst_existing = row_count(dst_conn, table)

    if src_count == 0:
        print(f"  {table}: empty in source -- skipping")
        return

    if dst_existing >= src_count:
        print(f"  {table}: already complete ({dst_existing:,} rows) -- skipping")
        return

    cols = get_columns(src_conn, table)
    cols_sql = ", ".join(f'"{c}"' for c in cols)

    if dst_existing > 0:
        print(f"  {table}: resuming ({dst_existing:,}/{src_count:,}) via temp table...")
        use_temp = True
    else:
        print(f"  {table}: copying {src_count:,} rows...", end="", flush=True)
        use_temp = False

    src_cur = src_conn.cursor()
    dst_cur = dst_conn.cursor()

    if use_temp:
        dst_cur.execute(f'CREATE TEMP TABLE _tmp_{table} (LIKE "{table}" INCLUDING ALL)')
        dst_conn.commit()
        copy_target = f"_tmp_{table}"   # copy_from takes unquoted name
    else:
        copy_target = table             # copy_from takes unquoted name

    # Stream from source in batches using server-side cursor
    src_cur.execute(f'DECLARE _mig CURSOR FOR SELECT {cols_sql} FROM "{table}"')

    total = 0
    while True:
        src_cur.execute(f"FETCH {CHUNK} FROM _mig")
        rows = src_cur.fetchall()
        if not rows:
            break

        text_buf = io.StringIO()
        for row in rows:
            parts = []
            for v in row:
                if v is None:
                    parts.append("\\N")
                else:
                    s = str(v)
                    s = (s.replace("\\", "\\\\")
                          .replace("\t", "\\t")
                          .replace("\n", "\\n")
                          .replace("\r", "\\r"))
                    parts.append(s)
            text_buf.write("\t".join(parts) + "\n")

        text_buf.seek(0)
        dst_cur.copy_from(text_buf, copy_target, columns=cols)
        dst_conn.commit()
        total += len(rows)
        print(f"\r  {table}: {total:,}/{src_count:,}", end="", flush=True)

    src_cur.execute("CLOSE _mig")
    src_conn.commit()

    if use_temp:
        dst_cur.execute(f"""
            INSERT INTO "{table}" ({cols_sql})
            SELECT {cols_sql} FROM _tmp_{table}
            ON CONFLICT DO NOTHING
        """)
        dst_conn.commit()
        dst_cur.execute(f"DROP TABLE _tmp_{table}")
        dst_conn.commit()

    final = row_count(dst_conn, table)
    print(f"\r  {table}: {final:,} rows done" + " " * 20)

    # Reset PK sequence so new inserts don't collide
    if "id" in cols:
        try:
            dst_cur.execute(
                f"SELECT setval(pg_get_serial_sequence('\"{table}\"', 'id'), "
                f"COALESCE((SELECT MAX(id) FROM \"{table}\"), 1))"
            )
            dst_conn.commit()
        except Exception:
            dst_conn.rollback()

    src_cur.close()
    dst_cur.close()


def migrate():
    print("=" * 60)
    print("Neon -> Supabase migration")
    print("=" * 60)

    # Step 1: create schema in Supabase via SQLAlchemy
    print("\n[1/3] Creating schema in Supabase...")
    engine = create_engine(TARGET_URL_DIRECT, connect_args={"sslmode": "require"})
    Base.metadata.create_all(engine)
    engine.dispose()
    print("      Schema ready.")

    # Step 2: open both raw psycopg2 connections
    print("\n[2/3] Connecting...")
    src = psycopg2.connect(SOURCE_URL)
    src.autocommit = False
    dst = psycopg2.connect(TARGET_URL_DIRECT)
    dst.autocommit = False
    print("      Source (Neon) and target (Supabase) connected.")

    # Discover which tables actually exist in the source DB
    with src.cursor() as cur:
        cur.execute(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname = 'public' ORDER BY tablename"
        )
        src_tables = {r[0] for r in cur.fetchall()}

    # Step 3: migrate each table
    print("\n[3/3] Migrating tables...")
    for table in TABLE_ORDER:
        if table not in src_tables:
            print(f"  {table}: not in source DB -- skipping")
            continue
        try:
            copy_table(src, dst, table)
        except Exception as exc:
            print(f"\n  ERROR on {table}: {exc}")
            src.rollback()
            dst.rollback()
            raise

    src.close()
    dst.close()

    print("\n" + "=" * 60)
    print("Migration complete!")
    print()
    print("DATABASE_URL for GitHub secret (Session pooler):")
    print(f"  {TARGET_URL_POOLER}")
    print("=" * 60)


if __name__ == "__main__":
    migrate()
