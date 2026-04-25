"""Database manager for Football Betting Agent."""

import os
from pathlib import Path
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text, inspect
from sqlalchemy.orm import sessionmaker, Session

from src.data.models import Base
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger()


class DatabaseManager:
    """Manages database connections, sessions, and operations."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(bind=self.engine)

    @property
    def is_postgres(self) -> bool:
        return self.engine.dialect.name == "postgresql"

    def _create_engine(self):
        db_config = self.config.database

        # 1. Prefer DATABASE_URL env var (works for both local + CI)
        database_url = db_config.get("url") or os.environ.get("DATABASE_URL")

        if database_url:
            logger.info(f"Using PostgreSQL database (Neon)")
            return create_engine(
                database_url, echo=False,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                pool_recycle=300,  # recycle connections every 5min (Neon scale-to-zero)
            )

        # 2. Fall back to SQLite (local dev without DATABASE_URL)
        db_path = Path(db_config.get("sqlite_path", "data/football_betting.db"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"
        logger.info(f"Using SQLite database: {db_path}")

        engine = create_engine(
            url, echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

        return engine

    def create_tables(self):
        """Create all tables defined in models, and add any missing columns/indexes."""
        Base.metadata.create_all(self.engine)
        self._migrate_missing_columns()
        self._migrate_missing_indexes()
        self._drop_removed_tables()
        logger.info("Database tables created successfully")

    # Allowlist of tables that were removed from models and should be dropped.
    # Only names in this set will be processed — never user/dynamic input.
    _REMOVED_TABLES = frozenset(["predictions"])

    def _drop_removed_tables(self):
        """Drop tables that no longer have corresponding models."""
        import re
        inspector = inspect(self.engine)
        existing = inspector.get_table_names()
        for table in self._REMOVED_TABLES:
            if not re.fullmatch(r"[a-z_][a-z0-9_]*", table):
                logger.warning(f"Skipping invalid table name in _REMOVED_TABLES: {table!r}")
                continue
            if table in existing:
                with self.engine.begin() as conn:
                    cascade = " CASCADE" if self.is_postgres else ""
                    conn.execute(text(f'DROP TABLE IF EXISTS "{table}"{cascade}'))
                logger.info(f"Dropped removed table: {table}")

    def _migrate_missing_columns(self):
        """Auto-add columns that exist in models but not in the DB.

        Only adds NULLABLE columns, OR NOT NULL columns that carry a server-
        side default (so existing rows can be back-filled atomically).
        Anything else requires a real migration script and is skipped with a
        WARNING — silently adding a NOT NULL column without a default would
        fail on tables that already have rows.
        """
        inspector = inspect(self.engine)
        for table_name, table in Base.metadata.tables.items():
            if table_name not in inspector.get_table_names():
                continue

            existing_cols = {c["name"] for c in inspector.get_columns(table_name)}
            for col in table.columns:
                if col.name in existing_cols:
                    continue

                has_server_default = col.server_default is not None
                if not col.nullable and not has_server_default:
                    logger.warning(
                        f"Migration: skipping {table_name}.{col.name} — column is "
                        f"NOT NULL without a server default. Write an explicit "
                        f"migration that backfills existing rows before adding "
                        f"the constraint."
                    )
                    continue

                col_type = col.type.compile(self.engine.dialect)
                nullable_clause = "" if col.nullable else " NOT NULL"
                default_clause = ""
                if has_server_default and hasattr(col.server_default, "arg"):
                    default_clause = f" DEFAULT {col.server_default.arg}"
                sql = (
                    f"ALTER TABLE {table_name} ADD COLUMN "
                    f"{col.name} {col_type}{default_clause}{nullable_clause}"
                )
                try:
                    with self.engine.begin() as conn:
                        conn.execute(text(sql))
                    logger.info(
                        f"Migration: added column {table_name}.{col.name} "
                        f"({col_type}{nullable_clause}{default_clause})"
                    )
                except Exception as e:
                    logger.debug(f"Column {table_name}.{col.name} migration skipped: {e}")

    def _migrate_missing_indexes(self):
        """Create any indexes defined in models that don't yet exist in the DB."""
        inspector = inspect(self.engine)
        for table_name, table in Base.metadata.tables.items():
            if table_name not in inspector.get_table_names():
                continue
            existing_indexes = {idx["name"] for idx in inspector.get_indexes(table_name)}
            for idx in table.indexes:
                if idx.name and idx.name not in existing_indexes:
                    try:
                        idx.create(self.engine)
                        logger.info(f"Migration: created index {idx.name} on {table_name}")
                    except Exception as e:
                        logger.debug(f"Index {idx.name} creation skipped: {e}")

    def prune_old_odds(self, keep_days: int = 400):
        """Delete odds for matches older than keep_days to control DB size.

        Keeps all odds for matches within the retention window and any match
        that has an associated saved_pick (so we never lose betting history).
        """
        from datetime import timedelta
        from src.data.models import Odds, Match, SavedPick
        from src.utils.logger import utcnow
        cutoff = utcnow() - timedelta(days=keep_days)
        with self.get_session() as session:
            # Subquery: match IDs older than cutoff that have NO saved picks
            old_match_ids = (
                session.query(Match.id)
                .filter(Match.match_date < cutoff)
                .subquery()
            )
            pick_match_ids = (
                session.query(SavedPick.match_id)
                .distinct()
                .subquery()
            )
            deleted = (
                session.query(Odds)
                .filter(
                    Odds.match_id.in_(session.query(old_match_ids.c.id)),
                    ~Odds.match_id.in_(session.query(pick_match_ids.c.match_id)),
                )
                .delete(synchronize_session=False)
            )
            if deleted:
                logger.info(f"Pruned {deleted:,} old odds rows (matches before {cutoff.date()})")

    def drop_tables(self):
        """Drop all tables. Use with caution."""
        Base.metadata.drop_all(self.engine)
        logger.warning("All database tables dropped")

    def health_check(self) -> bool:
        """Verify the database connection is alive."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    @contextmanager
    def get_session(self):
        """Provide a transactional session scope.

        Usage:
            with db.get_session() as session:
                session.add(team)
        """
        session: Session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def add(self, session: Session, obj):
        """Add a single object to the session."""
        session.add(obj)
        return obj

    def add_all(self, session: Session, objects: list):
        """Add multiple objects to the session."""
        session.add_all(objects)
        return objects

    def get_by_id(self, session: Session, model, obj_id: int):
        """Get a record by its primary key."""
        return session.get(model, obj_id)

    def get_all(self, session: Session, model, limit: int = None):
        """Get all records of a model, with optional limit."""
        query = session.query(model)
        if limit:
            query = query.limit(limit)
        return query.all()

    def query(self, session: Session, model):
        """Return a query object for the given model."""
        return session.query(model)


# Global database manager instance (thread-safe lazy init)
import threading
_db_manager = None
_db_lock = threading.Lock()


def get_db() -> DatabaseManager:
    """Get global database manager instance (thread-safe)."""
    global _db_manager
    if _db_manager is None:
        with _db_lock:
            if _db_manager is None:
                _db_manager = DatabaseManager()
    return _db_manager


def init_db():
    """Initialize database: create engine and all tables.

    For SQLite: if the cached DB file is malformed, it is deleted and recreated.
    For PostgreSQL: tables are created if they don't exist.

    When the env var TABLES_CREATED=1 is set (CI: set after the first invocation),
    create_tables() is skipped to avoid redundant Neon cold-start + DDL overhead.
    """
    import os as _os
    global _db_manager
    db = get_db()
    if _os.environ.get("TABLES_CREATED") == "1":
        logger.debug("Skipping create_tables — TABLES_CREATED env var set")
        return db
    try:
        db.create_tables()
    except Exception as e:
        if not db.is_postgres and (
            "malformed" in str(e).lower() or "corrupt" in str(e).lower()
        ):
            logger.warning(f"Cached database is corrupt ({e}). Deleting and recreating.")
            db_path = Path(db.config.database.get("sqlite_path", "data/football_betting.db"))
            for suffix in ("", "-wal", "-shm"):
                p = Path(str(db_path) + suffix)
                if p.exists():
                    p.unlink()
                    logger.warning(f"Removed: {p}")
            _db_manager = None
            db = get_db()
            db.create_tables()
        else:
            raise
    logger.info("Database initialized")
    return db
