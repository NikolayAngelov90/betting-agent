"""Database manager for Football Betting Agent."""

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

    def _create_engine(self):
        db_config = self.config.database
        db_type = db_config.get("type", "sqlite")

        if db_type == "sqlite":
            db_path = Path(db_config.get("sqlite_path", "data/football_betting.db"))
            db_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{db_path}"

            engine = create_engine(
                url, echo=False,
                pool_pre_ping=True,
                connect_args={"check_same_thread": False},
            )

            # Enable WAL mode and other SQLite optimizations on every connect
            @event.listens_for(engine, "connect")
            def _set_sqlite_pragmas(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA busy_timeout=5000")
                cursor.close()

            return engine

        elif db_type == "postgresql":
            host = db_config.get("host", "localhost")
            port = db_config.get("port", 5432)
            name = db_config.get("name", "football_betting")
            user = db_config.get("user", "")
            password = db_config.get("password", "")
            url = f"postgresql://{user}:{password}@{host}:{port}/{name}"

            return create_engine(
                url, echo=False,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
            )
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def create_tables(self):
        """Create all tables defined in models, and add any missing columns."""
        Base.metadata.create_all(self.engine)
        self._migrate_missing_columns()
        logger.info("Database tables created successfully")

    def _migrate_missing_columns(self):
        """Auto-add columns that exist in models but not in the DB (simple migration).

        Only handles adding nullable columns — safe for SQLite.
        """
        inspector = inspect(self.engine)
        for table_name, table in Base.metadata.tables.items():
            if table_name not in inspector.get_table_names():
                continue

            existing_cols = {c["name"] for c in inspector.get_columns(table_name)}
            for col in table.columns:
                if col.name not in existing_cols:
                    col_type = col.type.compile(self.engine.dialect)
                    sql = f"ALTER TABLE {table_name} ADD COLUMN {col.name} {col_type}"
                    try:
                        with self.engine.begin() as conn:
                            conn.execute(text(sql))
                        logger.info(f"Migration: added column {table_name}.{col.name} ({col_type})")
                    except Exception as e:
                        logger.debug(f"Column {table_name}.{col.name} migration skipped: {e}")

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


# Global database manager instance
_db_manager = None


def get_db() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def init_db():
    """Initialize database: create engine and all tables."""
    db = get_db()
    db.create_tables()
    logger.info("Database initialized")
    return db
