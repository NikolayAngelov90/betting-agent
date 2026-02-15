"""Database manager for Football Betting Agent."""

from pathlib import Path
from contextlib import contextmanager

from sqlalchemy import create_engine
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
        elif db_type == "postgresql":
            host = db_config.get("host", "localhost")
            port = db_config.get("port", 5432)
            name = db_config.get("name", "football_betting")
            user = db_config.get("user", "")
            password = db_config.get("password", "")
            url = f"postgresql://{user}:{password}@{host}:{port}/{name}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

        logger.info(f"Creating database engine: {db_type}")
        return create_engine(url, echo=False)

    def create_tables(self):
        """Create all tables defined in models."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created successfully")

    def drop_tables(self):
        """Drop all tables. Use with caution."""
        Base.metadata.drop_all(self.engine)
        logger.warning("All database tables dropped")

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
