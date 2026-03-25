"""
SENTINEL Database Connection Module
Provides SQLAlchemy engine, session management, and connection utilities.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from loguru import logger

from src.config import get_settings


_engine = None
_session_factory = None


def get_connection_string() -> str:
    """Get the database connection string from settings."""
    settings = get_settings()
    return settings.database_url


def get_engine():
    """Get or create the SQLAlchemy engine (singleton)."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False,
        )
        logger.info(f"Database engine created: {settings.db_host}:{settings.db_port}/{settings.db_name}")
    return _engine


def get_session_factory():
    """Get or create the session factory (singleton)."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = sessionmaker(bind=engine)
    return _session_factory


def get_session() -> Session:
    """Create a new database session."""
    factory = get_session_factory()
    return factory()


@contextmanager
def session_scope():
    """Context manager for database sessions with auto-commit/rollback."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def test_connection() -> bool:
    """Test the database connection and TimescaleDB extension."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Test basic connectivity
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

            # Test TimescaleDB extension
            result = conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
            )
            version = result.scalar()
            if version:
                logger.info(f"TimescaleDB version: {version}")
            else:
                logger.warning("TimescaleDB extension not found!")
                return False

            # Check schemas exist
            result = conn.execute(
                text("SELECT schema_name FROM information_schema.schemata WHERE schema_name IN ('raw', 'clean', 'analytics')")
            )
            schemas = [row[0] for row in result]
            logger.info(f"Available schemas: {schemas}")

            return True

    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def get_table_counts() -> dict:
    """Get row counts for all SENTINEL tables (for monitoring)."""
    tables = [
        "raw.eia_region_data",
        "raw.eia_fuel_type_data",
        "raw.eia_interchange_data",
        "raw.eia_gas_prices",
        "raw.eia_oil_prices",
        "raw.eia_nuclear_outages",
        "raw.weather_hourly",
        "clean.demand",
        "clean.fuel_mix",
        "analytics.features",
        "analytics.predictions",
    ]

    counts = {}
    try:
        engine = get_engine()
        with engine.connect() as conn:
            for table in tables:
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    counts[table] = result.scalar()
                except Exception:
                    counts[table] = -1  # table doesn't exist yet
    except Exception as e:
        logger.error(f"Failed to get table counts: {e}")

    return counts


if __name__ == "__main__":
    # Quick test
    if test_connection():
        print("✅ Database connection successful!")
        counts = get_table_counts()
        for table, count in counts.items():
            print(f"  {table}: {count:,} rows")
    else:
        print("❌ Database connection failed!")
