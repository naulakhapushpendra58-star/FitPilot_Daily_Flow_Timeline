import os
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "fitpilot.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def _normalize_preferred_activities(value):
    if value is None:
        return "Walking"
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return ", ".join(items) if items else "Walking"
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(items) if items else "Walking"
    return str(value).strip() or "Walking"


def ensure_preferred_activities_column():
    """Safely add the multi-activity column to the existing SQLite table."""
    if not DATABASE_URL.startswith("sqlite"):
        return

    with engine.begin() as connection:
        inspector = inspect(connection)
        if not inspector.has_table("preferences"):
            return

        columns = [column["name"] for column in inspector.get_columns("preferences")]
        if "preferred_activities" in columns:
            return

        connection.execute(text("ALTER TABLE preferences ADD COLUMN preferred_activities TEXT"))

        rows = connection.execute(
            text("SELECT id, preferred_activity FROM preferences WHERE preferred_activity IS NOT NULL")
        ).fetchall()

        for row_id, preferred_activity in rows:
            migrated_value = _normalize_preferred_activities(preferred_activity)
            connection.execute(
                text("UPDATE preferences SET preferred_activities = :value WHERE id = :id"),
                {"value": migrated_value, "id": row_id},
            )

        connection.execute(
            text(
                "UPDATE preferences SET preferred_activities = :default WHERE preferred_activities IS NULL OR preferred_activities = ''"
            ),
            {"default": "Walking"},
        )


def ensure_schedule_items_status_column():
    """Add status to legacy schedule_items rows without dropping or recreating data."""
    if not DATABASE_URL.startswith("sqlite"):
        return

    with engine.begin() as connection:
        inspector = inspect(connection)
        if not inspector.has_table("schedule_items"):
            return

        columns = [column["name"] for column in inspector.get_columns("schedule_items")]
        if "status" in columns:
            return

        connection.execute(text("ALTER TABLE schedule_items ADD COLUMN status VARCHAR(20) DEFAULT 'planned'"))
        connection.execute(
            text(
                "UPDATE schedule_items SET status = CASE WHEN completed = 1 THEN 'completed' ELSE 'planned' END "
                "WHERE status IS NULL OR status = ''"
            )
        )


def initialize_database():
    ensure_preferred_activities_column()
    ensure_schedule_items_status_column()
    Base.metadata.create_all(bind=engine)


initialize_database()
