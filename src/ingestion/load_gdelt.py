"""
SENTINEL — GDELT Daily Events CSV Ingestion
Loads pre-aggregated GDELT CSVs (US + Global) into raw.gdelt_events_daily.
Idempotent: uses ON CONFLICT DO NOTHING.
"""

import pandas as pd
from pathlib import Path
from sqlalchemy import text
from loguru import logger

from src.config.settings import PROJECT_ROOT
from src.database.connection import get_engine


# ── Schema DDL ───────────────────────────────────────────────────────
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS raw.gdelt_events_daily (
    event_date          DATE NOT NULL,

    -- US-level signals
    us_event_count      INTEGER,
    us_avg_goldstein    DOUBLE PRECISION,
    us_min_goldstein    DOUBLE PRECISION,
    us_max_goldstein    DOUBLE PRECISION,
    us_std_goldstein    DOUBLE PRECISION,
    us_avg_tone         DOUBLE PRECISION,
    us_min_tone         DOUBLE PRECISION,
    us_total_mentions   BIGINT,
    us_total_articles   BIGINT,
    us_severe_conflict  INTEGER,
    us_moderate_conflict INTEGER,
    us_tension_count    INTEGER,
    us_cooperation_count INTEGER,
    us_very_negative    INTEGER,
    us_crisis_events    INTEGER,

    -- Global-level signals
    global_event_count      INTEGER,
    global_avg_goldstein    DOUBLE PRECISION,
    global_min_goldstein    DOUBLE PRECISION,
    global_std_goldstein    DOUBLE PRECISION,
    global_avg_tone         DOUBLE PRECISION,
    global_total_articles   BIGINT,

    -- Oil-producing region signals
    oil_region_event_count      INTEGER,
    oil_region_avg_goldstein    DOUBLE PRECISION,

    -- Global conflict tiers
    global_severe_conflict      INTEGER,
    global_moderate_conflict    INTEGER,
    global_crisis_events        INTEGER,
    global_very_negative        INTEGER,

    ingested_at         TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (event_date)
);
"""


def create_gdelt_table(engine):
    """Create the GDELT daily events table if it doesn't exist."""
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
    logger.info("✅ raw.gdelt_events_daily table ensured")


def load_us_csv(engine, csv_path: Path) -> pd.DataFrame:
    """Load and return the US GDELT CSV."""
    logger.info(f"Reading US GDELT CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  → {len(df):,} rows, {len(df.columns)} columns")

    # Parse SQLDATE as date
    df["event_date"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d").dt.date

    # Rename to match our schema
    rename_map = {
        "event_count": "us_event_count",
        "avg_goldstein": "us_avg_goldstein",
        "min_goldstein": "us_min_goldstein",
        "max_goldstein": "us_max_goldstein",
        "std_goldstein": "us_std_goldstein",
        "avg_tone": "us_avg_tone",
        "min_tone": "us_min_tone",
        "total_mentions": "us_total_mentions",
        "total_articles": "us_total_articles",
        "severe_conflict_count": "us_severe_conflict",
        "moderate_conflict_count": "us_moderate_conflict",
        "tension_count": "us_tension_count",
        "cooperation_count": "us_cooperation_count",
        "very_negative_article_count": "us_very_negative",
        "crisis_event_count": "us_crisis_events",
    }
    df = df.rename(columns=rename_map)

    us_cols = ["event_date"] + list(rename_map.values())
    return df[us_cols]


def load_global_csv(engine, csv_path: Path) -> pd.DataFrame:
    """Load and return the Global GDELT CSV."""
    logger.info(f"Reading Global GDELT CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  → {len(df):,} rows, {len(df.columns)} columns")

    # Parse SQLDATE as date
    df["event_date"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d").dt.date

    # Rename to match our schema
    rename_map = {
        "global_event_count": "global_event_count",
        "global_avg_goldstein": "global_avg_goldstein",
        "global_min_goldstein": "global_min_goldstein",
        "global_std_goldstein": "global_std_goldstein",
        "global_avg_tone": "global_avg_tone",
        "global_total_articles": "global_total_articles",
        "oil_region_event_count": "oil_region_event_count",
        "oil_region_avg_goldstein": "oil_region_avg_goldstein",
        "severe_conflict_global": "global_severe_conflict",
        "moderate_conflict_global": "global_moderate_conflict",
        "crisis_event_count": "global_crisis_events",
        "very_negative_article_count": "global_very_negative",
    }
    df = df.rename(columns=rename_map)

    global_cols = ["event_date"] + list(rename_map.values())
    return df[global_cols]


def ingest_gdelt():
    """Main ingestion: merge US + Global CSVs, upsert into DB."""
    engine = get_engine()
    create_gdelt_table(engine)

    # --- Locate CSVs ---
    us_csv = PROJECT_ROOT / "gdelt.events.csv"
    global_csv = PROJECT_ROOT / "gdelt,global.events.csv"

    if not us_csv.exists():
        logger.error(f"US GDELT CSV not found: {us_csv}")
        return
    if not global_csv.exists():
        logger.error(f"Global GDELT CSV not found: {global_csv}")
        return

    # --- Load both ---
    df_us = load_us_csv(engine, us_csv)
    df_global = load_global_csv(engine, global_csv)

    # --- Merge on event_date ---
    df = pd.merge(df_us, df_global, on="event_date", how="outer")
    logger.info(f"Merged dataset: {len(df):,} rows × {len(df.columns)} columns")

    # --- Upsert via staging table ---
    staging = "raw._staging_gdelt"
    with engine.begin() as conn:
        # Drop staging if exists (orphan cleanup)
        conn.execute(text(f'DROP TABLE IF EXISTS "{staging}"'))

    # Write to staging
    df.to_sql(
        name="_staging_gdelt",
        schema="raw",
        con=engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=500,
    )
    logger.info(f"Wrote {len(df):,} rows to staging table")

    # Atomic upsert + drop
    columns = [c for c in df.columns if c != "event_date"]
    col_list = ", ".join(f'"{c}"' for c in ["event_date"] + columns)

    upsert_sql = f"""
    INSERT INTO raw.gdelt_events_daily ({col_list})
    SELECT {col_list}
    FROM raw._staging_gdelt
    ON CONFLICT (event_date) DO NOTHING;

    DROP TABLE IF EXISTS raw._staging_gdelt;
    """

    with engine.begin() as conn:
        result = conn.execute(text(upsert_sql))
        logger.info(f"✅ Upserted GDELT data into raw.gdelt_events_daily")

    # --- Verify ---
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM raw.gdelt_events_daily")
        ).scalar()
        date_range = conn.execute(
            text("SELECT MIN(event_date), MAX(event_date) FROM raw.gdelt_events_daily")
        ).fetchone()
        logger.info(f"✅ Verification: {count:,} rows, {date_range[0]} → {date_range[1]}")


if __name__ == "__main__":
    ingest_gdelt()
