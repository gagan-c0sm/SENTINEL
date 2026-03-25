"""
Script to apply the NLP database schema to the running TimescaleDB instance.
This creates the tables needed to link unstructured news text to structured BA data.
"""

from loguru import logger
from sqlalchemy import text
from src.database.connection import get_engine

NLP_SCHEMA_SQL = """
-- =============================================================================
-- NLP SCHEMA SETUP (ba_region_mapping, news_events, nlp_features_hourly)
-- =============================================================================

-- Table 1: BA Region Keyword Mapping
CREATE TABLE IF NOT EXISTS clean.ba_region_mapping (
    ba_code       VARCHAR(10) NOT NULL PRIMARY KEY,
    ba_name       TEXT NOT NULL,
    state_codes   TEXT[] NOT NULL,
    region_names  TEXT[] NOT NULL,
    fuel_profile  JSONB
);

CREATE INDEX IF NOT EXISTS idx_region_names_gin ON clean.ba_region_mapping USING GIN (region_names);
CREATE INDEX IF NOT EXISTS idx_fuel_profile ON clean.ba_region_mapping USING GIN (fuel_profile jsonb_path_ops);

-- Table 2: Normalized News Events
CREATE TABLE IF NOT EXISTS clean.news_events (
    event_time        TIMESTAMPTZ NOT NULL,  -- normalized to hour: 2025-03-15T15:00Z
    event_id          SERIAL,
    published_time    TIMESTAMPTZ NOT NULL,  -- original article timestamp
    ba_codes          TEXT[],                -- resolved BA codes
    fuel_types        TEXT[],                -- mentioned fuels
    event_type        TEXT,                  -- SUPPLY_DISRUPTION / POLICY / DEMAND / PRICE
    sentiment_score   DOUBLE PRECISION,      -- -1.0 to +1.0
    risk_magnitude    DOUBLE PRECISION,      -- 0.0 to 1.0
    source            TEXT,                  -- 'reuters', 'eia_rss', 'gdelt'
    headline          TEXT,
    raw_text          TEXT,
    content_hash      TEXT NOT NULL,         -- SHA-256 of headline+source for dedup
    PRIMARY KEY (event_time, event_id)       -- composite PK for hypertable partitioning
);

-- Convert to Hypertable
SELECT create_hypertable('clean.news_events', 'event_time', 
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE);

CREATE UNIQUE INDEX IF NOT EXISTS idx_dedup ON clean.news_events (content_hash, event_time);
CREATE INDEX IF NOT EXISTS idx_ba_time ON clean.news_events USING GIN (ba_codes) WITH (fastupdate = off);

-- Table 3: Pre-computed rolling NLP features per BA per hour
-- We use a regular table to store the pre-computed features and update via script
-- since continuous aggregates over JOINs get extremely complex in Timescale.
CREATE TABLE IF NOT EXISTS analytics.nlp_features_hourly (
    period                  TIMESTAMPTZ NOT NULL,
    ba_code                 VARCHAR(10) NOT NULL,
    news_sentiment_24h      DOUBLE PRECISION DEFAULT 0,
    news_max_risk_24h       DOUBLE PRECISION DEFAULT 0,
    news_event_count_24h    INTEGER DEFAULT 0,
    supply_disruption_24h   BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (period, ba_code)
);

SELECT create_hypertable('analytics.nlp_features_hourly', 'period', 
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE);
"""

def setup_nlp_tables():
    logger.info("Applying NLP schema to database...")
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(NLP_SCHEMA_SQL))
    logger.info("✅ NLP tables created successfully.")

if __name__ == "__main__":
    setup_nlp_tables()
