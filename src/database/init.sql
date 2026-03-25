-- =============================================================================
-- SENTINEL Database Initialization Script
-- Runs automatically when TimescaleDB container first starts
-- =============================================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- SCHEMA: raw (Bronze layer — unprocessed API data)
-- =============================================================================
CREATE SCHEMA IF NOT EXISTS raw;

-- =============================================================================
-- SCHEMA: clean (Silver layer — validated, deduplicated)
-- =============================================================================
CREATE SCHEMA IF NOT EXISTS clean;

-- =============================================================================
-- SCHEMA: analytics (Gold layer — feature-engineered, model-ready)
-- =============================================================================
CREATE SCHEMA IF NOT EXISTS analytics;

-- =============================================================================
-- RAW TABLES
-- =============================================================================

-- Table 1: EIA Hourly Demand / Forecast / Generation / Interchange
CREATE TABLE IF NOT EXISTS raw.eia_region_data (
    id              BIGSERIAL,
    period          TIMESTAMPTZ NOT NULL,       -- hourly timestamp (UTC)
    respondent      VARCHAR(10) NOT NULL,       -- BA code (e.g., 'ERCO')
    respondent_name VARCHAR(200),               -- full name
    type            VARCHAR(5) NOT NULL,        -- D, DF, NG, TI
    type_name       VARCHAR(100),               -- human-readable type name
    value           DOUBLE PRECISION,           -- MWh
    units           VARCHAR(20) DEFAULT 'megawatthours',
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),  -- when we pulled this data
    PRIMARY KEY (period, respondent, type)
);

-- Convert to TimescaleDB hypertable for efficient time-series queries
SELECT create_hypertable(
    'raw.eia_region_data',
    'period',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 month'
);

-- Table 2: EIA Hourly Generation by Fuel Type
CREATE TABLE IF NOT EXISTS raw.eia_fuel_type_data (
    id              BIGSERIAL,
    period          TIMESTAMPTZ NOT NULL,
    respondent      VARCHAR(10) NOT NULL,       -- BA code
    respondent_name VARCHAR(200),
    fueltype        VARCHAR(10) NOT NULL,       -- COL, NG, NUC, SUN, etc.
    type_name       VARCHAR(100),
    value           DOUBLE PRECISION,           -- MWh
    units           VARCHAR(20) DEFAULT 'megawatthours',
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (period, respondent, fueltype)
);

SELECT create_hypertable(
    'raw.eia_fuel_type_data',
    'period',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 month'
);

-- Table 3: EIA Hourly Interchange between BAs
CREATE TABLE IF NOT EXISTS raw.eia_interchange_data (
    id              BIGSERIAL,
    period          TIMESTAMPTZ NOT NULL,
    respondent      VARCHAR(10) NOT NULL,       -- source BA
    respondent_name VARCHAR(200),
    fromba          VARCHAR(10) NOT NULL,       -- from BA code
    fromba_name     VARCHAR(200),
    toba            VARCHAR(10) NOT NULL,       -- to BA code
    toba_name       VARCHAR(200),
    value           DOUBLE PRECISION,           -- MWh
    units           VARCHAR(20) DEFAULT 'megawatthours',
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (period, fromba, toba)
);

SELECT create_hypertable(
    'raw.eia_interchange_data',
    'period',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 month'
);

-- Table 4: Natural Gas Prices
CREATE TABLE IF NOT EXISTS raw.eia_gas_prices (
    id              BIGSERIAL,
    period          DATE NOT NULL,
    series_name     VARCHAR(200),
    area_name       VARCHAR(200),
    product_name    VARCHAR(200),
    process_name    VARCHAR(200),
    value           DOUBLE PRECISION,           -- $/MMBtu
    units           VARCHAR(50),
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (period, series_name)
);

-- Table 5: Petroleum (Oil) Prices
CREATE TABLE IF NOT EXISTS raw.eia_oil_prices (
    id              BIGSERIAL,
    period          DATE NOT NULL,
    series_name     VARCHAR(200),               -- e.g., 'RWTC' (WTI), 'RBRTE' (Brent)
    product_name    VARCHAR(200),
    area_name       VARCHAR(200),
    value           DOUBLE PRECISION,           -- $/barrel
    units           VARCHAR(50),
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (period, series_name)
);

-- Table 6: Nuclear Outages
CREATE TABLE IF NOT EXISTS raw.eia_nuclear_outages (
    id              BIGSERIAL,
    period          DATE NOT NULL,
    capacity        DOUBLE PRECISION,           -- total capacity MW
    outage          DOUBLE PRECISION,           -- outage capacity MW
    pct_outage      DOUBLE PRECISION,           -- percentage
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (period)
);

-- Weather data from Open-Meteo
CREATE TABLE IF NOT EXISTS raw.weather_hourly (
    id              BIGSERIAL,
    period          TIMESTAMPTZ NOT NULL,
    ba_code         VARCHAR(10) NOT NULL,       -- matched to BA
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    temperature_2m  DOUBLE PRECISION,           -- °C
    relative_humidity_2m DOUBLE PRECISION,      -- %
    wind_speed_10m  DOUBLE PRECISION,           -- km/h
    wind_direction_10m DOUBLE PRECISION,        -- degrees
    cloud_cover     DOUBLE PRECISION,           -- %
    shortwave_radiation DOUBLE PRECISION,       -- W/m²
    precipitation   DOUBLE PRECISION,           -- mm
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (period, ba_code)
);

SELECT create_hypertable(
    'raw.weather_hourly',
    'period',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 month'
);

-- =============================================================================
-- CLEAN TABLES (Silver layer)
-- =============================================================================

-- Clean demand data — validated, gaps filled, deduplicated
CREATE TABLE IF NOT EXISTS clean.demand (
    period          TIMESTAMPTZ NOT NULL,
    ba_code         VARCHAR(10) NOT NULL,
    demand_mw       DOUBLE PRECISION,
    forecast_mw     DOUBLE PRECISION,
    generation_mw   DOUBLE PRECISION,
    interchange_mw  DOUBLE PRECISION,
    demand_forecast_gap DOUBLE PRECISION,       -- demand - forecast (surprise)
    supply_demand_gap   DOUBLE PRECISION,       -- generation - demand (surplus/deficit)
    is_interpolated BOOLEAN DEFAULT FALSE,      -- flag if gap-filled
    PRIMARY KEY (period, ba_code)
);

SELECT create_hypertable(
    'clean.demand',
    'period',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 month'
);

-- Clean fuel mix per BA
CREATE TABLE IF NOT EXISTS clean.fuel_mix (
    period          TIMESTAMPTZ NOT NULL,
    ba_code         VARCHAR(10) NOT NULL,
    coal_mw         DOUBLE PRECISION DEFAULT 0,
    gas_mw          DOUBLE PRECISION DEFAULT 0,
    nuclear_mw      DOUBLE PRECISION DEFAULT 0,
    oil_mw          DOUBLE PRECISION DEFAULT 0,
    solar_mw        DOUBLE PRECISION DEFAULT 0,
    hydro_mw        DOUBLE PRECISION DEFAULT 0,
    wind_mw         DOUBLE PRECISION DEFAULT 0,
    other_mw        DOUBLE PRECISION DEFAULT 0,
    total_mw        DOUBLE PRECISION DEFAULT 0,
    gas_pct         DOUBLE PRECISION DEFAULT 0, -- % of total from gas
    renewable_pct   DOUBLE PRECISION DEFAULT 0, -- % from solar+wind+hydro
    PRIMARY KEY (period, ba_code)
);

SELECT create_hypertable(
    'clean.fuel_mix',
    'period',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 month'
);

-- =============================================================================
-- ANALYTICS TABLES (Gold layer — model-ready features)
-- =============================================================================

CREATE TABLE IF NOT EXISTS analytics.features (
    period              TIMESTAMPTZ NOT NULL,
    ba_code             VARCHAR(10) NOT NULL,

    -- Target
    demand_mw           DOUBLE PRECISION,

    -- Temporal features
    hour_of_day         SMALLINT,
    day_of_week         SMALLINT,
    month               SMALLINT,
    is_weekend          BOOLEAN,
    is_holiday          BOOLEAN DEFAULT FALSE,

    -- Demand features
    demand_lag_1h       DOUBLE PRECISION,
    demand_lag_24h      DOUBLE PRECISION,
    demand_lag_168h     DOUBLE PRECISION,       -- 1 week ago
    demand_rolling_24h  DOUBLE PRECISION,       -- 24h rolling mean
    demand_rolling_168h DOUBLE PRECISION,       -- 7-day rolling mean
    demand_std_24h      DOUBLE PRECISION,       -- 24h rolling std

    -- Weather features
    temperature_c       DOUBLE PRECISION,
    humidity_pct        DOUBLE PRECISION,
    wind_speed_kmh      DOUBLE PRECISION,
    cloud_cover_pct     DOUBLE PRECISION,
    solar_radiation     DOUBLE PRECISION,
    hdd                 DOUBLE PRECISION,       -- heating degree days
    cdd                 DOUBLE PRECISION,       -- cooling degree days

    -- Supply features
    generation_mw       DOUBLE PRECISION,
    supply_demand_gap   DOUBLE PRECISION,
    gas_pct             DOUBLE PRECISION,
    renewable_pct       DOUBLE PRECISION,
    interchange_mw      DOUBLE PRECISION,

    -- Price features
    gas_price           DOUBLE PRECISION,       -- $/MMBtu Henry Hub
    oil_price           DOUBLE PRECISION,       -- $/bbl WTI
    gas_price_change_7d DOUBLE PRECISION,       -- 7-day price change %

    -- Nuclear features  
    nuclear_outage_pct  DOUBLE PRECISION,       -- national nuclear outage %

    -- NLP features (filled by NLP pipeline)
    sentiment_mean_24h  DOUBLE PRECISION DEFAULT 0,
    sentiment_min_24h   DOUBLE PRECISION DEFAULT 0,
    event_count_24h     INTEGER DEFAULT 0,
    geo_risk_index      DOUBLE PRECISION DEFAULT 0,

    -- Labels (for classification)
    is_spike            BOOLEAN DEFAULT FALSE,  -- demand spike flag
    spike_magnitude     DOUBLE PRECISION DEFAULT 0,

    PRIMARY KEY (period, ba_code)
);

SELECT create_hypertable(
    'analytics.features',
    'period',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 month'
);

-- =============================================================================
-- PREDICTION LOGS (for future blockchain hashing)
-- =============================================================================

CREATE TABLE IF NOT EXISTS analytics.predictions (
    id              BIGSERIAL PRIMARY KEY,
    predicted_at    TIMESTAMPTZ DEFAULT NOW(),
    target_period   TIMESTAMPTZ NOT NULL,
    ba_code         VARCHAR(10) NOT NULL,
    model_name      VARCHAR(50) NOT NULL,
    predicted_demand DOUBLE PRECISION,
    actual_demand   DOUBLE PRECISION,
    confidence_lower DOUBLE PRECISION,
    confidence_upper DOUBLE PRECISION,
    mape            DOUBLE PRECISION,
    prediction_hash VARCHAR(66),               -- for blockchain accountability
    tx_hash         VARCHAR(66)                -- blockchain transaction hash
);

-- =============================================================================
-- INDEXES for common query patterns
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_region_data_ba_type
    ON raw.eia_region_data (respondent, type, period DESC);

CREATE INDEX IF NOT EXISTS idx_fuel_type_ba
    ON raw.eia_fuel_type_data (respondent, fueltype, period DESC);

CREATE INDEX IF NOT EXISTS idx_interchange_flow
    ON raw.eia_interchange_data (fromba, toba, period DESC);

CREATE INDEX IF NOT EXISTS idx_clean_demand_ba
    ON clean.demand (ba_code, period DESC);

CREATE INDEX IF NOT EXISTS idx_features_ba
    ON analytics.features (ba_code, period DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_target
    ON analytics.predictions (ba_code, target_period DESC);

-- =============================================================================
-- CONTINUOUS AGGREGATES (materialized views for fast dashboarding)
-- =============================================================================

-- Daily demand summary per BA
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.daily_demand_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', period) AS day,
    ba_code,
    AVG(demand_mw) AS avg_demand,
    MAX(demand_mw) AS peak_demand,
    MIN(demand_mw) AS min_demand,
    STDDEV(demand_mw) AS stddev_demand
FROM clean.demand
GROUP BY day, ba_code
WITH NO DATA;

-- Refresh policy: update daily aggregates every hour
SELECT add_continuous_aggregate_policy('analytics.daily_demand_summary',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Enable compression on old data (older than 30 days)
ALTER TABLE raw.eia_region_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'respondent, type'
);

SELECT add_compression_policy('raw.eia_region_data',
    compress_after => INTERVAL '30 days',
    if_not_exists => TRUE
);

ALTER TABLE raw.eia_fuel_type_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'respondent, fueltype'
);

SELECT add_compression_policy('raw.eia_fuel_type_data',
    compress_after => INTERVAL '30 days',
    if_not_exists => TRUE
);

RAISE NOTICE 'SENTINEL database initialized successfully!';
