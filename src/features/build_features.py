"""
SENTINEL — Gold Layer Feature Engineering
Joins Silver (clean.demand, clean.fuel_mix) + Weather + GDELT → analytics.features.

Strategy:
  - Pure SQL with window functions for lag/rolling features (runs inside DB, no pandas)
  - GDELT is daily → broadcast to all hours of that day via DATE() join
  - Weather is hourly × per-BA → direct join on (period, ba_code)
  - Processes one month at a time to manage transaction size
  - Idempotent via ON CONFLICT DO NOTHING

Run: python -m src.features.build_features
"""

from sqlalchemy import text
from loguru import logger
from datetime import datetime

from src.database.connection import get_engine


# ── The core feature query ───────────────────────────────────────────
# Uses CTEs: 1) base join  2) window functions  3) final insert
FEATURE_SQL = """
WITH base AS (
    SELECT
        d.period,
        d.ba_code,

        -- Target
        d.demand_mw,

        -- Temporal features
        EXTRACT(HOUR FROM d.period)::SMALLINT       AS hour_of_day,
        EXTRACT(DOW FROM d.period)::SMALLINT        AS day_of_week,
        EXTRACT(MONTH FROM d.period)::SMALLINT      AS month,
        EXTRACT(DOW FROM d.period) IN (0, 6)        AS is_weekend,
        FALSE                                        AS is_holiday,

        -- Supply features from clean tables
        d.generation_mw,
        d.supply_demand_gap,
        d.interchange_mw,
        COALESCE(fm.gas_pct, 0)                     AS gas_pct,
        COALESCE(fm.renewable_pct, 0)               AS renewable_pct,

        -- Weather features
        w.temperature_2m                            AS temperature_c,
        w.relative_humidity_2m                      AS humidity_pct,
        w.wind_speed_10m                            AS wind_speed_kmh,
        w.cloud_cover                               AS cloud_cover_pct,
        w.shortwave_radiation                       AS solar_radiation,
        GREATEST(18.0 - COALESCE(w.temperature_2m, 18), 0) AS hdd,
        GREATEST(COALESCE(w.temperature_2m, 18) - 18.0, 0) AS cdd,

        -- GDELT NLP/geopolitical signals (daily → broadcast to hours)
        COALESCE(g.us_avg_goldstein, 0)             AS sentiment_mean_24h,
        COALESCE(g.us_min_goldstein, 0)             AS sentiment_min_24h,
        COALESCE(g.us_event_count, 0)               AS event_count_24h,
        -- Geo risk index: scaled severe conflict + oil tension
        COALESCE(
            (g.us_severe_conflict::DOUBLE PRECISION / NULLIF(g.us_event_count, 0)) * 10
            + (g.oil_region_event_count::DOUBLE PRECISION / NULLIF(g.global_event_count, 0)) * 10,
            0
        )                                           AS geo_risk_index,

        -- Price features (daily → broadcast to hours)
        gp.value                                    AS gas_price,

        -- Nuclear outage features (daily → broadcast to hours)
        no.pct_outage                               AS nuclear_outage_pct

    FROM clean.demand d

    -- Fuel mix: same (period, ba_code) grain
    LEFT JOIN clean.fuel_mix fm
        ON fm.period = d.period AND fm.ba_code = d.ba_code

    -- Weather: same (period, ba_code) grain
    LEFT JOIN raw.weather_hourly w
        ON w.period = d.period AND w.ba_code = d.ba_code

    -- GDELT: daily → join on date portion
    LEFT JOIN raw.gdelt_events_daily g
        ON g.event_date = d.period::DATE

    -- Gas prices: daily Henry Hub spot → join on date
    LEFT JOIN raw.eia_gas_prices gp
        ON gp.period = d.period::DATE
       AND gp.series_name = 'Henry Hub Natural Gas Spot Price (Dollars per Million Btu)'

    -- Nuclear outages: daily national → join on date
    LEFT JOIN raw.eia_nuclear_outages no
        ON no.period = d.period::DATE

    WHERE d.period >= :start_ts
      AND d.period <  :end_ts
),

-- Add lag / rolling features via window functions
featured AS (
    SELECT
        b.*,

        -- Demand lags
        LAG(b.demand_mw, 1) OVER w       AS demand_lag_1h,
        LAG(b.demand_mw, 24) OVER w      AS demand_lag_24h,
        LAG(b.demand_mw, 168) OVER w     AS demand_lag_168h,

        -- Rolling stats
        AVG(b.demand_mw) OVER (
            PARTITION BY b.ba_code ORDER BY b.period
            ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
        )                                  AS demand_rolling_24h,
        AVG(b.demand_mw) OVER (
            PARTITION BY b.ba_code ORDER BY b.period
            ROWS BETWEEN 167 PRECEDING AND CURRENT ROW
        )                                  AS demand_rolling_168h,
        STDDEV(b.demand_mw) OVER (
            PARTITION BY b.ba_code ORDER BY b.period
            ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
        )                                  AS demand_std_24h,

        -- Gas price 7-day % change (daily, broadcast across hours)
        CASE WHEN LAG(b.gas_price, 168) OVER w > 0 THEN
            (b.gas_price - LAG(b.gas_price, 168) OVER w)
            / LAG(b.gas_price, 168) OVER w * 100.0
        ELSE NULL END                      AS gas_price_change_7d,

        -- Spike detection (demand deviates >2σ from 24h rolling mean)
        CASE WHEN ABS(b.demand_mw - AVG(b.demand_mw) OVER (
            PARTITION BY b.ba_code ORDER BY b.period
            ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
        )) > 2 * STDDEV(b.demand_mw) OVER (
            PARTITION BY b.ba_code ORDER BY b.period
            ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
        ) THEN TRUE ELSE FALSE END         AS is_spike,

        ABS(b.demand_mw - AVG(b.demand_mw) OVER (
            PARTITION BY b.ba_code ORDER BY b.period
            ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
        )) / NULLIF(STDDEV(b.demand_mw) OVER (
            PARTITION BY b.ba_code ORDER BY b.period
            ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
        ), 0)                              AS spike_magnitude

    FROM base b
    WINDOW w AS (PARTITION BY b.ba_code ORDER BY b.period)
)

INSERT INTO analytics.features (
    period, ba_code,
    demand_mw,
    hour_of_day, day_of_week, month, is_weekend, is_holiday,
    demand_lag_1h, demand_lag_24h, demand_lag_168h,
    demand_rolling_24h, demand_rolling_168h, demand_std_24h,
    temperature_c, humidity_pct, wind_speed_kmh, cloud_cover_pct,
    solar_radiation, hdd, cdd,
    generation_mw, supply_demand_gap, gas_pct, renewable_pct, interchange_mw,
    gas_price, gas_price_change_7d, nuclear_outage_pct,
    sentiment_mean_24h, sentiment_min_24h, event_count_24h, geo_risk_index,
    is_spike, spike_magnitude
)
SELECT
    period, ba_code,
    demand_mw,
    hour_of_day, day_of_week, month, is_weekend, is_holiday,
    demand_lag_1h, demand_lag_24h, demand_lag_168h,
    demand_rolling_24h, demand_rolling_168h, demand_std_24h,
    temperature_c, humidity_pct, wind_speed_kmh, cloud_cover_pct,
    solar_radiation, hdd, cdd,
    generation_mw, supply_demand_gap, gas_pct, renewable_pct, interchange_mw,
    gas_price, gas_price_change_7d, nuclear_outage_pct,
    sentiment_mean_24h, sentiment_min_24h, event_count_24h, geo_risk_index,
    is_spike, spike_magnitude
FROM featured
ON CONFLICT (period, ba_code) DO NOTHING;
"""


def build_features():
    """Build the Gold analytics.features table month by month."""
    engine = get_engine()

    # Determine date range from clean.demand
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT MIN(period), MAX(period) FROM clean.demand"
        )).fetchone()
        min_date, max_date = result[0], result[1]

    if not min_date or not max_date:
        logger.error("❌ clean.demand is empty — run build_silver.py first!")
        return

    logger.info(f"Building features from {min_date} to {max_date}")

    # Process month by month to manage memory and transaction size
    from datetime import timedelta
    import calendar

    current = min_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    total_inserted = 0

    while current <= max_date:
        # Calculate month end
        _, last_day = calendar.monthrange(current.year, current.month)
        month_end = current.replace(day=last_day, hour=23, minute=59, second=59)
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )

        logger.info(f"  Processing {current.strftime('%Y-%m')}...")

        with engine.begin() as conn:
            conn.execute(
                text(FEATURE_SQL),
                {"start_ts": current, "end_ts": next_month}
            )

        current = next_month
        total_inserted += 1

    # ── Verification ─────────────────────────────────────────────────
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM analytics.features")
        ).scalar()
        ba_count = conn.execute(
            text("SELECT COUNT(DISTINCT ba_code) FROM analytics.features")
        ).scalar()
        date_range = conn.execute(
            text("SELECT MIN(period), MAX(period) FROM analytics.features")
        ).fetchone()
        null_pcts = conn.execute(text("""
            SELECT
                ROUND(100.0 * SUM(CASE WHEN demand_mw IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS demand_null_pct,
                ROUND(100.0 * SUM(CASE WHEN temperature_c IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS weather_null_pct,
                ROUND(100.0 * SUM(CASE WHEN geo_risk_index IS NULL OR geo_risk_index = 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS gdelt_null_pct,
                ROUND(100.0 * SUM(CASE WHEN is_spike THEN 1 ELSE 0 END) / COUNT(*), 2) AS spike_pct
            FROM analytics.features
        """)).fetchone()

        logger.info("=" * 60)
        logger.info("✅ Gold Layer Complete!")
        logger.info(f"  Total rows:      {count:,}")
        logger.info(f"  BAs:             {ba_count}")
        logger.info(f"  Date range:      {date_range[0]} → {date_range[1]}")
        logger.info(f"  demand NULL%:    {null_pcts[0]}%")
        logger.info(f"  weather NULL%:   {null_pcts[1]}%")
        logger.info(f"  gdelt zero/NULL%:{null_pcts[2]}%")
        logger.info(f"  spike%:          {null_pcts[3]}%")
        logger.info("=" * 60)


if __name__ == "__main__":
    build_features()
