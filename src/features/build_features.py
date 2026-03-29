"""
SENTINEL — Gold Layer Feature Engineering
Joins Silver (clean.demand, clean.fuel_mix) + Weather + GDELT → analytics.features.

Strategy:
  - Pure SQL with window functions for lag/rolling features (runs inside DB)
  - Modifies CTEs to allow 7-day lookback so window functions don't truncate early in month
  - Updates holiday flags post-insert using pandas USFederalHolidayCalendar
  - Idempotent via ON CONFLICT DO UPDATE

Run: python -m src.features.build_features
"""

from sqlalchemy import text
from loguru import logger
from datetime import datetime
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.database.connection import get_engine

# The 12 BAs with full feature coverage (GKG sentiment mapping)
TARGET_BAS = [
    'ERCO', 'NYIS', 'ISNE', 'FPL', 'CISO', 'PJM',
    'MISO', 'SWPP', 'SOCO', 'TVA', 'DUK', 'BPAT'
]

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

        -- NEW: supply_margin_pct
        (d.generation_mw - d.demand_mw) / NULLIF(d.demand_mw, 0) AS supply_margin_pct,

        -- Commodity & Generation Signals (daily -> broadcast)
        gp.value                                    AS gas_price,
        op.value                                    AS oil_price,
        no.pct_outage                               AS nuclear_outage_pct,

        -- Weather features
        w.temperature_2m                            AS temperature_c,
        w.relative_humidity_2m                      AS humidity_pct,
        w.wind_speed_10m                            AS wind_speed_kmh,
        w.cloud_cover                               AS cloud_cover_pct,
        w.shortwave_radiation                       AS solar_radiation,
        GREATEST(18.0 - COALESCE(w.temperature_2m, 18), 0) AS hdd,
        GREATEST(COALESCE(w.temperature_2m, 18) - 18.0, 0) AS cdd,

        -- GDELT NLP/geopolitical signals
        COALESCE(g.us_avg_goldstein, 0)             AS sentiment_mean_24h,
        COALESCE(g.us_min_goldstein, 0)             AS sentiment_min_24h,
        COALESCE(g.us_event_count, 0)               AS event_count_24h,
        COALESCE(
            (g.us_severe_conflict::DOUBLE PRECISION / NULLIF(g.us_event_count, 0)) * 10
            + (g.oil_region_event_count::DOUBLE PRECISION / NULLIF(g.global_event_count, 0)) * 10,
            0
        )                                           AS geo_risk_index

    FROM clean.demand d

    LEFT JOIN clean.fuel_mix fm
        ON fm.period = d.period AND fm.ba_code = d.ba_code

    LEFT JOIN raw.weather_hourly w
        ON w.period = d.period AND w.ba_code = d.ba_code

    LEFT JOIN raw.gdelt_events_daily g
        ON g.event_date = d.period::DATE

    LEFT JOIN raw.eia_gas_prices gp
        ON gp.period::DATE = d.period::DATE AND gp.process_name = 'Spot Price'
    LEFT JOIN raw.eia_oil_prices op
        ON op.period::DATE = d.period::DATE
    LEFT JOIN raw.eia_nuclear_outages no
        ON no.period::DATE = d.period::DATE

    WHERE d.period >= :start_ts_lag -- Include 7 days before target start for window funcs
      AND d.period <  :end_ts
      AND d.ba_code IN ('ERCO','NYIS','ISNE','FPL','CISO','PJM','MISO','SWPP','SOCO','TVA','DUK','BPAT')
),

featured AS (
    SELECT
        b.*,

        -- NEW: gas_pct_delta_24h
        b.gas_pct - LAG(b.gas_pct, 24) OVER w AS gas_pct_delta_24h,

        -- Price change & volatility
        (b.gas_price - LAG(b.gas_price, 168) OVER w) 
            / NULLIF(LAG(b.gas_price, 168) OVER w, 0) AS gas_price_change_7d,
            
        -- NEW: gas_price_volatility_7d
        STDDEV(b.gas_price) OVER (
            PARTITION BY b.ba_code ORDER BY b.period
            ROWS BETWEEN 167 PRECEDING AND CURRENT ROW
        ) AS gas_price_volatility_7d,

        -- Demand lags & rolls
        LAG(b.demand_mw, 1) OVER w       AS demand_lag_1h,
        LAG(b.demand_mw, 24) OVER w      AS demand_lag_24h,
        LAG(b.demand_mw, 168) OVER w     AS demand_lag_168h,

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

        -- Spike detection
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
    gas_price, oil_price, gas_price_change_7d, nuclear_outage_pct,
    sentiment_mean_24h, sentiment_min_24h, event_count_24h, geo_risk_index,
    is_spike, spike_magnitude,
    supply_margin_pct, gas_pct_delta_24h, gas_price_volatility_7d
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
    gas_price, oil_price, gas_price_change_7d, nuclear_outage_pct,
    sentiment_mean_24h, sentiment_min_24h, event_count_24h, geo_risk_index,
    is_spike, spike_magnitude,
    supply_margin_pct, gas_pct_delta_24h, gas_price_volatility_7d
FROM featured
WHERE period >= :start_ts -- Filter down to exactly the target month
ON CONFLICT (period, ba_code) DO UPDATE SET
    gas_price = EXCLUDED.gas_price,
    oil_price = EXCLUDED.oil_price,
    gas_price_change_7d = EXCLUDED.gas_price_change_7d,
    nuclear_outage_pct = EXCLUDED.nuclear_outage_pct,
    supply_margin_pct = EXCLUDED.supply_margin_pct,
    gas_pct_delta_24h = EXCLUDED.gas_pct_delta_24h,
    gas_price_volatility_7d = EXCLUDED.gas_price_volatility_7d,
    demand_lag_168h = EXCLUDED.demand_lag_168h,
    demand_rolling_168h = EXCLUDED.demand_rolling_168h;
"""

def apply_holidays(engine, min_date, max_date):
    logger.info("Applying US Federal Holidays...")
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=min_date, end=max_date).to_pydatetime()
    holiday_dates = [h.strftime('%Y-%m-%d') for h in holidays]
    
    # Build a VALUES list for the IN clause (psycopg2 doesn't support ANY with :param)
    date_literals = ",".join([f"'{d}'" for d in holiday_dates])
    
    with engine.begin() as conn:
        conn.execute(text(f"""
            UPDATE analytics.features 
            SET is_holiday = TRUE 
            WHERE period::DATE IN ({date_literals})
        """))
    
    logger.info(f"Marked {len(holiday_dates)} holidays between {min_date} and {max_date}")

def build_features():
    engine = get_engine()

    with engine.connect() as conn:
        result = conn.execute(text("SELECT MIN(period), MAX(period) FROM clean.demand")).fetchone()
        min_date, max_date = result[0], result[1]

    if not min_date or not max_date:
        logger.error("❌ clean.demand is empty — run build_silver.py first!")
        return

    logger.info(f"Building features from {min_date} to {max_date}")

    from datetime import timedelta
    import calendar

    current = min_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    total_inserted = 0

    while current <= max_date:
        _, last_day = calendar.monthrange(current.year, current.month)
        month_end = current.replace(day=last_day, hour=23, minute=59, second=59)
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        
        # We need data from 7 days prior to satisfy the 168h window functions
        start_ts_lag = current - timedelta(days=7)

        logger.info(f"  Processing {current.strftime('%Y-%m')}...")

        with engine.begin() as conn:
            conn.execute(
                text(FEATURE_SQL),
                {"start_ts": current, "end_ts": next_month, "start_ts_lag": start_ts_lag}
            )

        current = next_month
        total_inserted += 1

    # Post processing: Apply Holidays
    apply_holidays(engine, min_date, max_date)

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM analytics.features")).scalar()
        ba_count = conn.execute(text("SELECT COUNT(DISTINCT ba_code) FROM analytics.features")).scalar()
        holiday_count = conn.execute(text("SELECT COUNT(*) FROM analytics.features WHERE is_holiday = TRUE")).scalar()
        
        logger.info("=" * 60)
        logger.info("✅ Gold Layer Complete!")
        logger.info(f"  Total rows:      {count:,}")
        logger.info(f"  BAs:             {ba_count}")
        logger.info(f"  Holiday rows:    {holiday_count:,}")
        logger.info("=" * 60)

if __name__ == "__main__":
    build_features()
