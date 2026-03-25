"""
SENTINEL — Robust Weather Data Backfill Script (Hardened)
Pulls 5 years of historical hourly weather data for all 25 BAs.
Idempotent: safe to re-run via staging table + ON CONFLICT.
"""
import sys
import os
from datetime import datetime
from loguru import logger
from tqdm import tqdm
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.ingestion.weather_client import WeatherClient, BA_COORDINATES
from src.config.settings import get_settings, BACKFILL_START_DATE, BACKFILL_END_DATE


def insert_weather_idempotent(df: pd.DataFrame, engine, ba_code: str):
    """Idempotent bulk insert via staging table + ON CONFLICT DO NOTHING."""
    if df.empty:
        return 0

    df["period"] = pd.to_datetime(df["period"])
    temp_table = f"temp_weather_{ba_code.lower()}"

    # Stage into temp table
    df.to_sql(
        temp_table,
        engine,
        schema="raw",
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=5000,
    )

    # Atomic merge + drop
    cols = list(df.columns)
    col_str = ", ".join([f'"{c}"' for c in cols])

    with engine.begin() as conn:
        result = conn.execute(text(f"""
            INSERT INTO raw.weather_hourly ({col_str})
            SELECT {col_str} FROM raw."{temp_table}"
            ON CONFLICT (period, ba_code) DO NOTHING;
        """))
        inserted = result.rowcount if result.rowcount >= 0 else len(df)
        conn.execute(text(f'DROP TABLE IF EXISTS raw."{temp_table}"'))

    return inserted


def backfill_weather():
    start_time = datetime.now()
    settings = get_settings()
    engine = create_engine(settings.database_url, pool_size=2, max_overflow=0)
    client = WeatherClient()

    ba_list = list(BA_COORDINATES.keys())
    total_inserted = 0
    failed_bas = []

    logger.info("=" * 60)
    logger.info("Starting Weather Data Backfill (Open-Meteo)")
    logger.info(f"Regions: {len(ba_list)}")
    logger.info(f"Targeting: {BACKFILL_START_DATE[:10]} to {BACKFILL_END_DATE[:10]}")
    logger.info("=" * 60)

    for i, ba_code in enumerate(ba_list, 1):
        logger.info(f"[{i}/{len(ba_list)}] Fetching {ba_code}...")

        try:
            df = client.fetch_historical_weather(
                ba_code,
                BACKFILL_START_DATE,
                BACKFILL_END_DATE
            )

            if not df.empty:
                rows_inserted = insert_weather_idempotent(df, engine, ba_code)
                total_inserted += rows_inserted
                logger.success(f"  [OK] {ba_code}: {rows_inserted:,} rows merged")
            else:
                logger.warning(f"  [EMPTY] {ba_code}: No data returned")

        except Exception as e:
            logger.error(f"  [FAIL] {ba_code}: {e}")
            failed_bas.append(ba_code)
            # Cleanup orphaned staging table
            try:
                with engine.begin() as conn:
                    conn.execute(text(f'DROP TABLE IF EXISTS raw."temp_weather_{ba_code.lower()}"'))
            except:
                pass

    elapsed_time = datetime.now() - start_time
    logger.info("=" * 60)
    logger.info(f"Weather Backfill Complete!")
    logger.info(f"Total rows inserted: {total_inserted:,}")
    logger.info(f"Failed BAs: {failed_bas if failed_bas else 'None'}")
    logger.info(f"Elapsed: {elapsed_time}")
    logger.info("=" * 60)


if __name__ == "__main__":
    backfill_weather()
