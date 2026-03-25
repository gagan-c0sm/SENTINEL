"""
SENTINEL — Historical Data Backfill Script
Pulls 5 years of data from all EIA tables and loads into TimescaleDB.

Usage:
    python -m src.ingestion.backfill                     # Full backfill (all BAs)
    python -m src.ingestion.backfill --ba ERCO PJM       # Specific BAs only
    python -m src.ingestion.backfill --table region_data  # Specific table only
    python -m src.ingestion.backfill --resume             # Resume from last checkpoint
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import text
from tqdm import tqdm

from src.config import get_settings, PROJECT_ROOT
from src.config.settings import (
    BACKFILL_START_DATE,
    BACKFILL_END_DATE,
    KEY_BALANCING_AUTHORITIES,
)
from src.database.connection import get_engine, test_connection
from src.ingestion.eia_client import EIAClient


# Checkpoint file for resume capability
CHECKPOINT_FILE = PROJECT_ROOT / "data" / ".backfill_checkpoint.json"


def save_checkpoint(table: str, ba: str, status: str):
    """Save progress checkpoint for resume capability."""
    checkpoint = load_checkpoint()
    if table not in checkpoint:
        checkpoint[table] = {}
    checkpoint[table][ba] = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
    }
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint() -> dict:
    """Load existing checkpoint."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def is_completed(table: str, ba: str) -> bool:
    """Check if a specific table/BA combination is already backfilled."""
    checkpoint = load_checkpoint()
    return checkpoint.get(table, {}).get(ba, {}).get("status") == "completed"


def insert_region_data(df: pd.DataFrame, engine):
    """Insert region data (demand/forecast/generation/interchange) into raw table."""
    if df.empty:
        return 0

    # Rename API columns to match our schema
    column_map = {
        "period": "period",
        "respondent": "respondent",
        "respondent-name": "respondent_name",
        "type": "type",
        "type-name": "type_name",
        "value": "value",
    }

    df = df.rename(columns=column_map)

    # Keep only columns we need
    keep_cols = [c for c in column_map.values() if c in df.columns]
    df = df[keep_cols].copy()

    # Parse timestamps
    df["period"] = pd.to_datetime(df["period"], utc=True)

    # Convert value to numeric
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Use COPY-like bulk insert via pandas
    rows = df.to_sql(
        "eia_region_data",
        engine,
        schema="raw",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=5000,
    )

    return len(df)


def insert_fuel_type_data(df: pd.DataFrame, engine):
    """Insert fuel type generation data into raw table."""
    if df.empty:
        return 0

    column_map = {
        "period": "period",
        "respondent": "respondent",
        "respondent-name": "respondent_name",
        "fueltype": "fueltype",
        "type-name": "type_name",
        "value": "value",
    }

    df = df.rename(columns=column_map)
    keep_cols = [c for c in column_map.values() if c in df.columns]
    df = df[keep_cols].copy()

    df["period"] = pd.to_datetime(df["period"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df.to_sql(
        "eia_fuel_type_data",
        engine,
        schema="raw",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=5000,
    )

    return len(df)


def insert_interchange_data(df: pd.DataFrame, engine):
    """Insert interchange data into raw table."""
    if df.empty:
        return 0

    column_map = {
        "period": "period",
        "respondent": "respondent",
        "respondent-name": "respondent_name",
        "fromba": "fromba",
        "fromba-name": "fromba_name",
        "toba": "toba",
        "toba-name": "toba_name",
        "value": "value",
    }

    df = df.rename(columns=column_map)
    keep_cols = [c for c in column_map.values() if c in df.columns]
    df = df[keep_cols].copy()

    df["period"] = pd.to_datetime(df["period"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # API changed: manually inject respondent if missing to satisfy PostgreSQL NOT NULL constraint
    if "respondent" not in df.columns:
        # Safely find a source for the respondent code
        source_col = "fromba" if "fromba" in df.columns else None
        if source_col:
            df["respondent"] = df[source_col]
        
        # Safely find a source for the respondent name
        name_col = "fromba_name" if "fromba_name" in df.columns else None
        if name_col:
            df["respondent_name"] = df[name_col]
        else:
            df["respondent_name"] = "Unknown"

    # PREVENT UniqueViolation CRASH: EIA API occasionally returns duplicate rows for the same hour
    # We use column name versions after the mapping
    subset = [c for c in ["period", "fromba", "toba"] if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset)

    df.to_sql(
        "eia_interchange_data",
        engine,
        schema="raw",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=5000,
    )

    return len(df)


def insert_gas_prices(df: pd.DataFrame, engine):
    """Insert natural gas price data."""
    if df.empty:
        return 0

    column_map = {
        "period": "period",
        "series-description": "series_name",
        "area-name": "area_name",
        "product-name": "product_name",
        "process-name": "process_name",
        "value": "value",
        "units": "units",
    }

    df = df.rename(columns=column_map)
    keep_cols = [c for c in column_map.values() if c in df.columns]
    df = df[keep_cols].copy()

    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df.to_sql(
        "eia_gas_prices",
        engine,
        schema="raw",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=5000,
    )

    return len(df)


def insert_oil_prices(df: pd.DataFrame, engine):
    """Insert crude oil price data."""
    if df.empty:
        return 0

    column_map = {
        "period": "period",
        "series": "series_name",
        "product-name": "product_name",
        "area-name": "area_name",
        "value": "value",
        "units": "units",
    }

    df = df.rename(columns=column_map)
    keep_cols = [c for c in column_map.values() if c in df.columns]
    df = df[keep_cols].copy()

    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df.to_sql(
        "eia_oil_prices",
        engine,
        schema="raw",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=5000,
    )

    return len(df)


def insert_nuclear_outages(df: pd.DataFrame, engine):
    """Insert nuclear outage data."""
    if df.empty:
        return 0

    column_map = {
        "period": "period",
        "capacity": "capacity",
        "outage": "outage",
        "percentOutage": "pct_outage",
    }

    df = df.rename(columns=column_map)
    keep_cols = [c for c in column_map.values() if c in df.columns]
    df = df[keep_cols].copy()

    df["period"] = pd.to_datetime(df["period"])
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
    df["outage"] = pd.to_numeric(df["outage"], errors="coerce")
    df["pct_outage"] = pd.to_numeric(df["pct_outage"], errors="coerce")

    df.to_sql(
        "eia_nuclear_outages",
        engine,
        schema="raw",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=5000,
    )

    return len(df)


def backfill_region_data(client: EIAClient, engine, ba_list: list, resume: bool):
    """
    Backfill hourly demand, forecast, generation, and interchange data
    for all specified BAs.
    """
    data_types = ["D", "DF", "NG", "TI"]
    total_rows = 0

    for ba in ba_list:
        for dtype in data_types:
            key = f"region_{ba}_{dtype}"

            if resume and is_completed("region_data", key):
                logger.info(f"Skipping (already done): {ba} — {dtype}")
                continue

            try:
                logger.info(f"Fetching: {ba} — {dtype}")
                df = client.fetch_region_data(
                    respondent=ba,
                    data_type=dtype,
                    start=BACKFILL_START_DATE,
                    end=BACKFILL_END_DATE,
                )

                if not df.empty:
                    rows = insert_region_data(df, engine)
                    total_rows += rows
                    logger.info(f"Inserted {rows:,} rows for {ba} — {dtype}")

                save_checkpoint("region_data", key, "completed")

            except Exception as e:
                logger.error(f"Failed: {ba} — {dtype}: {e}")
                save_checkpoint("region_data", key, f"failed: {e}")

    return total_rows


def backfill_fuel_type_data(client: EIAClient, engine, ba_list: list, resume: bool):
    """Backfill hourly generation by fuel type for all BAs."""
    total_rows = 0

    for ba in ba_list:
        key = f"fuel_{ba}"

        if resume and is_completed("fuel_type_data", key):
            logger.info(f"Skipping (already done): fuel type — {ba}")
            continue

        try:
            logger.info(f"Fetching fuel type data: {ba}")
            df = client.fetch_fuel_type_data(
                respondent=ba,
                start=BACKFILL_START_DATE,
                end=BACKFILL_END_DATE,
            )

            if not df.empty:
                rows = insert_fuel_type_data(df, engine)
                total_rows += rows
                logger.info(f"Inserted {rows:,} fuel type rows for {ba}")

            save_checkpoint("fuel_type_data", key, "completed")

        except Exception as e:
            logger.error(f"Failed fuel type — {ba}: {e}")
            save_checkpoint("fuel_type_data", key, f"failed: {e}")

    return total_rows


def backfill_interchange_data(client: EIAClient, engine, ba_list: list, resume: bool):
    """Backfill hourly interchange data for all BAs."""
    total_rows = 0

    for ba in ba_list:
        key = f"interchange_{ba}"

        if resume and is_completed("interchange_data", key):
            logger.info(f"Skipping (already done): interchange — {ba}")
            continue

        try:
            logger.info(f"Fetching interchange data: {ba}")
            df = client.fetch_interchange_data(
                respondent=ba,
                start=BACKFILL_START_DATE,
                end=BACKFILL_END_DATE,
            )

            if not df.empty:
                # 1. FAILSAFE: Save the hard-earned download to the hard drive immediately.
                csv_path = f"data/raw_interchange_{ba}.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved local backup to {csv_path}")

                # 2. CHUNKING: Prevent PostgreSQL from crashing by inserting in 50,000 row chunks
                chunk_size = 50000
                for i in range(0, len(df), chunk_size):
                    chunk_df = df.iloc[i:i+chunk_size]
                    chk_rows = insert_interchange_data(chunk_df, engine)
                    total_rows += chk_rows
                
                logger.info(f"Inserted {len(df):,} interchange rows for {ba} into TimescaleDB")

            save_checkpoint("interchange_data", key, "completed")

        except Exception as e:
            logger.error(f"Failed interchange — {ba}: {e}")
            save_checkpoint("interchange_data", key, f"failed: {e}")

    return total_rows


def backfill_prices_and_outages(client: EIAClient, engine, resume: bool):
    """Backfill gas prices, oil prices, and nuclear outages (non-BA data)."""
    total_rows = 0

    # Gas prices
    if not (resume and is_completed("prices", "gas")):
        try:
            logger.info("Fetching natural gas prices...")
            df = client.fetch_gas_prices(start=BACKFILL_START_DATE[:10])
            if not df.empty:
                rows = insert_gas_prices(df, engine)
                total_rows += rows
                logger.info(f"Inserted {rows:,} gas price rows")
            save_checkpoint("prices", "gas", "completed")
        except Exception as e:
            logger.error(f"Failed gas prices: {e}")
            save_checkpoint("prices", "gas", f"failed: {e}")

    # Oil prices
    if not (resume and is_completed("prices", "oil")):
        try:
            logger.info("Fetching crude oil prices...")
            df = client.fetch_oil_prices(start=BACKFILL_START_DATE[:10])
            if not df.empty:
                rows = insert_oil_prices(df, engine)
                total_rows += rows
                logger.info(f"Inserted {rows:,} oil price rows")
            save_checkpoint("prices", "oil", "completed")
        except Exception as e:
            logger.error(f"Failed oil prices: {e}")
            save_checkpoint("prices", "oil", f"failed: {e}")

    # Nuclear outages
    if not (resume and is_completed("prices", "nuclear")):
        try:
            logger.info("Fetching nuclear outage data...")
            df = client.fetch_nuclear_outages(start=BACKFILL_START_DATE[:10])
            if not df.empty:
                rows = insert_nuclear_outages(df, engine)
                total_rows += rows
                logger.info(f"Inserted {rows:,} nuclear outage rows")
            save_checkpoint("prices", "nuclear", "completed")
        except Exception as e:
            logger.error(f"Failed nuclear outages: {e}")
            save_checkpoint("prices", "nuclear", f"failed: {e}")

    return total_rows


def main():
    parser = argparse.ArgumentParser(description="SENTINEL Historical Data Backfill")
    parser.add_argument(
        "--ba",
        nargs="*",
        default=None,
        help="Specific BA codes to pull (default: all key BAs)",
    )
    parser.add_argument(
        "--table",
        choices=["region_data", "fuel_type", "interchange", "prices", "all"],
        default="all",
        help="Which table to backfill (default: all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        PROJECT_ROOT / "logs" / "backfill_{time}.log",
        rotation="100 MB",
        level="DEBUG",
    )

    # ── Pre-flight checks ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SENTINEL Data Backfill")
    logger.info(f"Date range: {BACKFILL_START_DATE} → {BACKFILL_END_DATE}")
    logger.info("=" * 60)

    # Test database
    if not test_connection():
        logger.error("Database connection failed! Is Docker running?")
        logger.error("Run: docker compose up -d")
        sys.exit(1)

    # Initialize
    client = EIAClient()
    engine = get_engine()
    ba_list = args.ba or list(KEY_BALANCING_AUTHORITIES.keys())

    logger.info(f"Balancing Authorities: {len(ba_list)} — {ba_list[:5]}...")
    if args.resume:
        logger.info("Resume mode: skipping completed items")

    # ── Run backfill ─────────────────────────────────────────────
    total = 0
    start_time = datetime.now()

    try:
        if args.table in ("region_data", "all"):
            logger.info("▶ Stage 1/4: Region data (demand, forecast, generation, interchange)")
            rows = backfill_region_data(client, engine, ba_list, args.resume)
            total += rows
            logger.info(f"  ✅ Region data: {rows:,} rows")

        if args.table in ("fuel_type", "all"):
            logger.info("▶ Stage 2/4: Fuel type data (generation by source)")
            rows = backfill_fuel_type_data(client, engine, ba_list, args.resume)
            total += rows
            logger.info(f"  ✅ Fuel type: {rows:,} rows")

        if args.table in ("interchange", "all"):
            logger.info("▶ Stage 3/4: Interchange data (BA-to-BA flows)")
            rows = backfill_interchange_data(client, engine, ba_list, args.resume)
            total += rows
            logger.info(f"  ✅ Interchange: {rows:,} rows")

        if args.table in ("prices", "all"):
            logger.info("▶ Stage 4/4: Prices and nuclear outages")
            rows = backfill_prices_and_outages(client, engine, args.resume)
            total += rows
            logger.info(f"  ✅ Prices & outages: {rows:,} rows")

    except KeyboardInterrupt:
        logger.warning("Backfill interrupted by user. Use --resume to continue later.")
        sys.exit(0)

    elapsed = datetime.now() - start_time
    logger.info("=" * 60)
    logger.info(f"BACKFILL COMPLETE")
    logger.info(f"Total rows inserted: {total:,}")
    logger.info(f"Elapsed time: {elapsed}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
