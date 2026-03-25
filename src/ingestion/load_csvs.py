"""
SENTINEL — Bulletproof CSV Interchange Ingestion
Hardened against: column mismatches, case-sensitivity, connection leaks, partial commits.
"""
import os
import glob
import json
import pandas as pd
from sqlalchemy import create_engine, text
import sys
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.settings import get_settings


def robust_csv_upsert():
    settings = get_settings()
    engine = create_engine(settings.database_url, pool_size=2, max_overflow=0)

    csv_files = sorted(glob.glob("d:/Projects/SENTINEL/data/raw_interchange_*.csv"))
    logger.info(f"Found {len(csv_files)} failsafe CSVs on local hard drive.")

    with open("d:/Projects/SENTINEL/data/.backfill_checkpoint.json", "r") as r:
        chk = json.load(r)

    succeeded = 0
    skipped = 0
    failed = 0

    for f in csv_files:
        ba = os.path.basename(f).replace("raw_interchange_", "").replace(".csv", "")

        # Skip previously completed BAs
        status = chk.get("interchange_data", {}).get(f"interchange_{ba}", {}).get("status", "")
        if status == "completed":
            logger.info(f"[SKIP] {ba} — already completed.")
            skipped += 1
            continue

        logger.info(f"[LOAD] {ba} from CSV ({os.path.getsize(f) / 1e6:.1f} MB)...")

        try:
            df = pd.read_csv(f)

            # ── Column Mapping (EIA API -> PostgreSQL) ──
            column_map = {
                "period": "period",
                "fromba": "fromba",
                "fromba-name": "fromba_name",
                "toba": "toba",
                "toba-name": "toba_name",
                "value": "value",
                "value-units": "units",      # DB column is 'units', NOT 'value_units'
            }
            df = df.rename(columns=column_map)

            # Only keep columns that exist in the DB table
            valid_db_cols = ["period", "respondent", "respondent_name", "fromba",
                             "fromba_name", "toba", "toba_name", "value", "units"]
            keep_cols = [c for c in valid_db_cols if c in df.columns]
            df = df[[c for c in keep_cols]].copy()

            # Parse types
            df["period"] = pd.to_datetime(df["period"], utc=True)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

            # Inject respondent (NOT NULL constraint)
            if "respondent" not in df.columns:
                if "fromba" in df.columns:
                    df["respondent"] = df["fromba"]
                if "fromba_name" in df.columns:
                    df["respondent_name"] = df["fromba_name"]
                else:
                    df["respondent_name"] = "Unknown"

            # Dedup within CSV
            subset = [c for c in ["period", "fromba", "toba"] if c in df.columns]
            if subset:
                before = len(df)
                df = df.drop_duplicates(subset=subset)
                if len(df) < before:
                    logger.warning(f"  Dropped {before - len(df)} internal CSV duplicates")

            # ── Staging Table (always lowercase, always quoted in SQL) ──
            temp_table = f"temp_csv_{ba.lower()}"

            logger.info(f"  Staging {len(df):,} rows into raw.\"{temp_table}\"...")
            df.to_sql(
                temp_table,
                engine,
                schema="raw",
                if_exists="replace",
                index=False,
                method="multi",
                chunksize=5000,
            )

            # ── Atomic Merge + Drop in ONE transaction ──
            cols = list(df.columns)
            col_str = ", ".join([f'"{c}"' for c in cols])

            with engine.begin() as conn:
                upsert_sql = f"""
                    INSERT INTO raw.eia_interchange_data ({col_str})
                    SELECT {col_str} FROM raw."{temp_table}"
                    ON CONFLICT (period, fromba, toba) DO NOTHING;
                """
                result = conn.execute(text(upsert_sql))
                inserted = result.rowcount if result.rowcount >= 0 else -1

                conn.execute(text(f'DROP TABLE IF EXISTS raw."{temp_table}"'))

            logger.success(f"  [OK] {ba}: merged {inserted:,} new rows, staging table dropped.")

            # Lock checkpoint
            chk.setdefault("interchange_data", {})[f"interchange_{ba}"] = {"status": "completed"}
            with open("d:/Projects/SENTINEL/data/.backfill_checkpoint.json", "w") as w:
                json.dump(chk, w, indent=4)

            succeeded += 1

        except Exception as e:
            logger.error(f"  [FAIL] {ba}: {e}")
            # Attempt cleanup of orphaned staging table
            try:
                with engine.begin() as conn:
                    conn.execute(text(f'DROP TABLE IF EXISTS raw."temp_csv_{ba.lower()}"'))
                logger.info(f"  Cleaned up orphaned staging table for {ba}")
            except:
                pass
            failed += 1

    logger.info(f"\n--- FINAL REPORT ---")
    logger.info(f"Succeeded: {succeeded} | Skipped: {skipped} | Failed: {failed}")


if __name__ == "__main__":
    robust_csv_upsert()
