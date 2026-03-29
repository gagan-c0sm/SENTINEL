import pandas as pd
from sqlalchemy import text
from loguru import logger
import sys

from src.database.connection import get_engine

def ingest_gpr():
    logger.info("Loading data_gpr_daily_recent.xls...")
    try:
        gpr = pd.read_excel('data_gpr_daily_recent.xls', parse_dates=['date'])
    except Exception as e:
        logger.error(f"Failed to load GPR index: {e}")
        sys.exit(1)
        
    gpr = gpr[gpr['date'] >= '2021-01-01'][['date', 'GPRD', 'GPRD_ACT', 'GPRD_THREAT', 'GPRD_MA7']].copy()
    
    # 90-day rolling Z-score for crisis detection
    # Using shift(1) to prevent leakage
    rolling_mean = gpr['GPRD_MA7'].shift(1).rolling(90, min_periods=7).mean()
    rolling_std = gpr['GPRD_MA7'].shift(1).rolling(90, min_periods=7).std().replace(0, 1)
    gpr['gpr_zscore'] = (gpr['GPRD_MA7'] - rolling_mean) / rolling_std
    
    gpr = gpr.fillna(0)
    
    logger.info("Connecting to database...")
    engine = get_engine()
    
    logger.info("Uploading GPR data to temporary table...")
    with engine.begin() as conn:
        gpr.to_sql("tmp_gpr", conn, if_exists="replace", index=False)
        
        # Broadcast to all BAs since GPR is global
        logger.info("Updating features table for all BAs...")
        conn.execute(text("""
            UPDATE analytics.features f
            SET 
                gpr_index = t."GPRD_MA7",
                gpr_zscore = t.gpr_zscore
            FROM tmp_gpr t
            WHERE f.period::DATE = t.date::DATE
        """))
        
        conn.execute(text("DROP TABLE tmp_gpr"))
        
    logger.info("GPR Index fully ingested.")

if __name__ == "__main__":
    ingest_gpr()
