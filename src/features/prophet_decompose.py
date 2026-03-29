import pandas as pd
from prophet import Prophet
from sqlalchemy import text
from loguru import logger
import sys

from src.database.connection import get_engine

def run_prophet():
    engine = get_engine()
    
    logger.info("Reading daily aggregated demand...")
    query = """
    SELECT ba_code, period::DATE as ds, AVG(demand_mw) as y
    FROM clean.demand
    WHERE demand_mw IS NOT NULL
    GROUP BY ba_code, period::DATE
    ORDER BY ba_code, ds
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
    except Exception as e:
        logger.error(f"Failed to load demand: {e}")
        sys.exit(1)
        
    bas = df['ba_code'].unique()
    logger.info(f"Found {len(bas)} BAs to process.")
    
    all_preds = []
    for ba in bas:
        logger.info(f"Fitting Prophet for {ba}...")
        ba_df = df[df['ba_code'] == ba].copy()
        
        # Disable daily seasonality because we are modeling at the daily grain
        m = Prophet(daily_seasonality=False)
        try:
            m.fit(ba_df)
            forecast = m.predict(ba_df[['ds']])
            
            res = forecast[['ds', 'trend', 'weekly', 'yearly']].copy()
            res['ba_code'] = ba
            all_preds.append(res)
        except Exception as e:
            logger.warning(f"Prophet failed for {ba}: {e}. Skipping.")
        
    if not all_preds:
        logger.error("No Prophet models succeeded.")
        sys.exit(1)
        
    final_df = pd.concat(all_preds, ignore_index=True)
    
    logger.info("Updating features table in chunks (via tmp_prophet table)...")
    with engine.begin() as conn:
        final_df.to_sql("tmp_prophet", conn, if_exists="replace", index=False)
        conn.execute(text("""
            UPDATE analytics.features f
            SET 
                prophet_trend = t.trend,
                prophet_weekly = t.weekly,
                prophet_yearly = t.yearly
            FROM tmp_prophet t
            WHERE f.ba_code = t.ba_code AND f.period::DATE = t.ds
        """))
        conn.execute(text("DROP TABLE tmp_prophet"))
        
    logger.info("Prophet Decomposition completely successfully.")

if __name__ == "__main__":
    # Standardize prophet logging to avoid spam
    import logging
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    logging.getLogger('prophet').setLevel(logging.WARNING)
    run_prophet()
