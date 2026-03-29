import pandas as pd
from src.database.connection import get_engine
from sqlalchemy import text
from loguru import logger

def compute_profiles():
    engine = get_engine()
    
    logger.info("Computing Fuel Sensitivity Profiles per BA...")
    query = """
    SELECT 
        ba_code,
        AVG(gas_pct) / 100.0 AS gas_sensitivity,
        AVG(renewable_pct) / 100.0 AS renewable_sensitivity,
        AVG(COALESCE(nuclear_mw, 0) / NULLIF(total_mw, 0)) AS nuclear_sensitivity
    FROM clean.fuel_mix
    GROUP BY ba_code
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    # Fill NAs
    df.fillna(0, inplace=True)
        
    logger.info(f"Computed profiles for {len(df)} BAs. Examples:")
    logger.info(df.head().to_string())
    
    logger.info("Updating features table...")
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text("""
                UPDATE analytics.features 
                SET 
                    gas_sensitivity = :gas,
                    renewable_sensitivity = :ren,
                    nuclear_sensitivity = :nuc
                WHERE ba_code = :ba
                """),
                {
                    "gas": row["gas_sensitivity"],
                    "ren": row["renewable_sensitivity"],
                    "nuc": row["nuclear_sensitivity"],
                    "ba": row["ba_code"]
                }
            )
            
    logger.info("Fuel Profiles successfully injected into Gold Layer.")

if __name__ == "__main__":
    compute_profiles()
