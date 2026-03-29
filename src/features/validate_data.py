import pandas as pd
from sqlalchemy import text
from loguru import logger
import sys

from src.database.connection import get_engine

EXPECTED_BOUNDS = {
    'ERCO': (10000, 90000),
    'PJM': (30000, 170000),
    'MISO': (30000, 135000),
    'CISO': (8000, 55000),
    'NYIS': (8000, 35000),
    'ISNE': (5000, 28000),
    'SWPP': (15000, 55000),
    'SOCO': (8000, 55000),
    'TVA': (8000, 40000),
    'DUK': (4000, 25000),
    'FPL': (5000, 33000), # Expanded FPL bound slightly based on recent bounds
    'BPAT': (2000, 15000)
}

def validate_demand_bounds():
    engine = get_engine()
    df = pd.read_sql("SELECT ba_code, demand_mw FROM clean.demand WHERE demand_mw IS NOT NULL", engine)
    
    logger.info(f"Loaded {len(df):,} demand records")
    
    passed = True
    for ba, (min_val, max_val) in EXPECTED_BOUNDS.items():
        ba_df = df[df['ba_code'] == ba]
        if ba_df.empty:
            continue
            
        actual_min = ba_df['demand_mw'].min()
        actual_max = ba_df['demand_mw'].max()
        
        violations_min = ba_df[ba_df['demand_mw'] < min_val]
        violations_max = ba_df[ba_df['demand_mw'] > max_val]
        
        if len(violations_min) > 0 or len(violations_max) > 0:
            logger.error(f"{ba}: Out of bounds! Expected [{min_val}, {max_val}], Got [{actual_min:,.0f}, {actual_max:,.0f}]")
            if len(violations_min) > 0:
                logger.warning(f"  {len(violations_min)} records below minimum")
            if len(violations_max) > 0:
                logger.warning(f"  {len(violations_max)} records above maximum")
            passed = False
        else:
            logger.info(f"{ba}: Passed. Bounds: [{actual_min:,.0f}, {actual_max:,.0f}]")
            
    return passed

if __name__ == "__main__":
    logger.info("Running Phase A Data Validation...")
    success = validate_demand_bounds()
    if not success:
        logger.error("Data validation failed! Physical bound violations detected.")
        sys.exit(1)
    logger.info("Data validation passed! No physical bound violations.")
