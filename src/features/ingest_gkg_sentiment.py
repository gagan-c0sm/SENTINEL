import pandas as pd
from sqlalchemy import text
from loguru import logger
import sys

from src.database.connection import get_engine

BA_STATE_MAP = {
    'ERCO': ['TX'],
    'NYIS': ['NY'],
    'ISNE': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT'],
    'FPL':  ['FL'],
    'CISO': ['CA'],
    'PJM':  ['PA', 'NJ', 'MD', 'DE', 'OH', 'VA', 'WV', 'IL', 'IN', 'MI', 'NC', 'TN', 'KY'],
    'MISO': ['MI', 'WI', 'MN', 'IA', 'IL', 'IN', 'AR', 'LA', 'MS'],
    'SWPP': ['ND', 'SD', 'NE', 'KS', 'OK', 'TX', 'NM', 'AR', 'LA'],
    'SOCO': ['GA', 'AL', 'FL', 'MS'],
    'TVA':  ['TN', 'AL', 'MS', 'KY', 'GA', 'NC', 'VA'],
    'DUK':  ['NC', 'SC'],
    'BPAT': ['WA', 'OR', 'ID', 'MT']
}

def ingest_gkg():
    logger.info("Loading gkg.csv...")
    try:
        gkg = pd.read_csv('gkg.csv', parse_dates=['article_date'])
    except Exception as e:
        logger.error(f"Failed to load gkg.csv: {e}")
        sys.exit(1)
        
    engine = get_engine()
    
    all_rows = []
    
    for ba, states in BA_STATE_MAP.items():
        logger.info(f"Aggregating GKG sentiment for {ba} (states: {states})...")
        
        # Filter by states relevant to this BA
        ba_data = gkg[gkg['state_code'].isin(states)].copy()
        
        if ba_data.empty:
            logger.warning(f"No GKG data for {ba}")
            continue
            
        # Aggregate across states per day
        # For articles, sum them. For tone, mean.
        daily = ba_data.groupby('article_date').agg({
            'total_energy_articles': 'sum',
            'grid_stress_articles': 'sum',
            'gas_pipeline_articles': 'sum',
            'electricity_articles': 'sum',
            'nuclear_articles': 'sum',
            'renewable_articles': 'sum',
            'avg_energy_tone': 'mean',
            'min_energy_tone': 'min'
        }).reset_index()
        
        # Calculate rates to normalize media volume changes
        daily['grid_stress_rate'] = daily['grid_stress_articles'] / daily['total_energy_articles'].replace(0, 1)
        daily['gas_pipeline_rate'] = daily['gas_pipeline_articles'] / daily['total_energy_articles'].replace(0, 1)
        daily['electricity_buzz_rate'] = daily['electricity_articles'] / daily['total_energy_articles'].replace(0, 1)
        
        # Calculate 30-day rolling Z-scores for sudden spikes
        # Shift(1) to prevent leakage (today's tone depends on yesterday's distribution)
        # Using min_periods=7 to allow early propagation
        rolling_mean_g = daily['grid_stress_rate'].shift(1).rolling(30, min_periods=7).mean()
        rolling_std_g = daily['grid_stress_rate'].shift(1).rolling(30, min_periods=7).std().replace(0, 1)
        daily['grid_stress_zscore'] = (daily['grid_stress_rate'] - rolling_mean_g) / rolling_std_g
        
        rolling_mean_p = daily['gas_pipeline_rate'].shift(1).rolling(30, min_periods=7).mean()
        rolling_std_p = daily['gas_pipeline_rate'].shift(1).rolling(30, min_periods=7).std().replace(0, 1)
        daily['gas_pipeline_zscore'] = (daily['gas_pipeline_rate'] - rolling_mean_p) / rolling_std_p
        
        rolling_mean_e = daily['electricity_buzz_rate'].shift(1).rolling(30, min_periods=7).mean()
        rolling_std_e = daily['electricity_buzz_rate'].shift(1).rolling(30, min_periods=7).std().replace(0, 1)
        daily['electricity_buzz_zscore'] = (daily['electricity_buzz_rate'] - rolling_mean_e) / rolling_std_e
        
        daily['energy_tone_regional'] = daily['avg_energy_tone']
        
        daily['ba_code'] = ba
        
        # Drop NAs created by shift/rolling
        daily = daily[['article_date', 'ba_code', 'grid_stress_zscore', 'gas_pipeline_zscore', 'electricity_buzz_zscore', 'energy_tone_regional']].fillna(0)
        
        all_rows.append(daily)

    final_df = pd.concat(all_rows, ignore_index=True)
    
    logger.info("Uploading GKG data to temporary table...")
    with engine.begin() as conn:
        final_df.to_sql("tmp_gkg", conn, if_exists="replace", index=False)
        
        logger.info("Updating features table with GKG sentiment...")
        conn.execute(text("""
            UPDATE analytics.features f
            SET 
                grid_stress_zscore = t.grid_stress_zscore,
                gas_pipeline_zscore = t.gas_pipeline_zscore,
                electricity_buzz_zscore = t.electricity_buzz_zscore,
                energy_tone_regional = t.energy_tone_regional
            FROM tmp_gkg t
            WHERE f.ba_code = t.ba_code AND f.period::DATE = t.article_date::DATE
        """))
        conn.execute(text("DROP TABLE tmp_gkg"))
        
    logger.info("GKG Sentiment fully ingested.")

if __name__ == "__main__":
    ingest_gkg()
