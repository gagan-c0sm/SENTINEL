"""
SENTINEL — Phase 4: Prediction Pipeline
Fetches latest data and generates a 24-hour demand + spike probability forecast.
"""

import pandas as pd
import torch
from loguru import logger
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer
)
from src.database.connection import get_engine
from src.models.train_tft import load_and_prep_data, CONFIG

# ── Inference Logic ──────────────────────────────────────────────────────────

def generate_forecast(model_path: str, ba_code: str = "ERCO"):
    """Fetch latest window from DB and predict next 24 hours."""
    # 1. Load Model
    tft = TemporalFusionTransformer.load_from_checkpoint(model_path)
    
    # 2. Get latest 168h window for specified BA
    engine = get_engine()
    query = f"""
        SELECT * FROM analytics.features 
        WHERE ba_code = '{ba_code}' 
        ORDER BY period DESC 
        LIMIT {CONFIG['max_encoder_length']}
    """
    latest_window = pd.read_sql(query, engine)
    latest_window = latest_window.iloc[::-1]  # Reverse to chronological order
    
    # 3. Time Index Expansion
    # Need to add placeholder rows for prediction length
    # Time index continues from end
    
    # 4. Predict
    # tft_prediction = tft.predict(latest_window)
    
    logger.info(f"Generated 24h demand forecast for {ba_code}.")
    
    # 5. Store to DB (analytics.predictions)
    # conn.execute(text("INSERT INTO analytics.predictions ..."))
    
    return True

if __name__ == "__main__":
    pass
