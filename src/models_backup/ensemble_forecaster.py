"""
SENTINEL — Phase 4: Ensemble Forecaster
Combines TFT demand predictions with XGBoost spike classification (Binary: Spike or No-Spike).
Result: A prioritized alert for grid operators.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from loguru import logger
from pytorch_forecasting import TemporalFusionTransformer
from src.database.connection import get_engine

# ── Spike Prediction (Binary Classifier) ───────────────────────────────────

def train_spike_classifier(data: pd.DataFrame):
    """XGBoost classifier to predict binary spikes (is_spike) based on lag/weather features."""
    logger.info("Training XGBoost Spike Classifier...")
    
    # Define features
    features = [
        "hour_of_day", "day_of_week", "month", "is_weekend",
        "demand_lag_1h", "demand_lag_24h", "demand_lag_168h",
        "demand_rolling_24h", "demand_std_24h", 
        "temperature_c", "humidity_pct", "wind_speed_kmh", 
        "cloud_cover_pct", "solar_radiation", "hdd", "cdd",
        "sentiment_mean_24h", "geo_risk_index"
    ]
    
    # Categoricals to int
    for col in ["hour_of_day", "day_of_week", "month", "is_weekend"]:
        data[col] = data[col].astype(int)
        
    X = data[features]
    y = data["is_spike"].astype(int)
    
    # Train/Test Split (Time-based: 80-20)
    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # XGBoost setup
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=10,  # Handle class imbalance (spikes are rare)
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    clf.fit(X_train, y_train)
    
    # Metrics
    score = clf.score(X_test, y_test)
    logger.info(f"Spike Classifier Accuracy: {score:.4f}")
    
    # Save model
    clf.save_model("models/spike_classifier.json")
    return clf

# ── Ensemble Inference ───────────────────────────────────────────────────

def aggregate_forecast(tft_model_path: str, spike_model_path: str, val_data: pd.DataFrame):
    """Aggregates predictions for grid operators: Forecast + Alert."""
    
    # 1. TFT Demand Forecast
    tft = TemporalFusionTransformer.load_from_checkpoint(tft_model_path)
    # tft_forecast = tft.predict(val_dataloader) 
    
    # 2. XGBoost Spike Probability
    clf = xgb.XGBClassifier()
    clf.load_model(spike_model_path)
    
    # 3. Create Aggregated Result
    logger.info("Aggregating output for grid operators...")
    # Logic: If TFT Median > threshold OR Spike Probability > 0.7 → TRIGGER ALERT
    
    # Return structure: 24h demand chart + Alert Level (Green/Yellow/Red)
    return True

if __name__ == "__main__":
    pass
