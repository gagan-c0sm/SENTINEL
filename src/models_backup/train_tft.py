"""
SENTINEL — Phase 4: TFT Model Training
Implementation of the Temporal Fusion Transformer (TFT) for energy demand forecasting.
Features integrated: EIA Demand, Weather, Fuel Mix, and GDELT Geopolitical Signals.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sqlalchemy import text
from datetime import datetime
from typing import Dict, List, Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
    QuantileLoss,
    RMSE,
    MAE,
    GroupNormalizer
)
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, RMSE, QuantileLoss

from src.database.connection import get_engine

# ── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "max_prediction_length": 24,       # Predict 24 hours ahead
    "max_encoder_length": 168,         # Use 1 week of history (168h)
    "batch_size": 128,                 # Fits well in 8GB VRAM
    "epochs": 50,
    "learning_rate": 1e-3,
    "hidden_size": 32,                 # Neurons per layer (robustness vs speed)
    "lstm_layers": 2,
    "attention_head_size": 4,
    "dropout": 0.1,
    "hidden_continuous_size": 16,
    "output_size": 7,                  # Default quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
}

# ── Data Loading & Preprocessing ───────────────────────────────────────────

def load_and_prep_data(limit_bas: List[str] = None):
    """Load from analytics.features and prepare for TimeSeriesDataSet."""
    engine = get_engine()
    
    query = "SELECT * FROM analytics.features"
    if limit_bas:
        bas_str = "', '".join(limit_bas)
        query += f" WHERE ba_code IN ('{bas_str}')"
    query += " ORDER BY ba_code, period"
    
    logger.info("Loading features from database...")
    data = pd.read_sql(query, engine)
    
    # 1. Convert period to datetime and ensure UTC
    data['period'] = pd.to_datetime(data['period'], utc=True)
    
    # 2. Add continuous time index (hours from earliest start)
    min_date = data['period'].min()
    data['time_idx'] = ((data['period'] - min_date).dt.total_seconds() // 3600).astype(int)
    
    # 3. Categorical Encoders
    data['hour_of_day'] = data['hour_of_day'].astype(str)
    data['day_of_week'] = data['day_of_week'].astype(str)
    data['month'] = data['month'].astype(str)
    
    # 4. Fill any remaining NaNs (TFT needs clean data)
    # Most NaNs should have been handled in SQL, but double check
    data = data.ffill().bfill()
    
    logger.info(f"Loaded {len(data):,} rows across {data['ba_code'].nunique()} BAs.")
    return data

# ── Model Training ───────────────────────────────────────────────────────────

def train_tft():
    # 1. Setup
    data = load_and_prep_data()
    
    # Define features
    static_categoricals = ["ba_code"]
    time_varying_known_categoricals = ["hour_of_day", "day_of_week", "month", "is_weekend"]
    time_varying_known_reals = [
        "temperature_c", "humidity_pct", "wind_speed_kmh", 
        "cloud_cover_pct", "solar_radiation", "hdd", "cdd"
    ]
    time_varying_unknown_reals = [
        "demand_mw", "gas_pct", "renewable_pct", "interchange_mw",
        "sentiment_mean_24h", "geo_risk_index"
    ]
    
    # 2. Create Dataset
    max_prediction_length = CONFIG["max_prediction_length"]
    max_encoder_length = CONFIG["max_encoder_length"]
    training_cutoff = data["time_idx"].max() - max_prediction_length
    
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="demand_mw",
        group_ids=["ba_code"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=["ba_code"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )
    
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
    
    train_dataloader = training.to_dataloader(train=True, batch_size=CONFIG["batch_size"], num_workers=4)
    val_dataloader = validation.to_dataloader(train=False, batch_size=CONFIG["batch_size"] * 10, num_workers=4)
    
    logger.info("Created DataLoaders. Initializing TFT Model...")

    # 3. Initialize Model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=CONFIG["learning_rate"],
        hidden_size=CONFIG["hidden_size"],
        lstm_layers=CONFIG["lstm_layers"],
        attention_head_size=CONFIG["attention_head_size"],
        dropout=CONFIG["dropout"],
        hidden_continuous_size=CONFIG["hidden_continuous_size"],
        output_size=CONFIG["output_size"],
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    
    # 4. Trainer with Early Stopping
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath="models/checkpoints/", filename="sentinel-tft-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3, mode="min"
    )
    
    trainer = pl.Trainer(
        max_epochs=CONFIG["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
        logger=TensorBoardLogger("logs/tft_runs")
    )
    
    logger.info("Starting training...")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    # 5. Best Model Metrics
    best_model_path = trainer.checkpoint_callback.best_model_path
    logger.info(f"Training complete. Best model saved at: {best_model_path}")
    
    return tft, best_model_path

if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs("models/checkpoints", exist_ok=True)
    
    # Run training
    train_tft()
