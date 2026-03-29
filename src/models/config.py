"""
SENTINEL — TFT Model Configuration
Hardware profiles: RTX 4060, RTX 5060, RTX 5070 Laptop (Model C).
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

from src.config import PROJECT_ROOT

# ── Output directories ──────────────────────────────────────────────
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

for d in [CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)


# ── Feature definitions ─────────────────────────────────────────────

TARGET = "demand_mw"
GROUP_IDS = ["ba_code"]

STATIC_CATEGORICALS = ["ba_code"]

# New Static Reals for Model C representing BA fuel sensitivity
STATIC_REALS = ["gas_sensitivity", "renewable_sensitivity", "nuclear_sensitivity"]

# Known future features (deterministic)
# Fix 7: temperature_c/hdd/cdd reclassified as known (24h weather forecasts >95% accurate)
# Fix 6: Prophet trend/weekly/yearly are deterministic extrapolations
TIME_VARYING_KNOWN_REALS = [
    "hour_of_day", "day_of_week", "month",
    "temperature_c", "hdd", "cdd",                     # Weather (reliable 24h ahead)
    "prophet_trend", "prophet_weekly", "prophet_yearly", # Prophet (deterministic)
]
TIME_VARYING_KNOWN_CATEGORICALS = ["is_weekend", "is_holiday"]

# Observed features — Model C (Full Pipeline with GKG and GPR)
TIME_VARYING_OBSERVED_REALS_MODEL_C = [
    # Demand lags & rolling stats
    "demand_lag_1h", "demand_lag_24h", "demand_lag_168h",
    "demand_rolling_24h", "demand_rolling_168h", "demand_std_24h",
    
    # Weather (observed-only: less reliable 24h ahead)
    "humidity_pct", "wind_speed_kmh",
    "cloud_cover_pct", "solar_radiation",
    
    # Supply
    "supply_margin_pct",
    "gas_pct", "renewable_pct", "gas_pct_delta_24h",
    
    # Price & nuclear
    "gas_price", "gas_price_volatility_7d", "nuclear_outage_pct",
    
    # GKG Regional Energy Sentiment (Replaces GDELT in Model B)
    "grid_stress_zscore", "gas_pipeline_zscore", "electricity_buzz_zscore",
    "energy_tone_regional",
    
    # GPR Global Index — Z-SCORE as regime modifier, not raw index
    # Raw GPR ranges ~50-400 (arbitrary units). Z-score tells the model
    # "how unusual is today's geopolitical risk vs recent 90-day baseline"
    "gpr_zscore"
]

# Legacy configurations (for ablation testing)
TIME_VARYING_OBSERVED_REALS_MODEL_B = [
    f for f in TIME_VARYING_OBSERVED_REALS_MODEL_C 
    if f not in ["prophet_trend", "prophet_weekly", "prophet_yearly", "supply_margin_pct",
                 "gas_pct_delta_24h", "gas_price_volatility_7d", "grid_stress_zscore", 
                 "gas_pipeline_zscore", "electricity_buzz_zscore", "energy_tone_regional",
                 "gpr_zscore"]
] + ["sentiment_mean_24h", "sentiment_min_24h", "event_count_24h", "geo_risk_index", "gas_price_change_7d"]

TIME_VARYING_OBSERVED_REALS_MODEL_A = [
    f for f in TIME_VARYING_OBSERVED_REALS_MODEL_B
    if f not in ["sentiment_mean_24h", "sentiment_min_24h", "event_count_24h", "geo_risk_index", 
                 "gas_price", "gas_price_change_7d", "nuclear_outage_pct"]
]

# ── Hardware Profiles ───────────────────────────────────────────────

@dataclass
class TFTConfig:
    """Base TFT hyperparameters (architecture — hardware-agnostic)."""

    # Architecture
    hidden_size: int = 64
    attention_head_size: int = 4
    dropout: float = 0.1
    hidden_continuous_size: int = 32
    lstm_layers: int = 2

    # Temporal windows
    encoder_length: int = 168
    prediction_length: int = 24

    # Loss — quantile regression
    quantiles: List[float] = field(
        default_factory=lambda: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    )

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 30
    gradient_clip_val: float = 0.1
    early_stop_patience: int = 5
    reduce_lr_patience: int = 3

    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4

    # Precision & hardware
    precision: str = "16-mixed"
    accelerator: str = "gpu"
    devices: int = 1

    # Logging
    log_every_n_steps: int = 50
    tft_log_interval: int = -1

# RTX 4060 Profile
RTX_4060_CONFIG = TFTConfig(
    batch_size=256,
    num_workers=2,
    pin_memory=True,
    prefetch_factor=4,
    precision="32-true",
    log_every_n_steps=50,
    tft_log_interval=-1,
)

# RTX 5060 Profile
RTX_5060_CONFIG = TFTConfig(
    batch_size=128,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=4,
    precision="bf16-mixed",
    log_every_n_steps=50,
    tft_log_interval=-1,
)

# RTX 5070 Laptop Profile (Model C target)
RTX_5070_LAPTOP_CONFIG = TFTConfig(
    batch_size=64,
    num_workers=2,
    pin_memory=True,
    prefetch_factor=4,
    precision="bf16-mixed",
    log_every_n_steps=50,
    tft_log_interval=-1,
)


@dataclass
class TrainSplitConfig:
    """Temporal data split (no leakage)."""
    train_start: str = "2021-01-01"
    train_end: str = "2024-07-01"
    val_start: str = "2024-07-01"
    val_end: str = "2025-07-01"
    test_start: str = "2025-07-01"
    test_end: str = "2026-04-01"


@dataclass
class OptunaConfig:
    """Optuna hyperparameter search."""
    n_trials: int = 20
    timeout: Optional[int] = None
    study_name: str = "sentinel_tft_model_c"
    storage: str = f"sqlite:///{RESULTS_DIR / 'optuna_study.db'}"

    hidden_size_range: tuple = (32, 128)
    attention_head_range: tuple = (1, 4)
    dropout_range: tuple = (0.05, 0.3)
    learning_rate_range: tuple = (1e-4, 1e-2)
    batch_size_choices: tuple = (32, 64, 128)
    lstm_layers_range: tuple = (1, 3)

# Default configs
DEFAULT_TFT_CONFIG = RTX_5070_LAPTOP_CONFIG
DEFAULT_SPLIT_CONFIG = TrainSplitConfig()
DEFAULT_OPTUNA_CONFIG = OptunaConfig()
