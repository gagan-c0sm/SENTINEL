"""
SENTINEL — TFT Model Configuration
Hyperparameters optimized for RTX 5060 (8GB VRAM) + 32GB RAM, ~4h training.
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

# Known future features (deterministic)
TIME_VARYING_KNOWN_REALS = ["hour_of_day", "day_of_week", "month"]
TIME_VARYING_KNOWN_CATEGORICALS = ["is_weekend", "is_holiday"]

# Observed features — Model B (full: EIA + Weather + Price + GDELT)
TIME_VARYING_OBSERVED_REALS_MODEL_B = [
    # Demand lags & rolling stats
    "demand_lag_1h", "demand_lag_24h", "demand_lag_168h",
    "demand_rolling_24h", "demand_rolling_168h", "demand_std_24h",
    # Weather
    "temperature_c", "humidity_pct", "wind_speed_kmh",
    "cloud_cover_pct", "solar_radiation", "hdd", "cdd",
    # Supply
    "generation_mw", "supply_demand_gap",
    "gas_pct", "renewable_pct", "interchange_mw",
    # Price & nuclear
    "gas_price", "gas_price_change_7d", "nuclear_outage_pct",
    # GDELT geopolitical
    "sentiment_mean_24h", "sentiment_min_24h",
    "event_count_24h", "geo_risk_index",
]

# Model A: baseline (for ablation — no GDELT/price/nuclear)
GDELT_FEATURES = [
    "sentiment_mean_24h", "sentiment_min_24h",
    "event_count_24h", "geo_risk_index",
]
PRICE_FEATURES = ["gas_price", "gas_price_change_7d", "nuclear_outage_pct"]

TIME_VARYING_OBSERVED_REALS_MODEL_A = [
    f for f in TIME_VARYING_OBSERVED_REALS_MODEL_B
    if f not in GDELT_FEATURES + PRICE_FEATURES
]


@dataclass
class TFTConfig:
    """TFT hyperparameters optimized for 8GB VRAM + 32GB RAM."""

    # Architecture
    hidden_size: int = 64               # Safe for 8GB VRAM with FP16
    attention_head_size: int = 4        # 4 heads: daily/weekly/weather/crisis
    dropout: float = 0.1               # Light — VSN handles selection
    hidden_continuous_size: int = 32    # Continuous variable embedding
    lstm_layers: int = 2                # Dual-layer encoder/decoder

    # Temporal windows
    encoder_length: int = 168           # 7-day lookback (weekly cycle)
    prediction_length: int = 24         # 24h forecast horizon

    # Loss — quantile regression
    quantiles: List[float] = field(
        default_factory=lambda: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    )

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 64                # Fits 8GB VRAM with FP16
    max_epochs: int = 50
    gradient_clip_val: float = 0.1
    early_stop_patience: int = 5
    reduce_lr_patience: int = 3
    num_workers: int = 4                # Leverage 32GB RAM for dataloading

    # Precision & hardware
    # RTX 5070 Laptop (Blackwell SM_120) — requires PyTorch nightly cu128
    # BF16 avoids overflow in TFT attention masking (FP16 overflows on mask_fill)
    precision: str = "bf16-mixed"       # BF16 — same range as FP32, native on Blackwell
    accelerator: str = "gpu"
    devices: int = 1

    log_every_n_steps: int = 50


@dataclass
class TrainSplitConfig:
    """Temporal data split (no leakage)."""
    train_start: str = "2021-01-01"
    train_end: str = "2024-07-01"       # 3.5 years
    val_start: str = "2024-07-01"
    val_end: str = "2025-07-01"         # 1 year (full seasonal cycle)
    test_start: str = "2025-07-01"
    test_end: str = "2026-04-01"        # ~9 months held-out


@dataclass
class OptunaConfig:
    """Optuna hyperparameter search."""
    n_trials: int = 20
    timeout: Optional[int] = None
    study_name: str = "sentinel_tft"
    storage: str = f"sqlite:///{RESULTS_DIR / 'optuna_study.db'}"

    hidden_size_range: tuple = (32, 128)
    attention_head_range: tuple = (1, 4)
    dropout_range: tuple = (0.05, 0.3)
    learning_rate_range: tuple = (1e-4, 1e-2)
    batch_size_choices: tuple = (32, 64, 128)
    lstm_layers_range: tuple = (1, 3)


DEFAULT_TFT_CONFIG = TFTConfig()
DEFAULT_SPLIT_CONFIG = TrainSplitConfig()
DEFAULT_OPTUNA_CONFIG = OptunaConfig()
