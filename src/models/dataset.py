"""
SENTINEL — TFT Dataset Builder
Loads analytics.features from TimescaleDB into pytorch-forecasting TimeSeriesDataSet.

Run: python -m src.models.dataset (for standalone test)
"""

import pandas as pd
import numpy as np
from loguru import logger
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.encoders import GroupNormalizer

from src.database.connection import get_engine
from src.models.config import (
    TARGET, GROUP_IDS, STATIC_CATEGORICALS, STATIC_REALS,
    TIME_VARYING_KNOWN_REALS, TIME_VARYING_KNOWN_CATEGORICALS,
    TIME_VARYING_OBSERVED_REALS_MODEL_C, TIME_VARYING_OBSERVED_REALS_MODEL_B,
    TIME_VARYING_OBSERVED_REALS_MODEL_A,
    DEFAULT_TFT_CONFIG, DEFAULT_SPLIT_CONFIG,
)

def load_features_df() -> pd.DataFrame:
    """Load analytics.features from DB into a pandas DataFrame."""
    engine = get_engine()

    logger.info("Loading analytics.features from database...")
    query = """
        SELECT *
        FROM analytics.features
        WHERE demand_mw IS NOT NULL
          AND demand_lag_168h IS NOT NULL
        ORDER BY ba_code, period
    """
    df = pd.read_sql(query, engine, parse_dates=["period"])
    logger.info(f"Loaded {len(df):,} rows, {df['ba_code'].nunique()} BAs")

    return df


def prepare_dataframe(df: pd.DataFrame, model_variant: str = "C") -> pd.DataFrame:
    """
    Prepare the raw DataFrame for TimeSeriesDataSet:
    - Create monotonic time_idx per BA
    - Cast types for pytorch-forecasting
    - Fill NaN in observed features (forward-fill, backward-fill, then 0)
    """
    df = df.copy()

    # Create monotonic time index (hours since dataset start, per BA)
    min_period = df["period"].min()
    df["time_idx"] = ((df["period"] - min_period).dt.total_seconds() / 3600).astype(int)

    # Cast booleans to strings for categorical encoding
    df["is_weekend"] = df["is_weekend"].astype(str)
    df["is_holiday"] = df["is_holiday"].astype(str)

    # Ensure ba_code is string
    df["ba_code"] = df["ba_code"].astype(str)

    # Cast known reals to float
    for col in TIME_VARYING_KNOWN_REALS:
        df[col] = df[col].astype(float)

    # ── Fix 2: Proper NaN handling (NOT fillna(0)) ──────────────────
    # Step 1: Drop oil_price entirely (100% NULL, irrecoverable)
    if 'oil_price' in df.columns:
        df = df.drop(columns=['oil_price'])
        logger.info("Dropped oil_price (100% NULL)")

    # Step 2: Forward-fill price/outage columns (Friday value → weekend)
    # These are daily-reported values; Friday's gas price IS Saturday's price
    FORWARD_FILL_COLS = ['gas_price', 'gas_price_change_7d', 'nuclear_outage_pct',
                         'gas_price_volatility_7d', 'gpr_index', 'gpr_zscore']
    for col in FORWARD_FILL_COLS:
        if col in df.columns:
            df[col] = df.groupby("ba_code")[col].transform(
                lambda s: s.ffill().bfill()
            )

    # Step 3: Fill remaining observed features with ffill/bfill within group
    if model_variant == "C":
        observed_cols = TIME_VARYING_OBSERVED_REALS_MODEL_C
    elif model_variant == "B":
        observed_cols = TIME_VARYING_OBSERVED_REALS_MODEL_B
    else:
        observed_cols = TIME_VARYING_OBSERVED_REALS_MODEL_A

    for col in observed_cols:
        if col in df.columns:
            df[col] = df.groupby("ba_code")[col].transform(
                lambda s: s.ffill().bfill()
            )
            df[col] = df[col].astype(float)

    # Handle Static Reals (Fill NA with 0)
    for col in STATIC_REALS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(float)

    # Drop unused columns
    keep_cols = (
        ["period", "time_idx", TARGET]
        + GROUP_IDS + STATIC_CATEGORICALS + STATIC_REALS
        + TIME_VARYING_KNOWN_REALS + TIME_VARYING_KNOWN_CATEGORICALS
        + observed_cols
    )
    keep_cols = list(dict.fromkeys(keep_cols))
    extra_cols = [c for c in df.columns if c not in keep_cols]
    if extra_cols:
        logger.info(f"Dropping {len(extra_cols)} unused columns: {extra_cols}")
        df = df.drop(columns=extra_cols)

    # Step 4: For remaining NaN, use per-BA median (not zero)
    # fillna(0) creates false signals ($0 gas, 0% nuclear outage on weekends)
    numeric_cols = df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isna().any():
            median_val = df.groupby("ba_code")[col].transform("median")
            df[col] = df[col].fillna(median_val)
            df[col] = df[col].fillna(0)  # Final fallback only if entire group is NaN

    # Drop rows where target is NaN, zero, or negative
    before = len(df)
    df = df[df[TARGET].notna() & (df[TARGET] > 0)]
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} rows with NaN/zero/negative demand_mw")

    # Safety check
    remaining_nans = df.isna().sum().sum()
    if remaining_nans > 0:
        nan_detail = df.isna().sum()
        logger.warning(f"⚠️ {remaining_nans} NaN values remain: {nan_detail[nan_detail > 0].to_dict()}")
    else:
        logger.info("✅ Zero NaN values in prepared DataFrame")

    logger.info(f"Prepared DataFrame: {len(df):,} rows, time_idx range [{df['time_idx'].min()}, {df['time_idx'].max()}]")
    return df


def build_datasets(
    df: pd.DataFrame,
    model_variant: str = "C",
    config: "TFTConfig" = None,
    split: "TrainSplitConfig" = None,
):
    if config is None:
        config = DEFAULT_TFT_CONFIG
    if split is None:
        split = DEFAULT_SPLIT_CONFIG

    if model_variant == "C":
        observed_reals = TIME_VARYING_OBSERVED_REALS_MODEL_C
    elif model_variant == "B":
        observed_reals = TIME_VARYING_OBSERVED_REALS_MODEL_B
    else:
        observed_reals = TIME_VARYING_OBSERVED_REALS_MODEL_A

    # Temporal splits
    train_df = df[(df["period"] >= split.train_start) & (df["period"] < split.train_end)]
    val_df = df[(df["period"] >= split.val_start) & (df["period"] < split.val_end)]
    test_df = df[(df["period"] >= split.test_start) & (df["period"] < split.test_end)]

    logger.info(f"Split sizes — Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # Training dataset
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=TARGET,
        group_ids=GROUP_IDS,
        max_encoder_length=config.encoder_length,
        max_prediction_length=config.prediction_length,
        min_encoder_length=config.encoder_length // 2,
        min_prediction_length=1,

        static_categoricals=STATIC_CATEGORICALS,
        static_reals=STATIC_REALS if model_variant == "C" else [],
        time_varying_known_categoricals=TIME_VARYING_KNOWN_CATEGORICALS,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=[TARGET] + observed_reals,

        target_normalizer=GroupNormalizer(
            groups=GROUP_IDS,
            transformation="log1p",  # Fix 3: log1p handles heterogeneous BA scales
            # softplus overflows on large demand values; log1p maps all BAs
            # to ~7-12 range (log1p(2000)=7.6, log1p(170000)=12.0)
        ),

        categorical_encoders={
            "ba_code": NaNLabelEncoder(add_nan=True),
            "is_weekend": NaNLabelEncoder(add_nan=True),
            "is_holiday": NaNLabelEncoder(add_nan=True),
        },

        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_df,
        predict=True,
        stop_randomization=True,
    )

    test = TimeSeriesDataSet.from_dataset(
        training,
        test_df,
        predict=True,
        stop_randomization=True,
    )

    logger.info(f"Datasets built — Model variant: {model_variant}")
    logger.info(f"  Training samples: {len(training)}")
    logger.info(f"  Validation samples: {len(validation)}")
    logger.info(f"  Test samples: {len(test)}")
    return training, validation, test

if __name__ == "__main__":
    df = load_features_df()
    df = prepare_dataframe(df, model_variant="C")
    training, validation, test = build_datasets(df, model_variant="C")

    sample = training[0]
    print(f"\nSample keys: {list(sample[0].keys())}")
    print(f"Encoder x shape: {sample[0]['encoder_cont'].shape}")
    print(f"Target shape: {sample[1][0].shape}")
