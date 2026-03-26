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
    TARGET, GROUP_IDS, STATIC_CATEGORICALS,
    TIME_VARYING_KNOWN_REALS, TIME_VARYING_KNOWN_CATEGORICALS,
    TIME_VARYING_OBSERVED_REALS_MODEL_B, TIME_VARYING_OBSERVED_REALS_MODEL_A,
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


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the raw DataFrame for TimeSeriesDataSet:
    - Create monotonic time_idx per BA
    - Cast types for pytorch-forecasting
    - Fill NaN in observed features (forward-fill then 0)
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

    # Fill NaN in observed features: forward-fill within group (for prices), then 0
    observed_cols = TIME_VARYING_OBSERVED_REALS_MODEL_B
    for col in observed_cols:
        if col in df.columns:
            df[col] = df.groupby("ba_code")[col].transform(
                lambda s: s.ffill().fillna(0)
            )
            df[col] = df[col].astype(float)

    # Drop columns not used by the model (e.g., oil_price is 100% NaN)
    keep_cols = (
        ["period", "time_idx", TARGET]
        + GROUP_IDS + STATIC_CATEGORICALS
        + TIME_VARYING_KNOWN_REALS + TIME_VARYING_KNOWN_CATEGORICALS
        + TIME_VARYING_OBSERVED_REALS_MODEL_B
    )
    keep_cols = list(dict.fromkeys(keep_cols))  # Dedupe preserving order
    extra_cols = [c for c in df.columns if c not in keep_cols]
    if extra_cols:
        logger.info(f"Dropping {len(extra_cols)} unused columns: {extra_cols}")
        df = df.drop(columns=extra_cols)

    # Aggressively fill ALL remaining numeric NaN/inf in the entire DataFrame
    # TimeSeriesDataSet rejects any row with NaN in any column
    numeric_cols = df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], 0)
        if df[col].isna().any():
            df[col] = df[col].fillna(0)

    # Drop rows where target (demand_mw) is NaN, zero, or negative
    # (softplus normalizer requires positive values)
    before = len(df)
    df = df[df[TARGET].notna() & (df[TARGET] > 0)]
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} rows with NaN/zero/negative demand_mw")

    # Final safety check
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
    model_variant: str = "B",
    config: "TFTConfig" = None,
    split: "TrainSplitConfig" = None,
):
    """
    Build train/val/test TimeSeriesDataSets.

    Args:
        df: Prepared DataFrame from prepare_dataframe()
        model_variant: "A" (baseline) or "B" (full + GDELT)
        config: TFT config (defaults to DEFAULT_TFT_CONFIG)
        split: Split config (defaults to DEFAULT_SPLIT_CONFIG)

    Returns:
        (training_dataset, validation_dataset, test_dataset)
    """
    if config is None:
        config = DEFAULT_TFT_CONFIG
    if split is None:
        split = DEFAULT_SPLIT_CONFIG

    # Select observed features based on model variant
    if model_variant == "A":
        observed_reals = TIME_VARYING_OBSERVED_REALS_MODEL_A
    else:
        observed_reals = TIME_VARYING_OBSERVED_REALS_MODEL_B

    # Temporal splits
    train_df = df[(df["period"] >= split.train_start) & (df["period"] < split.train_end)]
    val_df = df[(df["period"] >= split.val_start) & (df["period"] < split.val_end)]
    test_df = df[(df["period"] >= split.test_start) & (df["period"] < split.test_end)]

    logger.info(f"Split sizes — Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # Max encoder length (from training data)
    max_time_idx = train_df["time_idx"].max()
    min_time_idx = train_df["time_idx"].min()

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
        time_varying_known_categoricals=TIME_VARYING_KNOWN_CATEGORICALS,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=[TARGET] + observed_reals,

        target_normalizer=GroupNormalizer(
            groups=GROUP_IDS,
            transformation="softplus",
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

    # Validation dataset — uses training dataset parameters
    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_df,
        predict=True,
        stop_randomization=True,
    )

    # Test dataset — uses training dataset parameters
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
    logger.info(f"  Encoder length: {config.encoder_length}h, Prediction length: {config.prediction_length}h")
    logger.info(f"  Observed features: {len(observed_reals)}")

    return training, validation, test


if __name__ == "__main__":
    """Quick test: load data and build datasets."""
    df = load_features_df()
    df = prepare_dataframe(df)
    training, validation, test = build_datasets(df, model_variant="B")

    # Print a sample
    sample = training[0]
    print(f"\nSample keys: {list(sample[0].keys())}")
    print(f"Encoder x shape: {sample[0]['encoder_cont'].shape}")
    print(f"Target shape: {sample[1][0].shape}")
