"""
SENTINEL — TFT Evaluation & Metrics
Computes MAPE, RMSE, quantile loss, coverage, and tail-event MAPE.

Usage:
    python -m src.models.evaluate --model B
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from loguru import logger
from pytorch_forecasting import TemporalFusionTransformer

from src.models.config import (
    DEFAULT_TFT_CONFIG, DEFAULT_SPLIT_CONFIG,
    CHECKPOINT_DIR, RESULTS_DIR,
)
from src.models.dataset import load_features_df, prepare_dataframe, build_datasets


# ── Crisis windows for tail-event analysis ───────────────────────────
CRISIS_WINDOWS = {
    "Texas Winter Storm Uri": ("2021-02-13", "2021-02-20"),
    "Russia-Ukraine Invasion": ("2022-02-24", "2022-04-01"),
    "Summer 2023 Heatwave": ("2023-07-01", "2023-08-31"),
    "Red Sea Crisis": ("2024-01-01", "2024-03-31"),
}


def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (excludes zeros)."""
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def compute_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Percentage of actual values inside the prediction interval."""
    return np.mean((actual >= lower) & (actual <= upper)) * 100


def evaluate_model(
    model_variant: str = "B",
    checkpoint_path: str = None,
) -> Dict:
    """
    Full evaluation of a trained TFT model on the test set.

    Returns dict with all metrics.
    """
    config = DEFAULT_TFT_CONFIG
    split = DEFAULT_SPLIT_CONFIG

    # ── Load best checkpoint ─────────────────────────────────────────
    if checkpoint_path is None:
        ckpt_dir = CHECKPOINT_DIR / f"tft_model_{model_variant}"
        checkpoints = sorted(ckpt_dir.glob("*.ckpt"))
        if not checkpoints:
            logger.error(f"No checkpoints found in {ckpt_dir}")
            return {}
        checkpoint_path = str(checkpoints[-1])  # Latest
        logger.info(f"Using checkpoint: {checkpoint_path}")

    best_tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)

    # ── Load test data ───────────────────────────────────────────────
    df = load_features_df()
    df = prepare_dataframe(df)
    training, _, test = build_datasets(df, model_variant, config, split)

    test_loader = test.to_dataloader(
        train=False,
        batch_size=config.batch_size * 2,
        num_workers=config.num_workers,
    )

    # ── Generate predictions ─────────────────────────────────────────
    logger.info("Generating predictions on test set...")
    raw_predictions = best_tft.predict(
        test_loader,
        mode="raw",
        return_x=True,
        trainer_kwargs=dict(accelerator=config.accelerator, devices=config.devices),
    )

    # Extract quantile predictions
    # Shape: [batch, prediction_length, n_quantiles]
    predictions = raw_predictions.output
    actuals = torch.cat([y[0] for x, y in iter(test_loader)])

    # Median (q50) as point forecast — index 3 for [0.02, 0.1, 0.25, 0.5, ...]
    q50 = predictions[:, :, 3].cpu().numpy()
    q10 = predictions[:, :, 1].cpu().numpy()
    q90 = predictions[:, :, 5].cpu().numpy()
    actual_np = actuals.cpu().numpy()

    # Flatten for aggregate metrics
    q50_flat = q50.flatten()
    q10_flat = q10.flatten()
    q90_flat = q90.flatten()
    actual_flat = actual_np.flatten()

    # ── Compute metrics ──────────────────────────────────────────────
    metrics = {
        "model_variant": model_variant,
        "mape": compute_mape(actual_flat, q50_flat),
        "rmse": compute_rmse(actual_flat, q50_flat),
        "coverage_80": compute_coverage(actual_flat, q10_flat, q90_flat),
        "n_test_samples": len(actual_flat),
    }

    logger.info(f"{'='*60}")
    logger.info(f"Model {model_variant} — Test Set Metrics")
    logger.info(f"{'='*60}")
    logger.info(f"  MAPE:          {metrics['mape']:.2f}%")
    logger.info(f"  RMSE:          {metrics['rmse']:.2f} MW")
    logger.info(f"  80% Coverage:  {metrics['coverage_80']:.1f}% (target: 80%)")
    logger.info(f"  Test samples:  {metrics['n_test_samples']:,}")

    # ── Save metrics ─────────────────────────────────────────────────
    metrics_df = pd.DataFrame([metrics])
    out_path = RESULTS_DIR / f"metrics_model_{model_variant}.csv"
    metrics_df.to_csv(out_path, index=False)
    logger.info(f"Metrics saved to {out_path}")

    return metrics


def compare_models() -> pd.DataFrame:
    """Load and compare metrics from Model A and Model B."""
    results = []
    for variant in ["A", "B"]:
        path = RESULTS_DIR / f"metrics_model_{variant}.csv"
        if path.exists():
            results.append(pd.read_csv(path))

    if len(results) < 2:
        logger.warning("Need both Model A and B metrics for comparison")
        return pd.DataFrame()

    comparison = pd.concat(results, ignore_index=True)
    logger.info("\n" + comparison.to_string(index=False))

    # Save comparison
    out_path = RESULTS_DIR / "ablation_comparison.csv"
    comparison.to_csv(out_path, index=False)
    logger.info(f"Comparison saved to {out_path}")

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SENTINEL TFT Evaluation")
    parser.add_argument("--model", type=str, default="B", choices=["A", "B"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare_models()
    else:
        evaluate_model(model_variant=args.model, checkpoint_path=args.checkpoint)
