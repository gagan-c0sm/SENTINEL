"""
SENTINEL — TFT Evaluation & Metrics
Computes MAPE, RMSE, quantile loss, coverage — overall AND per-BA.

Usage:
    python -m src.models.evaluate --model C
    python -m src.models.evaluate --model C --cpu
    python -m src.models.evaluate --compare
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from loguru import logger
from pytorch_forecasting import TimeSeriesDataSet

from src.models.config import (
    DEFAULT_TFT_CONFIG, DEFAULT_SPLIT_CONFIG,
    CHECKPOINT_DIR, RESULTS_DIR,
)
from src.models.dataset import load_features_df, prepare_dataframe, build_datasets
from src.models.train_tft import SentinelTFT


def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.sqrt(np.mean((actual - predicted) ** 2))


def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.mean(np.abs(actual - predicted))


def compute_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return np.mean((actual >= lower) & (actual <= upper)) * 100


def evaluate_model(
    model_variant: str = "C",
    checkpoint_path: str = None,
    use_cpu: bool = False,
) -> Dict:
    config = DEFAULT_TFT_CONFIG
    split = DEFAULT_SPLIT_CONFIG

    if use_cpu:
        accelerator = "cpu"
        logger.info("Using CPU for evaluation")
    else:
        accelerator = config.accelerator

    # Load best checkpoint
    if checkpoint_path is None:
        ckpt_dir = CHECKPOINT_DIR / f"tft_model_{model_variant}"
        checkpoints = sorted(ckpt_dir.glob("*.ckpt"))
        if not checkpoints:
            logger.error(f"No checkpoints found in {ckpt_dir}")
            return {}
        def extract_val_loss(p):
            try:
                return float(str(p.stem).split("val_loss=")[1])
            except (IndexError, ValueError):
                return float("inf")
        checkpoint_path = str(min(checkpoints, key=extract_val_loss))
        logger.info(f"Using best checkpoint: {checkpoint_path}")

    best_tft = SentinelTFT.load_from_checkpoint(checkpoint_path)

    # Load test data
    df = load_features_df()
    df = prepare_dataframe(df, model_variant=model_variant)

    training, _, _ = build_datasets(df, model_variant, config, split)

    test_df = df[(df["period"] >= split.test_start) & (df["period"] < split.test_end)]

    test_dataset = TimeSeriesDataSet.from_dataset(
        training,
        test_df,
        predict=True,
        stop_randomization=True,
    )

    logger.info(f"Test dataset: {len(test_dataset)} samples (predict=True)")

    test_loader = test_dataset.to_dataloader(
        train=False,
        batch_size=len(test_dataset),
        num_workers=0,
    )

    # Generate predictions
    logger.info("Generating predictions on test set...")
    predictions = best_tft.predict(
        test_loader,
        mode="quantiles",
        return_x=False,
        trainer_kwargs=dict(accelerator=accelerator, devices=1),
    )

    actuals = torch.cat([y[0] for x, y in iter(test_loader)])
    logger.info(f"Predictions shape: {predictions.shape}, Actuals shape: {actuals.shape}")

    # Extract quantiles — index 3 = 0.5 (median), 1 = 0.1, 5 = 0.9
    q50 = predictions[:, :, 3].cpu().numpy()
    q10 = predictions[:, :, 1].cpu().numpy()
    q90 = predictions[:, :, 5].cpu().numpy()
    q02 = predictions[:, :, 0].cpu().numpy()
    q98 = predictions[:, :, 6].cpu().numpy()
    actual_np = actuals.cpu().numpy()

    # Per-BA metrics
    ba_order = sorted(test_df["ba_code"].unique())
    ba_results = []

    logger.info(f"\n{'='*70}")
    logger.info(f"  Model {model_variant} — PER-BA TEST METRICS (24h forecast)")
    logger.info(f"{'='*70}")
    logger.info(f"  {'BA':<8} {'MAPE':>8} {'MAE':>10} {'RMSE':>10} {'Mean MW':>10} {'Cov80':>7}")
    logger.info(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*7}")

    for i, ba in enumerate(ba_order):
        if i >= len(q50):
            break
        ba_actual = actual_np[i]
        ba_pred = q50[i]
        ba_q10 = q10[i]
        ba_q90 = q90[i]

        mape = compute_mape(ba_actual, ba_pred)
        mae = compute_mae(ba_actual, ba_pred)
        rmse = compute_rmse(ba_actual, ba_pred)
        cov = compute_coverage(ba_actual, ba_q10, ba_q90)
        mean_mw = ba_actual.mean()

        ba_results.append({
            "ba_code": ba, "mape": mape, "mae": mae,
            "rmse": rmse, "mean_mw": mean_mw, "coverage_80": cov,
        })
        logger.info(f"  {ba:<8} {mape:>7.2f}% {mae:>9.0f} {rmse:>9.0f} {mean_mw:>9.0f} {cov:>6.1f}%")

    ba_df = pd.DataFrame(ba_results)

    # Aggregate metrics
    healthy_mask = ba_df["mape"] < 50
    healthy_ba = ba_df[healthy_mask]
    outlier_ba = ba_df[~healthy_mask]

    if len(healthy_ba) > 0:
        weighted_mape = np.average(healthy_ba["mape"], weights=healthy_ba["mean_mw"])
    else:
        weighted_mape = ba_df["mape"].mean()

    q50_flat = q50.flatten()
    actual_flat = actual_np.flatten()
    q10_flat = q10.flatten()
    q90_flat = q90.flatten()

    metrics = {
        "model_variant": model_variant,
        "mape_simple": compute_mape(actual_flat, q50_flat),
        "mape_weighted": weighted_mape,
        "mape_healthy_mean": healthy_ba["mape"].mean() if len(healthy_ba) > 0 else float("nan"),
        "rmse": compute_rmse(actual_flat, q50_flat),
        "mae": compute_mae(actual_flat, q50_flat),
        "coverage_80": compute_coverage(actual_flat, q10_flat, q90_flat),
        "n_bas_total": len(ba_df),
        "n_bas_healthy": len(healthy_ba),
        "n_bas_outlier": len(outlier_ba),
        "n_predictions": len(actual_flat),
    }

    logger.info(f"\n{'='*70}")
    logger.info(f"  Model {model_variant} — AGGREGATE TEST METRICS")
    logger.info(f"{'='*70}")
    logger.info(f"  Simple MAPE (all BAs):     {metrics['mape_simple']:.2f}%")
    logger.info(f"  Weighted MAPE (by demand): {metrics['mape_weighted']:.2f}%")
    logger.info(f"  Healthy BAs Mean MAPE:     {metrics['mape_healthy_mean']:.2f}% ({metrics['n_bas_healthy']}/{metrics['n_bas_total']} BAs)")
    logger.info(f"  MAE:                       {metrics['mae']:.0f} MW")
    logger.info(f"  RMSE:                      {metrics['rmse']:.0f} MW")
    logger.info(f"  80% Coverage:              {metrics['coverage_80']:.1f}% (target: 80%)")
    if len(outlier_ba) > 0:
        logger.warning(f"  ⚠️  Outlier BAs (MAPE>50%): {list(outlier_ba['ba_code'])}")

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    out_path = RESULTS_DIR / f"metrics_model_{model_variant}.csv"
    metrics_df.to_csv(out_path, index=False)
    logger.info(f"  Aggregate metrics saved to {out_path}")

    ba_out = RESULTS_DIR / f"metrics_per_ba_model_{model_variant}.csv"
    ba_df.to_csv(ba_out, index=False)
    logger.info(f"  Per-BA metrics saved to {ba_out}")

    return metrics


def compare_models() -> pd.DataFrame:
    results = []
    for variant in ["A", "B", "C"]:
        path = RESULTS_DIR / f"metrics_model_{variant}.csv"
        if path.exists():
            results.append(pd.read_csv(path))

    if len(results) < 2:
        logger.warning("Need at least 2 model metrics for comparison")
        return pd.DataFrame()

    comparison = pd.concat(results, ignore_index=True)
    logger.info("\n" + comparison.to_string(index=False))

    out_path = RESULTS_DIR / "ablation_comparison.csv"
    comparison.to_csv(out_path, index=False)
    logger.info(f"Comparison saved to {out_path}")

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SENTINEL TFT Evaluation")
    parser.add_argument("--model", type=str, default="C", choices=["A", "B", "C"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Run evaluation on CPU")
    args = parser.parse_args()

    if args.compare:
        compare_models()
    else:
        evaluate_model(
            model_variant=args.model,
            checkpoint_path=args.checkpoint,
            use_cpu=args.cpu,
        )
