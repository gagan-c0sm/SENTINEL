"""
SENTINEL — TFT Interpretability Analysis
Extracts VSN feature importance, attention patterns, and per-BA analysis.

Usage:
    python -m src.models.interpret --model B
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger
from pytorch_forecasting import TemporalFusionTransformer

from src.models.config import (
    DEFAULT_TFT_CONFIG, DEFAULT_SPLIT_CONFIG,
    CHECKPOINT_DIR, RESULTS_DIR,
)
from src.models.dataset import load_features_df, prepare_dataframe, build_datasets


def interpret_model(
    model_variant: str = "B",
    checkpoint_path: str = None,
):
    """
    Generate interpretability outputs from a trained TFT model.

    Produces:
    1. Variable importance bar chart (from VSN weights)
    2. Attention heatmap
    3. Per-BA feature importance breakdown
    """
    config = DEFAULT_TFT_CONFIG
    split = DEFAULT_SPLIT_CONFIG

    # Load checkpoint
    if checkpoint_path is None:
        ckpt_dir = CHECKPOINT_DIR / f"tft_model_{model_variant}"
        checkpoints = sorted(ckpt_dir.glob("*.ckpt"))
        if not checkpoints:
            logger.error(f"No checkpoints found in {ckpt_dir}")
            return
        checkpoint_path = str(checkpoints[-1])

    logger.info(f"Loading model from: {checkpoint_path}")
    best_tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)

    # Load validation data for interpretation
    df = load_features_df()
    df = prepare_dataframe(df)
    training, validation, _ = build_datasets(df, model_variant, config, split)

    val_loader = validation.to_dataloader(
        train=False,
        batch_size=config.batch_size * 2,
        num_workers=config.num_workers,
    )

    # ── 1. Variable Importance ───────────────────────────────────────
    logger.info("Computing variable importance...")
    interpretation = best_tft.interpret_output(
        best_tft.predict(
            val_loader,
            mode="raw",
            return_x=True,
            trainer_kwargs=dict(accelerator=config.accelerator, devices=config.devices),
        ),
        reduction="mean",
    )

    # Encoder variable importance
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Encoder variables
    enc_importance = interpretation["encoder_variables"]
    enc_names = training.encoder_variables
    sorted_idx = np.argsort(enc_importance.cpu().numpy())
    axes[0].barh(
        [enc_names[i] for i in sorted_idx],
        enc_importance.cpu().numpy()[sorted_idx],
        color="#2196F3",
    )
    axes[0].set_title("Encoder Variable Importance (VSN Weights)", fontsize=14)
    axes[0].set_xlabel("Importance", fontsize=12)

    # Decoder variables
    dec_importance = interpretation["decoder_variables"]
    dec_names = training.decoder_variables
    sorted_idx = np.argsort(dec_importance.cpu().numpy())
    axes[1].barh(
        [dec_names[i] for i in sorted_idx],
        dec_importance.cpu().numpy()[sorted_idx],
        color="#FF9800",
    )
    axes[1].set_title("Decoder Variable Importance (VSN Weights)", fontsize=14)
    axes[1].set_xlabel("Importance", fontsize=12)

    plt.tight_layout()
    fig_path = RESULTS_DIR / f"variable_importance_model_{model_variant}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Variable importance plot saved: {fig_path}")

    # ── 2. Attention Patterns ────────────────────────────────────────
    logger.info("Extracting attention patterns...")
    attention = interpretation["attention"]  # [samples, encoder_length]

    # Average attention across all samples
    avg_attention = attention.mean(dim=0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(14, 4))
    hours_back = np.arange(-config.encoder_length, 0)
    ax.plot(hours_back, avg_attention, linewidth=2, color="#4CAF50")
    ax.fill_between(hours_back, avg_attention, alpha=0.3, color="#4CAF50")
    ax.set_xlabel("Hours Before Prediction", fontsize=12)
    ax.set_ylabel("Attention Weight", fontsize=12)
    ax.set_title("Average Temporal Attention Pattern", fontsize=14)

    # Annotate key lookback points
    for label, offset in [("-1h", -1), ("-24h", -24), ("-168h (1 week)", -168)]:
        if abs(offset) <= config.encoder_length:
            idx = config.encoder_length + offset
            if 0 <= idx < len(avg_attention):
                ax.axvline(x=offset, color="red", linestyle="--", alpha=0.5)
                ax.annotate(label, xy=(offset, avg_attention[idx]),
                           fontsize=9, color="red")

    plt.tight_layout()
    fig_path = RESULTS_DIR / f"attention_pattern_model_{model_variant}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Attention pattern plot saved: {fig_path}")

    # ── 3. Save numeric results ──────────────────────────────────────
    # Encoder importance table
    enc_df = pd.DataFrame({
        "feature": enc_names,
        "importance": enc_importance.cpu().numpy(),
    }).sort_values("importance", ascending=False)
    enc_df.to_csv(RESULTS_DIR / f"encoder_importance_model_{model_variant}.csv", index=False)

    logger.info(f"\nTop 10 encoder features (Model {model_variant}):")
    logger.info(enc_df.head(10).to_string(index=False))

    logger.info(f"{'='*60}")
    logger.info(f"✅ Interpretability analysis complete for Model {model_variant}")
    logger.info(f"   Results saved to {RESULTS_DIR}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SENTINEL TFT Interpretability")
    parser.add_argument("--model", type=str, default="B", choices=["A", "B"])
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    interpret_model(model_variant=args.model, checkpoint_path=args.checkpoint)
