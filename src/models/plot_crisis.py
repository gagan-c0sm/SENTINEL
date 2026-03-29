"""
SENTINEL — Crisis Window Visualization
Generates publication-ready plots comparing Model A vs Model B during crisis events.

Produces 3 types of plots:
  1. Actual vs Predicted demand (Model A red, Model B green)
  2. Confidence interval "Hurricane Cone" (prediction fan)
  3. Variable Importance (VSN weights) bar chart

Usage:
    python -m src.models.plot_crisis --ba ERCO --crisis iran_2026
    python -m src.models.plot_crisis --ba PJM --crisis iran_2026
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from loguru import logger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from src.models.config import (
    DEFAULT_TFT_CONFIG, DEFAULT_SPLIT_CONFIG,
    CHECKPOINT_DIR, RESULTS_DIR,
)
from src.models.dataset import load_features_df, prepare_dataframe, build_datasets


# ── Crisis Windows ───────────────────────────────────────────────────
CRISIS_WINDOWS = {
    "iran_2026": {
        "label": "Iran Crisis — March 2026",
        "start": "2026-02-25",
        "end": "2026-03-16",
    },
    "texas_2021": {
        "label": "Texas Winter Storm Uri — Feb 2021",
        "start": "2021-02-10",
        "end": "2021-02-25",
    },
    "ukraine_2022": {
        "label": "Russia-Ukraine Invasion — Feb 2022",
        "start": "2022-02-20",
        "end": "2022-03-15",
    },
}

# Plot styling
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "font.family": "sans-serif",
    "font.size": 11,
})


def load_crisis_data(
    ba_code: str,
    crisis_key: str,
    model_variant: str,
):
    """
    Load model checkpoint, filter data to crisis window,
    and generate predictions with mode='raw' for full interpretability.
    """
    config = DEFAULT_TFT_CONFIG
    split = DEFAULT_SPLIT_CONFIG
    crisis = CRISIS_WINDOWS[crisis_key]

    # Load checkpoint
    ckpt_dir = CHECKPOINT_DIR / f"tft_model_{model_variant}"
    checkpoints = sorted(ckpt_dir.glob("*.ckpt"))
    if not checkpoints:
        logger.error(f"No checkpoints for model {model_variant}")
        return None

    def extract_val_loss(p):
        try:
            return float(str(p.stem).split("val_loss=")[1])
        except (IndexError, ValueError):
            return float("inf")
    best_ckpt = str(min(checkpoints, key=extract_val_loss))
    logger.info(f"Model {model_variant} checkpoint: {best_ckpt}")

    tft = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)

    # Load and prepare full dataset
    df = load_features_df()
    df = prepare_dataframe(df)

    # Build training dataset (needed for normalizer params)
    training, _, _ = build_datasets(df, model_variant, config, split)

    # Filter to crisis window + encoder lookback for the target BA
    encoder_start = pd.Timestamp(crisis["start"]) - pd.Timedelta(hours=config.encoder_length + 48)
    crisis_df = df[
        (df["ba_code"] == ba_code)
        & (df["period"] >= str(encoder_start))
        & (df["period"] <= crisis["end"])
    ].copy()

    logger.info(f"Crisis window data for {ba_code}: {len(crisis_df)} rows "
                f"({crisis['start']} to {crisis['end']})")

    if len(crisis_df) < config.encoder_length + config.prediction_length:
        logger.error(f"Not enough data for {ba_code} in crisis window")
        return None

    # Build crisis dataset
    crisis_dataset = TimeSeriesDataSet.from_dataset(
        training,
        crisis_df,
        predict=True,
        stop_randomization=True,
    )

    crisis_loader = crisis_dataset.to_dataloader(
        train=False, batch_size=len(crisis_dataset), num_workers=0
    )

    # Quantile predictions (safe, small dataset)
    predictions_q = tft.predict(
        crisis_loader,
        mode="quantiles",
        return_x=False,
        trainer_kwargs=dict(accelerator="cpu", devices=1),
    )


    # Actuals
    actuals = torch.cat([y[0] for x, y in iter(crisis_loader)])

    # Get the time axis for the prediction window
    # The prediction covers the last `prediction_length` hours of the crisis window
    pred_end = pd.Timestamp(crisis_df["period"].max())
    pred_start = pred_end - pd.Timedelta(hours=config.prediction_length - 1)
    time_axis = pd.date_range(pred_start, pred_end, freq="h")

    return {
        "tft": tft,
        "predictions_q": predictions_q,
        "actuals": actuals,
        "time_axis": time_axis,
        "crisis": crisis,
        "config": config,
        "ba_code": ba_code,
        "model_variant": model_variant,
    }


def plot_actual_vs_predicted(data_a, data_b=None, save_dir=None):
    """
    Plot 1: Actual vs Predicted demand lines.
    Black = actual, Red = Model A, Green = Model B.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    time = data_a["time_axis"]
    actual = data_a["actuals"][0].cpu().numpy()
    pred_a = data_a["predictions_q"][0, :, 3].cpu().numpy()  # q50

    ax.plot(time, actual, color="#f0f6fc", linewidth=2.5, label="Actual Demand", zorder=5)
    ax.plot(time, pred_a, color="#f85149", linewidth=2, linestyle="--",
            label=f"Model A (Baseline)", alpha=0.9, zorder=4)

    if data_b is not None:
        pred_b = data_b["predictions_q"][0, :, 3].cpu().numpy()
        ax.plot(time, pred_b, color="#3fb950", linewidth=2, linestyle="--",
                label=f"Model B (SENTINEL)", alpha=0.9, zorder=4)

    ax.set_title(f"{data_a['crisis']['label']} — {data_a['ba_code']} Grid",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Demand (MW)")
    ax.legend(loc="upper left", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))

    plt.tight_layout()
    if save_dir:
        path = save_dir / f"crisis_demand_{data_a['ba_code']}_{list(CRISIS_WINDOWS.keys())[0]}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.close()
    return fig


def plot_hurricane_cone(data_a, data_b=None, save_dir=None):
    """
    Plot 3: Confidence interval fan chart ("Hurricane Cone").
    Shows q10-q90 (80% CI) and q02-q98 (96% CI).
    """
    fig, axes = plt.subplots(1, 2 if data_b else 1, figsize=(14 if data_b else 8, 6),
                              sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for idx, (data, ax, color, label) in enumerate([
        (data_a, axes[0], "#f85149", "Model A (Baseline)"),
    ] + ([
        (data_b, axes[1], "#3fb950", "Model B (SENTINEL)"),
    ] if data_b else [])):

        time = data["time_axis"]
        actual = data["actuals"][0].cpu().numpy()
        q50 = data["predictions_q"][0, :, 3].cpu().numpy()
        q10 = data["predictions_q"][0, :, 1].cpu().numpy()
        q90 = data["predictions_q"][0, :, 5].cpu().numpy()
        q02 = data["predictions_q"][0, :, 0].cpu().numpy()
        q98 = data["predictions_q"][0, :, 6].cpu().numpy()

        # 96% cone (outer)
        ax.fill_between(time, q02, q98, alpha=0.15, color=color, label="96% CI")
        # 80% cone (inner)
        ax.fill_between(time, q10, q90, alpha=0.3, color=color, label="80% CI")
        # Median prediction
        ax.plot(time, q50, color=color, linewidth=2, label=f"{label} (median)")
        # Actual
        ax.plot(time, actual, color="#f0f6fc", linewidth=2.5, label="Actual", zorder=5)

        # Check coverage
        inside_80 = np.mean((actual >= q10) & (actual <= q90)) * 100
        inside_96 = np.mean((actual >= q02) & (actual <= q98)) * 100
        ax.set_title(f"{label}\n80% coverage: {inside_80:.0f}% | 96% coverage: {inside_96:.0f}%",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (UTC)")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))

    axes[0].set_ylabel("Demand (MW)")
    fig.suptitle(f"Prediction Confidence — {data_a['crisis']['label']} ({data_a['ba_code']})",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    if save_dir:
        path = save_dir / f"crisis_cone_{data_a['ba_code']}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.close()
    return fig


def plot_variable_importance(data, save_dir=None):
    """
    Plot 2: Variable Selection Network (VSN) importance weights.
    Shows which features the model paid attention to during the crisis.
    """
    tft = data["tft"]
    raw = data["raw_predictions"]

    # Interpret the output to get variable importance
    interpretation = tft.interpret_output(raw, reduction="sum")

    # Get encoder variable importance
    encoder_importance = interpretation["encoder_variables"]
    # This is a dict of variable_name -> importance_score for the encoder

    # Convert to DataFrame and sort
    var_names = list(encoder_importance.keys()) if isinstance(encoder_importance, dict) else \
                [f"var_{i}" for i in range(len(encoder_importance))]

    if isinstance(encoder_importance, dict):
        values = [v.item() if isinstance(v, torch.Tensor) else v for v in encoder_importance.values()]
    else:
        values = encoder_importance.cpu().numpy().tolist() if isinstance(encoder_importance, torch.Tensor) else list(encoder_importance)

    imp_df = pd.DataFrame({"variable": var_names, "importance": values})
    imp_df = imp_df.sort_values("importance", ascending=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(imp_df) * 0.35)))

    colors = ["#3fb950" if "gdelt" in v.lower() or "sentiment" in v.lower()
              or "geo_risk" in v.lower() or "event_count" in v.lower()
              else "#58a6ff" for v in imp_df["variable"]]

    ax.barh(imp_df["variable"], imp_df["importance"], color=colors, edgecolor="none")
    ax.set_xlabel("Relative Importance")
    ax.set_title(f"Variable Importance — Model {data['model_variant']} ({data['ba_code']})\n"
                 f"{data['crisis']['label']}",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3fb950", label="Geopolitical (GDELT)"),
        Patch(facecolor="#58a6ff", label="Other Features"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.8)

    plt.tight_layout()
    if save_dir:
        path = save_dir / f"crisis_vsn_{data['ba_code']}_{data['model_variant']}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.close()
    return fig


def main():
    parser = argparse.ArgumentParser(description="SENTINEL Crisis Visualization")
    parser.add_argument("--ba", type=str, default="ERCO", help="Balancing Authority code")
    parser.add_argument("--crisis", type=str, default="iran_2026",
                        choices=list(CRISIS_WINDOWS.keys()))
    parser.add_argument("--model-a-only", action="store_true",
                        help="Only plot Model A (if Model B not yet trained)")
    args = parser.parse_args()

    save_dir = RESULTS_DIR / "plots"
    save_dir.mkdir(exist_ok=True)

    logger.info(f"Loading Model A data for {args.ba} during {args.crisis}...")
    data_a = load_crisis_data(args.ba, args.crisis, "A")

    data_b = None
    if not args.model_a_only:
        ckpt_b = CHECKPOINT_DIR / "tft_model_B"
        if ckpt_b.exists() and list(ckpt_b.glob("*.ckpt")):
            logger.info(f"Loading Model B data for {args.ba} during {args.crisis}...")
            data_b = load_crisis_data(args.ba, args.crisis, "B")
        else:
            logger.warning("Model B checkpoints not found — plotting Model A only")

    if data_a is None:
        logger.error("Failed to load crisis data for Model A")
        return

    # Plot 1: Actual vs Predicted
    logger.info("Generating Plot 1: Actual vs Predicted...")
    plot_actual_vs_predicted(data_a, data_b, save_dir)

    # Plot 2: Hurricane Cone
    logger.info("Generating Plot 2: Confidence Intervals (Hurricane Cone)...")
    plot_hurricane_cone(data_a, data_b, save_dir)

    # Plot 3: Variable Importance (VSN) — skipped for now
    # interpret_output() has compatibility issues with this PyTorch Forecasting version
    # Will be enabled when Model B is ready with a fixed interpret pipeline
    logger.info("Skipping Plot 3 (VSN) — will be generated with Model B comparison")

    logger.info(f"\n✅ All plots saved to: {save_dir}")


if __name__ == "__main__":
    main()
