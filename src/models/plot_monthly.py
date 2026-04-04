"""
SENTINEL — Monthly Crisis Visualization (All BAs + VSN)
Generates publication-ready plots for the full Iran 2026 crisis month.

Plots:
  1. Multi-BA demand comparison (actual vs predicted, all 25 BAs)
  2. Hurricane Cone for selected BAs
  3. VSN Variable Importance (using model.forward() workaround)
  4. GDELT Crisis Signal Timeline

Usage:
    python -m src.models.plot_monthly --crisis iran_2026
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from src.models.config import (
    DEFAULT_TFT_CONFIG, DEFAULT_SPLIT_CONFIG,
    CHECKPOINT_DIR, RESULTS_DIR,
)
from src.models.dataset import load_features_df, prepare_dataframe, build_datasets

# ── Crisis Windows ───────────────────────────────────────────────────
CRISIS_WINDOWS = {
    "iran_2026": {
        "label": "Iran Crisis — Feb/Mar 2026",
        "start": "2026-02-15",
        "end": "2026-03-22",
        "escalation": "2026-02-28",  # Geo risk spike date
    },
    "tva_2025": {
        "label": "Shoulder Season Heatwave — TVA 2025",
        "start": "2025-05-14",
        "end": "2025-05-18",
        "escalation": "2025-05-16",  # Peak GKG stress date
    },
}

# Dark theme
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

# Color palette for BAs
BA_COLORS = {
    "ERCO": "#f85149", "PJM": "#58a6ff", "MISO": "#3fb950",
    "CISO": "#d2a8ff", "NYIS": "#f0883e", "ISNE": "#56d4dd",
    "SWPP": "#db61a2", "SOCO": "#7ee787", "TVA": "#ffa657",
    "FPL": "#79c0ff", "DUK": "#ff7b72", "BPAT": "#a5d6ff",
}

# Top BAs to highlight in demand plots
TOP_BAS = ["ERCO", "PJM", "MISO", "CISO", "NYIS", "SWPP", "FPL", "TVA"]


def load_model_and_data(model_variant="A"):
    """Load checkpoint, features, and build datasets."""
    config = DEFAULT_TFT_CONFIG
    split = DEFAULT_SPLIT_CONFIG

    ckpt_dir = CHECKPOINT_DIR / f"tft_model_{model_variant}"
    checkpoints = sorted(ckpt_dir.glob("*.ckpt"))
    if not checkpoints:
        logger.error(f"No checkpoints for model {model_variant}")
        return None, None, None, None

    def extract_val_loss(p):
        try:
            return float(str(p.stem).split("val_loss=")[1])
        except:
            return float("inf")

    best_ckpt = str(min(checkpoints, key=extract_val_loss))
    logger.info(f"Model {model_variant}: {best_ckpt}")

    tft = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)

    df = load_features_df()
    df = prepare_dataframe(df)
    training, _, _ = build_datasets(df, model_variant, config, split)

    return tft, df, training, config


def plot_gdelt_timeline(df, crisis, save_dir):
    """Plot 4: GDELT signal timeline showing crisis escalation."""
    # Get daily GDELT averages
    crisis_df = df[
        (df["period"] >= crisis["start"]) & (df["period"] <= crisis["end"])
    ].copy()
    crisis_df["date"] = crisis_df["period"].dt.date

    daily = crisis_df.groupby("date").agg({
        "energy_tone_regional": "mean",
        "grid_stress_zscore": "mean",
        "electricity_buzz_zscore": "mean",
    }).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Grid Stress Index
    ax = axes[0]
    ax.fill_between(daily["date"], daily["grid_stress_zscore"], alpha=0.4, color="#f85149")
    ax.plot(daily["date"], daily["grid_stress_zscore"], color="#f85149", linewidth=2)
    ax.set_ylabel("Grid Stress (Z)")
    ax.set_title("SENTINEL Systemic Grid Stress Index", fontweight="bold")
    ax.axvline(pd.Timestamp(crisis["escalation"]), color="#ffa657",
               linestyle="--", alpha=0.8, label="Peak Anomaly Date")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Sentiment
    ax = axes[1]
    ax.fill_between(daily["date"], daily["energy_tone_regional"], alpha=0.4, color="#58a6ff")
    ax.plot(daily["date"], daily["energy_tone_regional"], color="#58a6ff", linewidth=2)
    ax.set_ylabel("Avg Sentiment (Tone)")
    ax.set_title("Regional Energy Sentiment (GDELT Tone)", fontweight="bold")
    ax.axvline(pd.Timestamp(crisis["escalation"]), color="#ffa657",
               linestyle="--", alpha=0.8)
    ax.grid(True, alpha=0.3)

    # Event Buzz
    ax = axes[2]
    ax.bar(daily["date"], daily["electricity_buzz_zscore"], color="#3fb950", alpha=0.6, width=0.8)
    ax.set_ylabel("Event Buzz (Z)")
    ax.set_title("Geopolitical Event Volume (Electricity Buzz)", fontweight="bold")
    ax.axvline(pd.Timestamp(crisis["escalation"]), color="#ffa657",
               linestyle="--", alpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    fig.suptitle(f"GDELT Crisis Signals — {crisis['label']}",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = save_dir / "gdelt_crisis_timeline.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close()


def plot_multi_ba_predictions(tft, df, training, config, crisis, save_dir):
    """
    Plot 1: Multi-BA demand predictions for the full crisis month.
    Generates one 24h prediction per BA at the end of the crisis window.
    """
    split = DEFAULT_SPLIT_CONFIG

    # Create per-BA predictions
    results = []
    for ba in TOP_BAS:
        encoder_start = pd.Timestamp(crisis["end"]) - pd.Timedelta(
            hours=config.encoder_length + config.prediction_length + 24
        )
        ba_df = df[
            (df["ba_code"] == ba)
            & (df["period"] >= str(encoder_start))
            & (df["period"] <= crisis["end"])
        ].copy()

        if len(ba_df) < config.encoder_length + config.prediction_length:
            logger.warning(f"Skipping {ba}: not enough data")
            continue

        try:
            ba_dataset = TimeSeriesDataSet.from_dataset(
                training, ba_df, predict=True, stop_randomization=True
            )
            ba_loader = ba_dataset.to_dataloader(
                train=False, batch_size=1, num_workers=0
            )

            preds = tft.predict(
                ba_loader, mode="quantiles", return_x=False,
                trainer_kwargs=dict(accelerator="cpu", devices=1),
            )
            actuals = torch.cat([y[0] for x, y in iter(ba_loader)])

            pred_end = pd.Timestamp(ba_df["period"].max())
            pred_start = pred_end - pd.Timedelta(hours=config.prediction_length - 1)
            time_axis = pd.date_range(pred_start, pred_end, freq="h")

            results.append({
                "ba": ba,
                "time": time_axis,
                "actual": actuals[0].cpu().numpy(),
                "q50": preds[0, :, 3].cpu().numpy(),
                "q10": preds[0, :, 1].cpu().numpy(),
                "q90": preds[0, :, 5].cpu().numpy(),
            })
            logger.info(f"  {ba}: ✅ predicted ({len(time_axis)} hours)")
        except Exception as e:
            logger.warning(f"  {ba}: ❌ {e}")

    if not results:
        logger.error("No predictions generated!")
        return

    # Plot: 2×4 grid of top BAs
    n = len(results)
    rows = (n + 3) // 4
    fig, axes = plt.subplots(rows, min(4, n), figsize=(20, 5 * rows), squeeze=False)

    for idx, r in enumerate(results):
        ax = axes[idx // 4][idx % 4]
        ba = r["ba"]
        color = BA_COLORS.get(ba, "#58a6ff")

        ax.fill_between(r["time"], r["q10"], r["q90"], alpha=0.25, color=color)
        ax.plot(r["time"], r["actual"], color="#f0f6fc", linewidth=2, label="Actual")
        ax.plot(r["time"], r["q50"], color=color, linewidth=1.5, linestyle="--", label="Predicted")

        mape = np.mean(np.abs((r["actual"] - r["q50"]) / np.maximum(r["actual"], 1))) * 100
        ax.set_title(f"{ba} — MAPE: {mape:.1f}%", fontweight="bold", fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlabel("Hour (UTC)")
        if idx % 4 == 0:
            ax.set_ylabel("Demand (MW)")

    # Remove unused axes
    for idx in range(len(results), rows * 4):
        axes[idx // 4][idx % 4].set_visible(False)

    fig.suptitle(f"24h Demand Forecast — All Major BAs\n{crisis['label']}",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = save_dir / "multi_ba_predictions.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close()


def plot_vsn_importance(tft, df, training, config, crisis, save_dir):
    """
    Plot 3: VSN Variable Importance using model.forward() directly.
    This bypasses the broken interpret_output() by manually extracting
    attention weights from the TFT's internal layers.
    """
    # Use TVA as representative BA
    ba = "TVA"
    encoder_start = pd.Timestamp(crisis["end"]) - pd.Timedelta(
        hours=config.encoder_length + config.prediction_length + 24
    )
    ba_df = df[
        (df["ba_code"] == ba)
        & (df["period"] >= str(encoder_start))
        & (df["period"] <= crisis["end"])
    ].copy()

    ba_dataset = TimeSeriesDataSet.from_dataset(
        training, ba_df, predict=True, stop_randomization=True
    )
    ba_loader = ba_dataset.to_dataloader(
        train=False, batch_size=1, num_workers=0
    )

    # Get a single batch
    batch = next(iter(ba_loader))
    x, y = batch

    # Move to CPU and run forward pass directly
    tft.eval()
    tft.cpu()

    with torch.no_grad():
        # The TFT forward() method returns a dict with prediction + attention info
        try:
            raw_out = tft(x)
            interpretation = tft.interpret_output(raw_out, reduction="sum")
            logger.info("Successfully extracted VSN interpretation from forward pass")

        except Exception as e:
            logger.warning(f"interpret_output failed: {e}")
            interpretation = None

    # Fallback: Extract VSN weights from model parameters directly
    if interpretation is None:
        logger.info("Using fallback: extracting learned VSN weights from model parameters")
        interpretation = _extract_vsn_from_params(tft, ba_dataset)

    if interpretation is None:
        logger.error("Could not extract VSN weights")
        return

    # Build importance DataFrame
    if isinstance(interpretation, dict) and "encoder_variables" in interpretation:
        enc_imp = interpretation["encoder_variables"]
        if isinstance(enc_imp, torch.Tensor):
            # Get variable names from the dataset
            var_names = ba_dataset.reals
            values = enc_imp.cpu().numpy()
            if len(values) > len(var_names):
                values = values[:len(var_names)]
            elif len(var_names) > len(values):
                var_names = var_names[:len(values)]
            imp_df = pd.DataFrame({"variable": var_names, "importance": values})
        elif isinstance(enc_imp, dict):
            imp_df = pd.DataFrame([
                {"variable": k, "importance": v.item() if isinstance(v, torch.Tensor) else v}
                for k, v in enc_imp.items()
            ])
        else:
            imp_df = pd.DataFrame({"variable": [f"var_{i}" for i in range(len(enc_imp))],
                                   "importance": list(enc_imp)})
    else:
        imp_df = interpretation  # Already a DataFrame from fallback

    imp_df = imp_df.sort_values("importance", ascending=True).tail(20)

    # Plot
    fig, ax = plt.subplots(figsize=(12, max(6, len(imp_df) * 0.4)))

    gdelt_keywords = ["gdelt", "sentiment", "geo_risk", "event_count"]
    price_keywords = ["gas_price", "nuclear", "oil"]

    colors = []
    for v in imp_df["variable"]:
        vl = v.lower()
        if any(k in vl for k in gdelt_keywords):
            colors.append("#3fb950")  # Green for GDELT
        elif any(k in vl for k in price_keywords):
            colors.append("#ffa657")  # Orange for price
        else:
            colors.append("#58a6ff")  # Blue for other

    ax.barh(imp_df["variable"], imp_df["importance"], color=colors, edgecolor="none",
            height=0.7)
    ax.set_xlabel("Relative Importance (Higher = More Influential)")
    ax.set_title(f"TFT Variable Selection Network — Encoder Importance\n"
                 f"Model A (Baseline) — {ba} during {crisis['label']}",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    legend_elements = [
        Patch(facecolor="#3fb950", label="Geopolitical (GDELT)"),
        Patch(facecolor="#ffa657", label="Commodity Prices"),
        Patch(facecolor="#58a6ff", label="Demand/Weather/Grid"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.8, fontsize=10)

    plt.tight_layout()
    path = save_dir / "vsn_importance.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close()


def _extract_vsn_from_params(tft, dataset):
    """
    Fallback: Extract variable importance from the TFT's learned
    Variable Selection Network weights. These are the static weights
    learned during training — they represent the model's overall
    feature ranking across all data (not crisis-specific).
    """
    try:
        # The TFT stores encoder variable importances in its prescalers and VSN
        # We can extract them from the model's internal state
        encoder_vars = dataset.reals

        # Get the encoder variable selection network weights
        # The VSN has a softmax gate that produces importance weights
        importances = {}

        # Method 1: Use the model's built-in variable importance extraction
        for name, param in tft.named_parameters():
            if "encoder_variable_selection" in name and "weight" in name and "flattened" in name:
                # This is the gate weight matrix
                weights = param.data.cpu().abs().mean(dim=0).numpy()
                if len(weights) == len(encoder_vars):
                    for i, var in enumerate(encoder_vars):
                        importances[var] = float(weights[i])
                break

        if not importances:
            # Method 2: Try accessing prescalers
            logger.info("Trying prescaler-based importance extraction...")
            for i, var in enumerate(encoder_vars):
                key = f"encoder_prescalers.{i}"
                total_weight = 0
                for name, param in tft.named_parameters():
                    if key in name:
                        total_weight += param.data.cpu().abs().sum().item()
                if total_weight > 0:
                    importances[var] = total_weight

        if not importances:
            # Method 3: Simple norm of embedding weights
            logger.info("Using embedding weight norms as proxy for importance...")
            for i, var in enumerate(encoder_vars):
                if i < len(tft.prescalers):
                    w = list(tft.prescalers[i].parameters())
                    norm = sum(p.data.abs().sum().item() for p in w)
                    importances[var] = norm

        if importances:
            # Normalize to [0, 1]
            total = sum(importances.values())
            importances = {k: v / total for k, v in importances.items()}
            return pd.DataFrame([
                {"variable": k, "importance": v}
                for k, v in importances.items()
            ])

    except Exception as e:
        logger.error(f"Fallback VSN extraction failed: {e}")

    return None


def main():
    parser = argparse.ArgumentParser(description="SENTINEL Monthly Crisis Plots")
    parser.add_argument("--crisis", type=str, default="iran_2026",
                        choices=list(CRISIS_WINDOWS.keys()))
    parser.add_argument("--model", type=str, default="A", choices=["A", "B", "C"],
                        help="Model variant to evaluate (A=Baseline, B=SENTINEL)")
    args = parser.parse_args()

    crisis = CRISIS_WINDOWS[args.crisis]
    save_dir = RESULTS_DIR / "plots"
    save_dir.mkdir(exist_ok=True)

    # Load model and data
    logger.info(f"Loading Model {args.model} for {crisis['label']}...")
    tft, df, training, config = load_model_and_data(args.model)
    if tft is None:
        return

    # Plot 1: GDELT Crisis Timeline (Only need this once, but fine to re-gen)
    logger.info("Generating GDELT Crisis Timeline...")
    plot_gdelt_timeline(df, crisis, save_dir)

    # Plot 2: Multi-BA demand predictions
    logger.info(f"Generating Multi-BA demand predictions for Model {args.model}...")
    plot_multi_ba_predictions_variant(tft, df, training, config, crisis, save_dir, args.model)

    # Plot 3: VSN Variable Importance
    logger.info(f"Generating VSN Variable Importance for Model {args.model}...")
    plot_vsn_importance_variant(tft, df, training, config, crisis, save_dir, args.model)

    logger.info(f"\n✅ All monthly crisis plots for Model {args.model} saved to: {save_dir}")


def plot_multi_ba_predictions_variant(tft, df, training, config, crisis, save_dir, variant):
    """Modified multi-BA plot to include variant in filename/title."""
    # ... (rest of logic remains same, just update filename and title)
    results = []
    for ba in TOP_BAS:
        encoder_start = pd.Timestamp(crisis["end"]) - pd.Timedelta(
            hours=config.encoder_length + config.prediction_length + 24
        )
        ba_df = df[
            (df["ba_code"] == ba)
            & (df["period"] >= str(encoder_start if is_utc(df) else encoder_start.tz_localize('UTC')))
            & (df["period"] <= crisis["end"])
        ].copy()
        
        # Ensure UTC
        if ba_df['period'].dt.tz is None:
            ba_df['period'] = ba_df['period'].dt.tz_localize('UTC')

        if len(ba_df) < config.encoder_length + config.prediction_length:
            continue

        try:
            ba_dataset = TimeSeriesDataSet.from_dataset(training, ba_df, predict=True, stop_randomization=True)
            ba_loader = ba_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
            preds = tft.predict(ba_loader, mode="quantiles", trainer_kwargs=dict(accelerator="cpu", devices=1))
            actuals = torch.cat([y[0] for x, y in iter(ba_loader)])
            pred_end = pd.Timestamp(ba_df["period"].max())
            pred_start = pred_end - pd.Timedelta(hours=config.prediction_length - 1)
            results.append({
                "ba": ba, "time": pd.date_range(pred_start, pred_end, freq="h"),
                "actual": actuals[0].cpu().numpy(), "q50": preds[0, :, 3].cpu().numpy(),
                "q10": preds[0, :, 1].cpu().numpy(), "q90": preds[0, :, 5].cpu().numpy(),
            })
        except: pass

    if not results: return

    n = len(results)
    rows = (n + 3) // 4
    fig, axes = plt.subplots(rows, min(4, n), figsize=(20, 5 * rows), squeeze=False)
    for idx, r in enumerate(results):
        ax = axes[idx // 4][idx % 4]
        ba = r["ba"]; color = BA_COLORS.get(ba, "#58a6ff")
        ax.fill_between(r["time"], r["q10"], r["q90"], alpha=0.25, color=color)
        ax.plot(r["time"], r["actual"], color="#f0f6fc", linewidth=2, label="Actual")
        ax.plot(r["time"], r["q50"], color=color, linewidth=1.5, linestyle="--", label="Predicted")
        mape = np.mean(np.abs((r["actual"] - r["q50"]) / np.maximum(r["actual"], 1))) * 100
        ax.set_title(f"{ba} — MAPE: {mape:.1f}%", fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    fig.suptitle(f"24h Demand Forecast — Model {variant}\n{crisis['label']}", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = save_dir / f"multi_ba_predictions_{variant}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_vsn_importance_variant(tft, df, training, config, crisis, save_dir, variant):
    """Modified VSN plot to include variant in title/filename."""
    # Use TVA as representative BA
    ba = "TVA"
    encoder_start = pd.Timestamp(crisis["end"]) - pd.Timedelta(
        hours=config.encoder_length + config.prediction_length + 24
    )
    ba_df = df[
        (df["ba_code"] == ba)
        & (df["period"] >= str(encoder_start if df["period"].dt.tz is None else encoder_start.tz_localize('UTC')))
        & (df["period"] <= crisis["end"])
    ].copy()

    # Ensure UTC
    if ba_df['period'].dt.tz is None:
        ba_df['period'] = ba_df['period'].dt.tz_localize('UTC')

    ba_dataset = TimeSeriesDataSet.from_dataset(training, ba_df, predict=True, stop_randomization=True)
    ba_loader = ba_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
    batch = next(iter(ba_loader)); x, y = batch

    tft.eval(); tft.cpu()
    with torch.no_grad():
        try:
            raw_out = tft(x)
            interpretation = tft.interpret_output(raw_out, reduction="sum")
        except: interpretation = None

    if interpretation is None:
        interpretation = _extract_vsn_from_params(tft, ba_dataset)
    if interpretation is None: return

    # Build importance DataFrame
    if isinstance(interpretation, dict) and "encoder_variables" in interpretation:
        enc_imp = interpretation["encoder_variables"]
        var_names = ba_dataset.reals
        values = enc_imp.cpu().numpy() if isinstance(enc_imp, torch.Tensor) else enc_imp
        # Handle length mismatch if any
        min_len = min(len(var_names), len(values))
        imp_df = pd.DataFrame({"variable": var_names[:min_len], "importance": values[:min_len]})
    else:
        imp_df = interpretation

    imp_df = imp_df.sort_values("importance", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(12, max(6, len(imp_df) * 0.4)))
    gdelt_keywords = ["gdelt", "sentiment", "geo_risk", "event_count"]
    price_keywords = ["gas_price", "nuclear", "oil"]
    colors = []
    for v in imp_df["variable"]:
        vl = v.lower()
        if any(k in vl for k in gdelt_keywords): colors.append("#3fb950")
        elif any(k in vl for k in price_keywords): colors.append("#ffa657")
        else: colors.append("#58a6ff")

    ax.barh(imp_df["variable"], imp_df["importance"], color=colors, height=0.7)
    ax.set_title(f"TFT Variable Selection Network — Model {variant}\n{ba} during {crisis['label']}", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    
    legend_elements = [
        Patch(facecolor="#3fb950", label="Geopolitical (GDELT)"),
        Patch(facecolor="#ffa657", label="Commodity Prices"),
        Patch(facecolor="#58a6ff", label="Demand/Weather/Grid"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.8)

    plt.tight_layout()
    path = save_dir / f"vsn_importance_{variant}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

# Helper to check timezone
def is_utc(df):
    return df['period'].dt.tz is not None


if __name__ == "__main__":
    main()
