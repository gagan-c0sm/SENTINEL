"""
SENTINEL — Rolling 30-Day Crisis Forecast
Simulates a grid operator performing daily 24h forecasts throughout a full month.
Stitches predictions together to show the "Long View" of model performance.

Usage:
    python -m src.models.forecast_rolling --ba ERCO --start 2026-02-20 --end 2026-03-22
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from src.models.config import (
    DEFAULT_TFT_CONFIG, DEFAULT_SPLIT_CONFIG,
    CHECKPOINT_DIR, RESULTS_DIR,
)
from src.models.dataset import load_features_df, prepare_dataframe, build_datasets

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

def rolling_forecast(tft, df, training, ba_code, start_date, end_date, config):
    """
    Performs rolling 24h forecasts. 
    Predicts T to T+24, then moves to T+24 and predicts T+24 to T+48.
    """
    logger.info(f"Starting rolling forecast for {ba_code} from {start_date} to {end_date}")
    
    current_time = pd.Timestamp(start_date).tz_localize("UTC")
    end_time = pd.Timestamp(end_date).tz_localize("UTC")
    
    all_times = []
    all_actuals = []
    all_q50 = []
    all_q10 = []
    all_q90 = []
    
    tft.eval()
    tft.cpu()
    
    step_hours = config.prediction_length # 24h
    
    while current_time < end_time:
        # Encoder window starts config.encoder_length hours before current_time
        enc_start = current_time - pd.Timedelta(hours=config.encoder_length)
        # Sequence ends at current_time + prediction_length
        seq_end = current_time + pd.Timedelta(hours=config.prediction_length)
        
        # Slice data
        window_df = df[
            (df["ba_code"] == ba_code) & 
            (df["period"] >= enc_start) & 
            (df["period"] <= seq_end)
        ].copy()
        
        if len(window_df) < config.encoder_length + config.prediction_length:
            logger.warning(f"Insufficient data at {current_time}, skipping...")
            current_time += pd.Timedelta(hours=step_hours)
            continue
            
        # Build dataset for this single prediction window
        try:
            ds = TimeSeriesDataSet.from_dataset(
                training, window_df, predict=True, stop_randomization=True
            )
            loader = ds.to_dataloader(train=False, batch_size=1, num_workers=0)
            
            # Predict
            out = tft.predict(
                loader, mode="quantiles", return_x=False,
                trainer_kwargs=dict(accelerator="cpu", devices=1)
            )
            
            # Extract actuals for the prediction window
            batch = next(iter(loader))
            actuals = batch[1][0][0].cpu().numpy()
            
            # Extract quantiles
            q50 = out[0, :, 3].cpu().numpy()
            q10 = out[0, :, 1].cpu().numpy()
            q90 = out[0, :, 5].cpu().numpy()
            
            # Time axis for this 24h window
            window_times = pd.date_range(current_time, periods=step_hours, freq="h")
            
            all_times.extend(window_times)
            all_actuals.extend(actuals)
            all_q50.extend(q50)
            all_q10.extend(q10)
            all_q90.extend(q90)
            
            logger.debug(f"  Predicted {current_time.date()}")
            
        except Exception as e:
            logger.error(f"  Error at {current_time}: {e}")
            
        # Advance clock by 24h
        current_time += pd.Timedelta(hours=step_hours)

    return pd.DataFrame({
        "period": all_times,
        "actual": all_actuals,
        "q50": all_q50,
        "q10": all_q10,
        "q90": all_q90
    })

def plot_rolling_results(res_df, ba_code, title, save_path):
    """Plot the multi-day continuous forecast."""
    fig, ax = plt.subplots(figsize=(20, 8))
    
    ax.fill_between(res_df["period"], res_df["q10"], res_df["q90"], 
                    color="#f85149", alpha=0.15, label="80% Prediction Interval")
    
    ax.plot(res_df["period"], res_df["actual"], color="#f0f6fc", 
            linewidth=2, label="Actual Demand", alpha=0.9)
    
    ax.plot(res_df["period"], res_df["q50"], color="#f85149", 
            linewidth=1.5, linestyle="--", label="TFT Rolling Forecast (24h)", alpha=0.8)
    
    # Calculate rolling MAPE (24h window)
    res_df["error"] = np.abs(res_df["actual"] - res_df["q50"]) / np.maximum(res_df["actual"], 1)
    overall_mape = res_df["error"].mean() * 100
    
    ax.set_title(f"{title} — {ba_code}\nOverall Monthly MAPE: {overall_mape:.2f}%", 
                 fontsize=16, fontweight="bold", pad=20)
    
    ax.set_ylabel("Demand (MW)")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.grid(True, alpha=0.2)
    
    # Format X axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Annotate GDELT Escalation
    escalation_date = pd.Timestamp("2026-02-28").tz_localize("UTC")
    if res_df["period"].min() <= escalation_date <= res_df["period"].max():
        ax.axvline(escalation_date, color="#ffa657", linestyle=":", linewidth=2, alpha=0.8)
        ax.annotate("Crisis Escalation (GDELT Spike)", 
                    xy=(escalation_date, ax.get_ylim()[1]*0.9),
                    xytext=(escalation_date + pd.Timedelta(days=1), ax.get_ylim()[1]*0.95),
                    color="#ffa657", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#ffa657"))

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="SENTINEL Rolling Forecast")
    parser.add_argument("--ba", type=str, default="ERCO")
    parser.add_argument("--start", type=str, default="2026-02-15")
    parser.add_argument("--end", type=str, default="2026-03-22")
    parser.add_argument("--model", type=str, default="A")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit path to a .ckpt file")
    args = parser.parse_args()

    # Determine checkpoint
    if args.checkpoint:
        best_ckpt = args.checkpoint
        logger.info(f"Using explicit checkpoint: {best_ckpt}")
    else:
        ckpt_dir = CHECKPOINT_DIR / f"tft_model_{args.model}"
        checkpoints = sorted(ckpt_dir.glob("*.ckpt"))
        if not checkpoints:
            logger.error(f"No checkpoints for Model {args.model}")
            return

        # Use best checkpoint
        def extract_val_loss(p):
            try: return float(str(p.stem).split("val_loss=")[1])
            except: return float("inf")
        best_ckpt = str(min(checkpoints, key=extract_val_loss))
        logger.info(f"Using best checkpoint: {best_ckpt}")
    
    tft = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
    
    df = load_features_df()
    df = prepare_dataframe(df)
    training, _, _ = build_datasets(df, args.model, DEFAULT_TFT_CONFIG, DEFAULT_SPLIT_CONFIG)
    
    # Run rolling forecast
    save_dir = RESULTS_DIR / "plots"
    save_dir.mkdir(exist_ok=True)
    
    bas_to_run = args.ba.split(",")
    for ba in bas_to_run:
        res_df = rolling_forecast(tft, df, training, ba, args.start, args.end, DEFAULT_TFT_CONFIG)
        if not res_df.empty:
            save_path = save_dir / f"rolling_forecast_{ba}_{args.model}.png"
            model_desc = "Full (GDELT+Weather)" if args.model == "B" else "Baseline (Weather-only)"
            title = f"30-Day Rolling Crisis Forecast\nModel {args.model}: {model_desc}"
            plot_rolling_results(res_df, ba, title, save_path)
            
            # Save results to CSV for further analysis
            csv_path = RESULTS_DIR / f"rolling_results_{ba}_{args.model}.csv"
            res_df.to_csv(csv_path, index=False)
            logger.info(f"Saved data to {csv_path}")

if __name__ == "__main__":
    main()
