"""
SENTINEL — Phase 4: TFT Model Evaluation & Interpretability
Generates interpretability plots: variable selection weights and attention.
"""

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer
)
from src.database.connection import get_engine
from src.models.train_tft import load_and_prep_data, CONFIG

# ── Feature Evaluation ───────────────────────────────────────────────────────

def evaluate_interpretability(model_path: str):
    """Load model and extract feature importance."""
    logger.info(f"Loading best model from {model_path}...")
    model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    
    # 1. Variable Selection Weights (Which features mattered?)
    interpretation = model.interpret_output(model.predict(val_dataloader, mode="raw", return_x=True)[0], reduction="sum")
    
    # Static Variables
    plot_importance(interpretation["static_variables"], "Static Variable Importance")
    
    # Encoder (Historical) Variables
    plot_importance(interpretation["encoder_variables"], "Encoder (History) Variable Importance")
    
    # Decoder (Predictive) Variables
    plot_importance(interpretation["decoder_variables"], "Decoder (Future) Variable Importance")
    
    # 2. Attention Map (Where in time did the model look?)
    plot_attention(interpretation["attention"], "Attention Across Timesteps")
    
    logger.info("Interpretability plots generated in models/results/")

def plot_importance(importance_dict: dict, title: str):
    """Create a bar chart of feature importance."""
    df = pd.DataFrame(importance_dict.items(), columns=["Feature", "Importance"])
    df = df.sort_values(by="Importance", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=df, palette="viridis")
    plt.title(title)
    plt.tight_layout()
    
    os.makedirs("models/results", exist_ok=True)
    clean_title = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"models/results/{clean_title}.png")
    plt.close()

def plot_attention(attention_weights, title: str):
    """Visualize mean attention across timesteps."""
    mean_attention = attention_weights.mean(0).cpu().numpy()
    
    plt.figure(figsize=(10, 4))
    plt.plot(range(-len(mean_attention) + 24, 24), mean_attention)  # Time alignment
    plt.axvline(0, color='r', linestyle='--', label='Forecast Boundary')
    plt.xlabel("Time Relative to Forecast Start (hours)")
    plt.ylabel("Attention Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f"models/results/attention_profile.png")
    plt.close()

if __name__ == "__main__":
    # This requires a trained checkpoint path
    # Example: evaluate_interpretability("models/checkpoints/sentinel-tft-epoch=XX-val_loss=YY.ckpt")
    pass
