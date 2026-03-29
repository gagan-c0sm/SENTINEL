"""
SENTINEL — Optuna Hyperparameter Optimization for TFT
Fast tuning mode (Model C): Single split, 3 representative BAs, max 10 epochs with pruning.

Usage:
    python -m src.models.optimize              # Run optimization
    python -m src.models.optimize --trials 30  # Override trial count
"""

import argparse
import warnings
from copy import deepcopy

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting.metrics import QuantileLoss
from loguru import logger

from src.models.config import (
    DEFAULT_TFT_CONFIG, DEFAULT_OPTUNA_CONFIG, RESULTS_DIR,
    TrainSplitConfig
)
from src.models.dataset import load_features_df, prepare_dataframe, build_datasets
from src.models.train_tft import SentinelTFT

warnings.filterwarnings("ignore", ".*does not have many workers.*")


# Fast tuning configuration
TUNING_SPLIT = TrainSplitConfig(
    train_start="2021-01-01", train_end="2024-07-01",
    val_start="2024-07-01", val_end="2025-01-01",
    test_start="2025-01-01", test_end="2025-07-01",
)
TUNING_BAS = ['ERCO', 'PJM', 'CISO']


def objective(trial: optuna.Trial, df, optuna_cfg) -> float:
    """Optuna objective: val_loss on a single split, pruned early."""

    # Sample hyperparameters
    config = deepcopy(DEFAULT_TFT_CONFIG)
    config.hidden_size = trial.suggest_int(
        "hidden_size", *optuna_cfg.hidden_size_range, step=32
    )
    config.attention_head_size = trial.suggest_int(
        "attention_head_size", *optuna_cfg.attention_head_range
    )
    config.dropout = trial.suggest_float(
        "dropout", *optuna_cfg.dropout_range
    )
    config.learning_rate = trial.suggest_float(
        "learning_rate", *optuna_cfg.learning_rate_range, log=True
    )
    config.batch_size = trial.suggest_categorical(
        "batch_size", list(optuna_cfg.batch_size_choices)
    )
    config.lstm_layers = trial.suggest_int(
        "lstm_layers", *optuna_cfg.lstm_layers_range
    )

    logger.info(f"Trial {trial.number}: hidden={config.hidden_size}, "
                f"attn={config.attention_head_size}, drop={config.dropout:.3f}, "
                f"lr={config.learning_rate:.1e}, batch={config.batch_size}, "
                f"lstm={config.lstm_layers}")

    try:
        # Filter to 3 representative BAs for speed
        df_tune = df[df["ba_code"].isin(TUNING_BAS)].copy()

        training, validation, _ = build_datasets(
            df_tune, model_variant="C", config=config, split=TUNING_SPLIT
        )

        train_loader = training.to_dataloader(
            train=True, batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        )
        val_loader = validation.to_dataloader(
            train=False, batch_size=config.batch_size * 2,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        )

        tft = SentinelTFT.from_dataset(
            training,
            learning_rate=config.learning_rate,
            hidden_size=config.hidden_size,
            attention_head_size=config.attention_head_size,
            dropout=config.dropout,
            hidden_continuous_size=config.hidden_continuous_size,
            lstm_layers=config.lstm_layers,
            loss=QuantileLoss(quantiles=config.quantiles),
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
            PyTorchLightningPruningCallback(trial, monitor="val_loss"),
        ]

        trainer = pl.Trainer(
            max_epochs=10,  # Cap at 10 epochs for faster tuning
            accelerator=config.accelerator,
            devices=config.devices,
            precision=config.precision,
            gradient_clip_val=config.gradient_clip_val,
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
        val_loss = trainer.callback_metrics["val_loss"].item()

        logger.info(f"Trial {trial.number} completed. val_loss = {val_loss:.4f}")
        return val_loss

    except optuna.exceptions.TrialPruned:
        logger.info(f"Trial {trial.number} PRUNED (unpromising val_loss)")
        raise
    except Exception as e:
        logger.warning(f"Trial {trial.number} failed: {e}")
        return float("inf")


def run_optimization(n_trials: int = None):
    """Run Optuna hyperparameter search."""
    optuna_cfg = DEFAULT_OPTUNA_CONFIG
    if n_trials is not None:
        optuna_cfg.n_trials = n_trials

    logger.info(f"{'='*60}")
    logger.info(f"SENTINEL — Optuna Hyperparameter Search (FAST MODE)")
    logger.info(f"  Trials: {optuna_cfg.n_trials}")
    logger.info(f"  Target BAs: {TUNING_BAS}")
    logger.info(f"  Pruning: Enabled")
    logger.info(f"{'='*60}")

    # Load data once
    df = load_features_df()
    df = prepare_dataframe(df, model_variant="C")

    study = optuna.create_study(
        study_name=optuna_cfg.study_name,
        storage=optuna_cfg.storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    study.optimize(
        lambda trial: objective(trial, df, optuna_cfg),
        n_trials=optuna_cfg.n_trials,
        timeout=optuna_cfg.timeout,
        show_progress_bar=True,
    )

    # Results
    logger.info(f"\n{'='*60}")
    logger.info("Best hyperparameters:")
    for key, val in study.best_params.items():
        logger.info(f"  {key}: {val}")
    logger.info(f"Best val_loss: {study.best_value:.4f}")
    logger.info(f"{'='*60}")

    # Save results
    results_df = study.trials_dataframe()
    results_df.to_csv(RESULTS_DIR / "optuna_results.csv", index=False)
    logger.info(f"Results saved to {RESULTS_DIR / 'optuna_results.csv'}")

    return study.best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SENTINEL Optuna Search")
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()

    run_optimization(n_trials=args.trials)
