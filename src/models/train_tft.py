"""
SENTINEL — TFT Training Pipeline
Trains Model C (full features), Model B (Geo), or Model A (baseline) with early stopping.

Usage:
    python -m src.models.train_tft                 # Train Model C (default)
    python -m src.models.train_tft --model B       # Train Model B
    python -m src.models.train_tft --smoke-test     # Quick 2-epoch test on 1 BA
    python -m src.models.train_tft --resume path/to/checkpoint.ckpt
"""

import argparse
import time
from pathlib import Path

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from src.models.crisis_loss import CrisisAwareQuantileLoss

from loguru import logger

from src.models.config import (
    DEFAULT_TFT_CONFIG, DEFAULT_SPLIT_CONFIG,
    CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR
)
from src.models.dataset import load_features_df, prepare_dataframe, build_datasets


class SentinelTFT(TemporalFusionTransformer):
    """Subclassed to override Ranger with AdamW + OneCycleLR"""
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3, # Wait 3 epochs before reducing
            min_lr=1e-6, # Floor
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

def train_tft(
    model_variant: str = "C",
    config=None,
    split=None,
    smoke_test: bool = False,
    ckpt_path: str = None,
):
    if config is None:
        config = DEFAULT_TFT_CONFIG
    if split is None:
        split = DEFAULT_SPLIT_CONFIG

    start_time = time.time()

    # ── 0. Hardware hints ────────────────────────────────────────────
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True

    # ── 1. Load and prepare data ─────────────────────────────────────
    logger.info(f"{'='*60}")
    logger.info(f"SENTINEL TFT Training — Model {model_variant}")
    logger.info(f"  Hardware: {config.precision}, batch={config.batch_size}, workers={config.num_workers}")
    logger.info(f"{'='*60}")

    df = load_features_df()

    if smoke_test:
        logger.info("🔥 SMOKE TEST: using only ERCO, 2 epochs")
        df = df[df["ba_code"] == "ERCO"]
        config.max_epochs = 2
        config.batch_size = 32

    df = prepare_dataframe(df, model_variant=model_variant)

    # ── 2. Build datasets ────────────────────────────────────────────
    training, validation, _ = build_datasets(df, model_variant, config, split)

    # Dataloaders
    train_loader = training.to_dataloader(
        train=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        persistent_workers=True if config.num_workers > 0 else False,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )
    val_loader = validation.to_dataloader(
        train=False,
        batch_size=config.batch_size * 2,
        num_workers=config.num_workers,
        persistent_workers=True if config.num_workers > 0 else False,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )

    # ── 3. Configure model ───────────────────────────────────────────
    tft = SentinelTFT.from_dataset(
        training,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_head_size,
        dropout=config.dropout,
        hidden_continuous_size=config.hidden_continuous_size,
        lstm_layers=config.lstm_layers,
        loss=CrisisAwareQuantileLoss(
            quantiles=config.quantiles,
            outer_weight=2.0,   # Q2/Q98 errors penalized 2x (conservative)
            crisis_boost=5.0,   # Reserved for future dynamic weighting
        ),
        log_interval=config.tft_log_interval,
        log_val_interval=-1,
    )

    param_count = sum(p.numel() for p in tft.parameters())
    logger.info(f"TFT model created: {param_count:,} parameters")

    # ── 4. Callbacks ─────────────────────────────────────────────────
    model_name = f"tft_model_{model_variant}"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.early_stop_patience,
            verbose=True,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=CHECKPOINT_DIR / model_name,
            filename="{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=3,
            mode="min",
            verbose=True,
        ),
    ]

    # ── 5. Train ─────────────────────────────────────────────────────
    tb_logger = TensorBoardLogger(
        save_dir=LOG_DIR,
        name=model_name,
    )

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=config.log_every_n_steps,
        enable_progress_bar=True,
    )

    # IMPORTANT: The load_from_checkpoint block was previously trying to manually reconstruct states
    # due to the bugs with Ranger. With AdamW + PyTorch Lightning, trainer.fit(ckpt_path=) natively works correctly.
    if ckpt_path:
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
    else:
        logger.info("Starting training from scratch...")
    
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

    elapsed = (time.time() - start_time) / 60
    logger.info(f"Training complete in {elapsed:.1f} minutes")

    # ── 6. Load best checkpoint ──────────────────────────────────────
    best_path = trainer.checkpoint_callback.best_model_path
    logger.info(f"Best checkpoint: {best_path}")

    # ── 7. Quick validation metrics ──────────────────────────────────
    try:
        best_tft = SentinelTFT.load_from_checkpoint(best_path)
        val_results = trainer.validate(best_tft, dataloaders=val_loader)
        logger.info(f"Validation loss: {val_results[0]['val_loss']:.4f}")
    except Exception as e:
        logger.warning(f"Post-training validation skipped: {e}")
        logger.info("Run `python -m src.models.evaluate --model {model_variant}` for full metrics.")
        best_tft = tft  # Fallback

    logger.info(f"{'='*60}")
    logger.info(f"✅ Model {model_variant} training complete")
    logger.info(f"   Best checkpoint: {best_path}")
    logger.info(f"   Training time: {elapsed:.1f} minutes")
    logger.info(f"{'='*60}")

    return best_tft, trainer, training

def find_optimal_lr(model_variant: str = "C"):
    config = DEFAULT_TFT_CONFIG
    split = DEFAULT_SPLIT_CONFIG

    df = load_features_df()
    df = prepare_dataframe(df, model_variant=model_variant)
    training, validation, _ = build_datasets(df, model_variant, config, split)

    train_loader = training.to_dataloader(
        train=True, batch_size=config.batch_size, num_workers=config.num_workers
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=config.batch_size * 2, num_workers=config.num_workers
    )

    tft = SentinelTFT.from_dataset(
        training,
        learning_rate=1e-5,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_head_size,
        dropout=config.dropout,
        hidden_continuous_size=config.hidden_continuous_size,
        lstm_layers=config.lstm_layers,
        loss=QuantileLoss(quantiles=config.quantiles),
    )

    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=config.devices,
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val,
    )

    lr_finder = trainer.tuner.lr_find(
        tft, train_dataloaders=train_loader, val_dataloaders=val_loader,
        min_lr=1e-6, max_lr=1.0, num_training=100,
    )

    optimal_lr = lr_finder.suggestion()
    logger.info(f"Optimal learning rate: {optimal_lr}")

    fig = lr_finder.plot(suggest=True)
    fig.savefig(RESULTS_DIR / "lr_finder.png", dpi=150, bbox_inches="tight")
    logger.info(f"LR finder plot saved to {RESULTS_DIR / 'lr_finder.png'}")

    return optimal_lr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SENTINEL TFT Training")
    parser.add_argument("--model", type=str, default="C", choices=["A", "B", "C"],
                        help="Model variant: A (baseline), B (geo), C (GKG/GPR)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick 2-epoch test on 1 BA")
    parser.add_argument("--find-lr", action="store_true",
                        help="Run LR finder instead of training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint (.ckpt) to resume training from")
    args = parser.parse_args()

    if args.find_lr:
        find_optimal_lr(args.model)
    else:
        train_tft(
            model_variant=args.model,
            smoke_test=args.smoke_test,
            ckpt_path=args.resume
        )
