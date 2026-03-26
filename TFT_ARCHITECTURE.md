# SENTINEL — TFT Model Architecture Design

> **Objective**: Design the most robust Temporal Fusion Transformer architecture for 24-hour multi-horizon electricity demand forecasting across 25 US Balancing Authorities, incorporating EIA grid data, weather, and GDELT geopolitical sentiment.

---

## 1. Data Pipeline: Gold Layer → Model Input

The TFT consumes data from `analytics.features` (hourly × 25 BAs, ~5 years). The `pytorch-forecasting` library requires a `TimeSeriesDataSet` with features classified into four categories:

### 1.1 Feature Classification

| Category | Feature | Source | Rationale |
|---|---|---|---|
| **Target** | `demand_mw` | `clean.demand` | What we predict |
| **Group ID** | `ba_code` | Static per series | Identifies each of the 25 independent time series |
| **Time Index** | `time_idx` | Derived integer | Monotonically increasing hourly counter from the dataset start |

#### Static Categoricals (per BA, don't change over time)
| Feature | Type | Notes |
|---|---|---|
| `ba_code` | Categorical | 25 unique BAs — TFT learns BA-specific embeddings |

#### Time-Varying Known Reals (future values are known at prediction time)
| Feature | Source | Notes |
|---|---|---|
| `hour_of_day` | Derived | 0–23, cyclical encoding |
| `day_of_week` | Derived | 0–6, cyclical encoding |
| `month` | Derived | 1–12, cyclical encoding |
| `is_weekend` | Derived | Boolean → 0/1 |
| `is_holiday` | Calendar | Boolean → 0/1 |

#### Time-Varying Observed Reals (only historical values available)
| Feature | Source | Notes |
|---|---|---|
| `demand_mw` | Target (past) | Autoregressive input |
| `demand_lag_1h` | Window fn | 1-hour lag |
| `demand_lag_24h` | Window fn | Day-ago lag |
| `demand_lag_168h` | Window fn | Week-ago lag |
| `demand_rolling_24h` | Window fn | 24h rolling mean |
| `demand_rolling_168h` | Window fn | 7-day rolling mean |
| `demand_std_24h` | Window fn | 24h rolling std (volatility) |
| `temperature_c` | Weather | Primary demand driver |
| `humidity_pct` | Weather | Heat index component |
| `wind_speed_kmh` | Weather | Wind chill + wind gen |
| `cloud_cover_pct` | Weather | Solar gen proxy |
| `solar_radiation` | Weather | Direct solar gen driver |
| `hdd` | Derived | Heating degree days |
| `cdd` | Derived | Cooling degree days |
| `generation_mw` | `clean.demand` | Total generation |
| `supply_demand_gap` | `clean.demand` | Generation - demand |
| `gas_pct` | `clean.fuel_mix` | % from natural gas |
| `renewable_pct` | `clean.fuel_mix` | % from renewables |
| `interchange_mw` | `clean.demand` | Net power flow |
| `sentiment_mean_24h` | GDELT | 24h avg Goldstein scale |
| `sentiment_min_24h` | GDELT | 24h min (worst crisis) |
| `event_count_24h` | GDELT | Event volume |
| `geo_risk_index` | GDELT | Composite risk score |

### 1.2 Why This Classification?

- **Known covariates** are calendar features whose future values are deterministic — the TFT uses these to condition the decoder (future prediction steps).
- **Observed covariates** are all real-world measurements that are only available up to `t=now`. The TFT encoder processes these to build temporal context.
- **Static categoricals** enable BA-specific embeddings — critical because ERCO (Texas, gas-heavy) and CISO (California, solar-heavy) respond differently to the same weather/geopolitical signals.

---

## 2. Model Architecture

### 2.1 TFT Configuration

```python
# Core architecture parameters
TFT_CONFIG = {
    "hidden_size": 64,              # Hidden state dimension (sweet spot for 25 BAs × 5yr)
    "attention_head_size": 4,       # Multi-head attention heads
    "dropout": 0.1,                 # Regularization
    "hidden_continuous_size": 32,   # Continuous variable embedding size
    "lstm_layers": 2,               # Encoder/decoder LSTM depth

    # Temporal windows
    "encoder_length": 168,          # 7 days lookback (captures weekly seasonality)
    "prediction_length": 24,        # 24-hour forecast horizon

    # Loss function
    "loss": "QuantileLoss",         # Quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]

    # Training
    "learning_rate": 1e-3,          # Initial LR (tuned via LR finder)
    "batch_size": 64,               # Fits in 8GB VRAM
    "max_epochs": 50,               # With early stopping
    "gradient_clip_val": 0.1,       # Prevent exploding gradients
}
```

### 2.2 Architecture Rationale

| Parameter | Value | Why |
|---|---|---|
| `hidden_size=64` | 64 | Balances capacity vs. VRAM. With 25 BAs and ~24 input features, 64 is sufficient. Literature shows limited gains above 128 for grid-level data. |
| `attention_heads=4` | 4 | Standard from original TFT paper. 4 heads capture: daily pattern, weekly pattern, weather correlation, and crisis response. |
| `encoder_length=168` | 168h (7 days) | Captures full weekly cycle. Critical for weekday/weekend demand patterns. Also catches GDELT's 24-72h delayed effect. |
| `prediction_length=24` | 24h | Standard day-ahead forecasting horizon. Matches EIA's own forecast window. |
| `QuantileLoss` | 7 quantiles | Provides probabilistic forecasts with 80% and 96% prediction intervals. Essential for spike detection confidence. |
| `lstm_layers=2` | 2 | Deeper than default (1) to capture complex temporal interactions between weather and demand. |
| `dropout=0.1` | 10% | Light regularization — TFT's VSN already handles feature selection, so heavy dropout is unnecessary. |
| `gradient_clip=0.1` | 0.1 | Prevents training instability from GDELT's occasional extreme values (crisis spikes). |

### 2.3 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEMPORAL FUSION TRANSFORMER                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              VARIABLE SELECTION NETWORK (VSN)            │   │
│  │  Learns per-feature importance weights at each timestep  │   │
│  │  → GDELT features get weight ≈ 0 in normal periods      │   │
│  │  → GDELT features get weight >> 0 during crises          │   │
│  └──────────────┬───────────────────────┬───────────────────┘   │
│                 │                       │                       │
│  ┌──────────────▼──────────┐ ┌──────────▼──────────────────┐   │
│  │   ENCODER (Past)        │ │   DECODER (Future)          │   │
│  │   168h lookback window  │ │   24h prediction window     │   │
│  │                         │ │                             │   │
│  │   2-layer LSTM          │ │   2-layer LSTM              │   │
│  │   Processes:            │ │   Processes:                │   │
│  │   - Past demand + lags  │ │   - Known future: calendar  │   │
│  │   - Past weather        │ │     (hour, day, month,      │   │
│  │   - Past fuel mix       │ │      weekend, holiday)      │   │
│  │   - Past GDELT signals  │ │                             │   │
│  └──────────────┬──────────┘ └──────────┬──────────────────┘   │
│                 │                       │                       │
│  ┌──────────────▼───────────────────────▼──────────────────┐   │
│  │         INTERPRETABLE MULTI-HEAD ATTENTION              │   │
│  │         4 heads × 64 hidden dim                         │   │
│  │         Identifies which past timesteps matter most     │   │
│  │         for each future prediction step                 │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │              GATED RESIDUAL NETWORKS                    │   │
│  │    GLU gates can completely shut off information paths   │   │
│  │    → Acts as automatic feature pruning                  │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │              QUANTILE OUTPUT LAYER                      │   │
│  │    Outputs 7 quantiles × 24 hours = 168 values          │   │
│  │    [q02, q10, q25, q50, q75, q90, q98]                  │   │
│  │    q50 = point forecast, [q10,q90] = 80% CI             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Training Strategy

### 3.1 Data Split (Temporal, No Leakage)

```
|←───── Training ─────→|←─ Val ─→|←─ Test ─→|
2021-01     →      2024-06    2024-07→2025-06   2025-07→2026-03
        3.5 years           1 year          ~9 months
```

- **Training**: 2021-01 → 2024-06 (~3.5 years, ~30,600h × 25 BAs = ~765K samples)
- **Validation**: 2024-07 → 2025-06 (1 year — captures full seasonal cycle)
- **Test**: 2025-07 → 2026-03 (~9 months — final held-out evaluation)

> [!IMPORTANT]
> **Strictly temporal split — no random shuffling.** This prevents future information from leaking into training. The validation set spans a full year to ensure seasonal patterns are evaluated.

### 3.2 Walk-Forward Validation (for Hyperparameter Tuning)

For Optuna hyperparameter search, use 3-fold expanding window:

| Fold | Train | Validate |
|---|---|---|
| Fold 1 | 2021-01 → 2022-12 | 2023-01 → 2023-06 |
| Fold 2 | 2021-01 → 2023-06 | 2023-07 → 2023-12 |
| Fold 3 | 2021-01 → 2024-06 | 2024-07 → 2024-12 |

Average validation MAPE across all 3 folds determines the best hyperparameter configuration.

### 3.3 Training Configuration

```python
TRAINER_CONFIG = {
    "accelerator": "gpu",           # RTX 4060 / 5060
    "devices": 1,
    "precision": "16-mixed",        # FP16 mixed precision → 2× VRAM savings
    "max_epochs": 50,
    "gradient_clip_val": 0.1,
    "callbacks": [
        "EarlyStopping(patience=5, monitor='val_loss')",
        "LearningRateMonitor()",
        "ModelCheckpoint(save_top_k=3, monitor='val_loss')",
    ],
    "log_every_n_steps": 50,
}
```

### 3.4 Normalization Strategy

- Use `EncoderNormalizer(method="robust")` per group (per BA) — avoids look-ahead bias.
- Robust scaling (median/IQR) is preferred over standard scaling because demand distributions are often right-skewed with heavy tails from spike events.

---

## 4. Ablation Experiment Design

> **Core Research Question**: Does geopolitical sentiment (GDELT) improve energy demand forecasting?

### Model A (Baseline): EIA + Weather + Calendar
- **Observed features**: demand lags, rolling stats, weather, fuel mix, interchange, HDD/CDD
- **Known features**: hour, day_of_week, month, is_weekend, is_holiday
- **Static**: ba_code
- **Excludes**: All `sentiment_*`, `event_count_24h`, `geo_risk_index`

### Model B (Full): EIA + Weather + Calendar + GDELT
- Same as Model A, **plus**:
- `sentiment_mean_24h`, `sentiment_min_24h`, `event_count_24h`, `geo_risk_index`

### Evaluation Metrics

| Metric | What It Measures | Formula |
|---|---|---|
| **MAPE** | Overall accuracy | `mean(abs(actual - predicted) / actual) × 100` |
| **RMSE** | Penalizes large errors | `sqrt(mean((actual - predicted)²))` |
| **P50 QL** | Quantile calibration | Quantile loss at median |
| **P90 QL** | Tail calibration | Quantile loss at 90th percentile |
| **Tail-MAPE** | Crisis accuracy | MAPE computed only during known crisis periods (Texas freeze, Ukraine invasion, etc.) |
| **Coverage** | CI calibration | % of actual values falling within 80% prediction interval (target: 80%) |

### Key Crisis Windows for Tail-MAPE

| Event | Period | Expected Impact |
|---|---|---|
| Texas Winter Storm Uri | 2021-02-13 → 2021-02-20 | Extreme demand spike in ERCO |
| Russia-Ukraine Invasion | 2022-02-24 → 2022-04-01 | Gas price shock → grid dispatch shift |
| Summer 2023 Heatwave | 2023-07 → 2023-08 | Demand spikes across southern BAs |
| Red Sea Crisis | 2024-01 → 2024-03 | Oil supply chain disruption |

### Expected Outcomes

- **Normal periods**: Model A ≈ Model B (VSN suppresses GDELT)
- **Crisis periods**: Model B >> Model A (GDELT captures anomaly)
- **Overall MAPE improvement**: 0.3–1.5 percentage points

---

## 5. XGBoost Spike Classifier (Companion Model)

TFT handles continuous demand forecasting. For **binary spike detection**, XGBoost on tabular features remains optimal:

```python
SPIKE_CLASSIFIER_CONFIG = {
    "features": [
        # All demand lags + rolling stats
        # Weather features (especially HDD/CDD extremes)
        # GDELT crisis indicators
        # TFT's own prediction residuals (2nd-stage stacking)
    ],
    "target": "is_spike",           # demand > 2σ from 24h rolling mean
    "model": "XGBClassifier",
    "hyperparameters": {
        "max_depth": 6,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "scale_pos_weight": 10,     # Class imbalance: spikes are rare (~2-5%)
    },
}
```

### 2-Stage Stacking

```
Stage 1: TFT produces 24h demand forecast + quantile intervals
    ↓
Stage 2: XGBoost consumes TFT's predictions + residuals + original features
    → Binary: "Will a spike occur in the next 24h?"
    → Probability: Spike confidence score
```

---

## 6. Interpretability Analysis

TFT's built-in interpretability provides three types of analysis:

### 6.1 Variable Importance (from VSN weights)
- Shows per-feature contribution globally and per-BA
- Expected: temperature > demand_lag_24h > hour_of_day > ... > GDELT features
- **Key finding**: Which BAs show GDELT sensitivity (likely gas-dependent ones like ERCO)

### 6.2 Temporal Attention Patterns
- Shows which past timesteps the model attends to for each prediction
- Expected: strong attention at t-1, t-24, t-168 (1h, 1d, 1w ago)
- **During crises**: attention shifts to GDELT-enriched timesteps

### 6.3 Per-BA Feature Decomposition
- TFT can decompose each prediction into feature contributions
- Enables: "For ERCO on 2022-02-25, 40% of the prediction was driven by gas_pct, 25% by temperature, 15% by geo_risk_index"

---

## 7. Proposed File Structure

```
src/models/
├── config.py              [NEW] — Hyperparameter configs (TFT_CONFIG, TRAINER_CONFIG, etc.)
├── dataset.py             [NEW] — TimeSeriesDataSet builder from analytics.features
├── train_tft.py           [NEW] — TFT training pipeline with early stopping
├── train_xgb_spike.py     [NEW] — XGBoost spike classifier
├── evaluate.py            [NEW] — Evaluation metrics, tail-MAPE, coverage
├── interpret.py           [NEW] — VSN weights, attention heatmaps, feature importance
├── optimize.py            [NEW] — Optuna hyperparameter search with walk-forward CV
└── predict.py             [NEW] — Inference pipeline for new data
```

---

## 8. Hardware Requirements

| Resource | Requirement | Your Hardware |
|---|---|---|
| GPU VRAM | ~4-6 GB (FP16 mixed) | RTX 4060 (8GB) or RTX 5060 ✅ |
| RAM | ~8-12 GB (pandas + dataloader) | 16GB ✅ |
| Disk | ~500 MB (checkpoints + logs) | Within 10GB budget ✅ |
| Training Time | ~2-4 hours per model | Acceptable |
| Optuna Search | ~12-24 hours (20 trials × 3 folds) | Run overnight |

---

## Verification Plan

### Automated Tests
1. **Data integrity check**: After loading `analytics.features` into `TimeSeriesDataSet`, verify:
   - No NaN values in target column
   - `time_idx` is monotonically increasing per BA
   - All 25 BAs present
   - Date range matches expected span
   ```powershell
   & d:\Projects\SENTINEL\venv\Scripts\python.exe -m pytest src/models/tests/test_dataset.py -v
   ```

2. **Training smoke test**: Train for 2 epochs on 1 BA (ERCO) with small batch:
   ```powershell
   & d:\Projects\SENTINEL\venv\Scripts\python.exe -m src.models.train_tft --smoke-test
   ```

3. **Prediction shape check**: Verify output shape is `[batch, 24, 7]` (24 hours × 7 quantiles)

### Manual Verification
1. After full training, inspect TFT interpretability plots:
   - Variable importance bar chart — confirm temperature and demand lags rank highest
   - Attention heatmap during Texas freeze — confirm model attends to GDELT features
2. Compare Model A vs Model B MAPE on the test set
3. Verify prediction intervals: ~80% of actual values should fall within [q10, q90]
