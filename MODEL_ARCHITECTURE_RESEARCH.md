# SENTINEL — Model Architecture Research: What Does the Literature Actually Say?

> Sources: Google Scholar, ResearchGate, MDPI, IEEE, arXiv (2023–2025 papers)

---

## First: Why CNN for Time Series Isn't Crazy

Your instinct is right — **2D CNNs are for images**. But we're not using 2D CNNs. We're using **1D CNNs**, which are a completely different operation:

| | 2D CNN (Images) | 1D CNN (Time Series) |
|---|---|---|
| **Input** | Height × Width × Channels (e.g., 224×224×3) | Timesteps × Features (e.g., 168×15) |
| **Filter slides** | Across 2D spatial grid | Across time axis only |
| **Detects** | Edges, textures, shapes | Short-term temporal patterns, motifs |
| **Example** | "This is a cat" | "Demand rises sharply in a 3-hour window before spikes" |

A 1D CNN sliding a kernel of size 3 over a time series is like saying: **"Look at every 3-hour window and find recurring patterns."** This is well-established in the literature — not a hack.

However — is it the **best** choice? Let's see what the papers say.

---

## The 5 Competing Architectures (From Literature)

### 1. CNN-LSTM (What We Originally Planned)

**How it works:** 1D-CNN extracts local temporal features → feeds into LSTM for sequential modeling

**Published evidence:**
- 2024 MDPI paper: CNN-LSTM with attention for day-ahead demand forecasting in Australia. Achieved **significant MAPE reduction** vs standalone models.
- 2024 Dergipark paper: CNN-LSTM outperformed standalone CNN and LSTM for multi-source energy prediction using hourly data from Turkey (2018–2023). **Lowest RMSE, MAE, highest R²**.
- 2024 IEEE/SPIE paper: CNN-Attention-LSTM for short-term load forecasting — outperformed traditional models.
- 2025 MDPI paper: CNN-LSTM for city-level electrical load estimation — lower MAPE than standalone LSTM.

**Verdict:** ✅ Well-validated. Consistently beats standalone models. But is it the *best*?

---

### 2. Temporal Fusion Transformer (TFT) — The Emerging Leader

**How it works:** Transformer architecture specifically designed for time series. Uses multi-head attention, gating layers, and variable selection networks.

**Published evidence:**
- 2024 arXiv paper: TFT outperformed LSTM, TCN for daily/weekly/monthly energy consumption.
- 2024 MDPI paper: TFT outperformed ARIMA, LSTM, MLP, and XGBoost for PV power forecasting.
- 2024 arXiv paper: TFT showed **significant improvements over LSTM for week-ahead forecasting**, especially at substation level.
- Key advantage: **Built-in interpretability** — tells you WHICH features mattered for each prediction.

**Why TFT might be better for SENTINEL:**

| Advantage | Why It Matters for Us |
|---|---|
| **Multi-horizon forecasting** | We predict 24 hours ahead — TFT natively handles multi-step output |
| **Variable selection** | Automatically learns which of our 15+ features matter (demand, weather, gas price, news sentiment) |
| **Static + dynamic inputs** | Can handle BA-specific static features (fuel mix) alongside time-varying features (hourly demand) |
| **Interpretability** | For a research paper, being able to say "the model weighted gas price 3x higher during this crisis" is gold |
| **Probabilistic forecasting** | Outputs confidence intervals natively (quantile loss), not just point predictions |

**VRAM requirement:** ~4–6 GB for training (fits your RTX 4060 and RTX 5060)

**Verdict:** ⭐ **Strong candidate to replace or complement CNN-LSTM for our use case.**

---

### 3. Temporal Convolutional Network (TCN)

**How it works:** Uses dilated causal convolutions to achieve large receptive fields without recurrence. Processes the full sequence in parallel (not step-by-step like LSTM).

**Published evidence:**
- 2024 MDPI paper: TCN achieved **lower WAPE than best LSTM model** for national electric demand.
- 2024 ResearchGate paper: TCN outperformed LSTM for EV charging station demand prediction.
- Key advantage: **Faster training** than LSTM (parallel processing, no sequential dependency).
- 2024 ResearchGate paper: TCN-LSTM hybrid outperformed standalone TCN for medium-term prediction.

**Why TCN is interesting for SENTINEL:**

| Advantage | Why It Matters |
|---|---|
| **Faster training** | With a 5-day sprint, training speed matters |
| **No vanishing gradient** | Can handle very long sequences (our 168h window) efficiently |
| **Parallelizable** | Uses full GPU utilization better than LSTM |

**Verdict:** ✅ Viable alternative. Especially good as a component in a hybrid model.

---

### 4. Standalone LSTM / BiLSTM

**Published evidence:**
- Still the most-cited architecture in energy forecasting papers (2020–2024)
- 2024 papers show it's competitive for **short-term, short-input** forecasting
- But consistently beaten by hybrids (CNN-LSTM, TCN-LSTM) and Transformers for complex multi-feature scenarios

**Verdict:** ✅ Good baseline, but not state-of-the-art alone for multi-source data.

---

### 5. xLSTM (Extended LSTM — New in 2024)

**How it works:** A 2024 redesign of LSTM with exponential gating and matrix memory. Claims to match Transformer performance with lower memory consumption.

**Published evidence:**
- 2024 arXiv paper: Comparable performance to Transformers for load forecasting, with **lower memory consumption** (important for your 8GB VRAM)
- Very new — limited energy-specific papers yet

**Verdict:** ⚠️ Promising but too new for a research paper that needs established baselines.

---

## Head-to-Head Comparison for SENTINEL

| Criteria | CNN-LSTM | TFT | TCN | LSTM | xLSTM |
|---|---|---|---|---|---|
| **Multi-source features** (15+ inputs) | Good | **Best** (variable selection) | Good | OK | Good |
| **Multi-horizon output** (24h) | OK (needs decoder) | **Built-in** | OK | OK | OK |
| **Interpretability** (for research) | Poor | **Excellent** | Poor | Poor | Poor |
| **Training speed** | Moderate | Moderate | **Fast** | Slow | Moderate |
| **VRAM usage** (fit in 8GB) | ✅ 4–6 GB | ✅ 4–6 GB | ✅ 2–4 GB | ✅ 3–5 GB | ✅ 3–5 GB |
| **Probabilistic output** | No (needs extra work) | **Yes** (quantile loss) | No | No | No |
| **Published energy papers** | Many (100+) | Growing (50+) | Growing (30+) | Many (500+) | Few (<10) |
| **Handles static features** (BA fuel mix) | No | **Yes** (native) | No | No | No |
| **Spike detection** | Good | Good | Good | Good | Unknown |
| **Novelty for research** | Low (well-explored) | **Medium** | Medium | None | High (too new) |

---

## My Revised Recommendation

Based on the literature, here's what I now propose for SENTINEL:

### Primary Forecaster: **Temporal Fusion Transformer (TFT)**

**Replace CNN-LSTM with TFT as the main demand forecasting model.**

Why:
1. **Handles our multi-source data natively** — static BA features (fuel mix) + time-varying (demand, weather, prices, sentiment) without manual feature engineering
2. **Built-in interpretability** — for a research paper, this is a massive advantage. You can show attention heatmaps of which features drove each prediction
3. **Probabilistic forecasts** — outputs prediction intervals, not just point estimates
4. **Multi-horizon** — predicts all 24 hours simultaneously, no autoregressive rollout needed
5. **Available in PyTorch**: `pytorch-forecasting` library has a production-ready TFT implementation

### Keep: **XGBoost for Spike Classification**

XGBoost remains the best choice for the binary classification task (spike vs. no-spike). Literature consistently supports gradient boosting for tabular classification with engineered features.

### Keep: **Prophet for Baseline**

Prophet remains the comparison benchmark — it's the "can your model beat a simple decomposition approach?" test.

### Optional Add: **TCN as Ablation/Comparison**

Include TCN as an ablation study model — "we compared TFT vs TCN vs LSTM and found TFT outperformed on our multi-source dataset" strengthens the paper.

---

## Updated Model Architecture

```
                     ALL FEATURES (6 EIA tables + weather + news)
                                    │
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
        ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
        │   XGBoost    │   │     TFT      │   │  Rule-Based  │
        │              │   │  (Temporal    │   │  Cascading   │
        │  Spike       │   │   Fusion     │   │  Engine      │
        │  Classifier  │   │  Transformer)│   │              │
        ├──────────────┤   ├──────────────┤   ├──────────────┤
        │ Binary:      │   │ 24h demand   │   │ Domain       │
        │ spike or not │   │ forecast +   │   │ knowledge:   │
        │              │   │ confidence   │   │ oil→gas→grid │
        │ Tabular      │   │ intervals    │   │              │
        │ features     │   │              │   │ Handles rare │
        │              │   │ Interpetable │   │ events ML    │
        │              │   │ attention    │   │ can't learn  │
        └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
               │                  │                   │
               └──────────────────┼───────────────────┘
                                  ▼
                         ┌────────────────┐
                         │  ENSEMBLE      │
                         │  Final output  │
                         └────────────────┘

        Comparison models (ablation study):
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │   Prophet    │  │  LSTM        │  │  TCN         │
        │  (baseline)  │  │  (baseline)  │  │  (baseline)  │
        └──────────────┘  └──────────────┘  └──────────────┘
```

---

## Key Papers to Cite

| Paper | Year | Key Finding | Source |
|---|---|---|---|
| TFT for multi-horizon energy forecasting | 2024 | TFT outperforms LSTM for week-ahead at substation level | arXiv |
| CNN-LSTM with attention for day-ahead demand | 2024 | CNN-LSTM-Attention beats traditional models | MDPI Energies |
| TCN vs LSTM for national electric demand | 2024 | TCN achieves lower WAPE than best LSTM | MDPI |
| CNN-LSTM for multi-source energy prediction | 2024 | Hybrid outperforms standalone on hourly data | Dergipark |
| TFT for PV power forecasting | 2024 | TFT beats ARIMA, LSTM, MLP, XGBoost | MDPI |
| xLSTM vs Transformer for load forecasting | 2024 | Comparable performance, lower memory | arXiv |
| LSTM + LightGBM vs TCN + LightGBM | 2024 | LSTM hybrid better overall, TCN hybrid better for peak loads | ResearchGate |
| Geopolitical uncertainty → electricity prices | 2024 | Russia-Ukraine war aggravated electricity price increases | SciTePress |

---

## Impact on SENTINEL's 5-Day Plan

| Change | Effect on Timeline |
|---|---|
| Replace CNN-LSTM with TFT | **Saves time** — `pytorch-forecasting` has TFT ready to use; less custom architecture code |
| Add TCN as comparison model | Adds ~2–3 hours on Day 4 (Person B) |
| Add LSTM as comparison model | Already planned |
| Interpretability analysis | Adds ~1–2 hours on Day 5 but hugely strengthens the paper |

The TFT is arguably **easier** to implement than a custom CNN-LSTM because `pytorch-forecasting` handles the data loading, training loop, and prediction logic. Person B just needs to define the dataset and hyperparameters.
