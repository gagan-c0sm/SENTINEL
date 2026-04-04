# SENTINEL ⚡
> **S**upply **E**nergy **N**etwork **T**hreat **I**dentification and **N**ational **E**arly-warning **L**ayer

![SENTINEL Banner](https://img.shields.io/badge/SENTINEL-v1.0--Complete-brightgreen?style=for-the-badge&logo=pytorch)
![Accuracy](https://img.shields.io/badge/Avg._MAPE-2.94%25-blue?style=for-the-badge)
![Coverage](https://img.shields.io/badge/96%25_Interval_Coverage-88.2%25-orange?style=for-the-badge)

SENTINEL is an advanced, deep-learning powered predictive energy monitoring framework. By fusing massive multi-dimensional sequences of U.S. electrical grid behavior (EIA data) with high-density global geopolitical conflict indicators (GDELT), SENTINEL pioneers a robust **Temporal Fusion Transformer (TFT)** architecture designed to anticipate structural grid volatility *before* it manifests in weather or seasonal cycles.

---

## 🌟 Research Breakthrough: The GKG Advantage

Conventional deep learning models forecast electrical demand using strictly temporal and meteorological covariates. SENTINEL integrates the **GKG (Global Knowledge Graph)** to construct geopolitical structural covariates that capture systemic risks hidden from traditional sensors.

### 📊 Performance Benchmark (Model C vs. Baselines)
Extensive validation across **12 U.S. Balancing Authorities** over a 3-month rolling horizon demonstrates the structural superiority of the GKG-infused architecture.

| Feature | SENTINEL (Model C) | DeepAR / NHiTS |
|:---|:---:|:---:|
| **Avg. MAPE** | **2.94%** | ~4.5% - 6.0% |
| **P-Value (DM Test)** | **< 0.001** | Reference |
| **Uncertainty Variance** | **42.4% Reduction** | Baseline |
| **Crisis Resilience** | **High** | Reactive |

### 🧠 Interpretable Variable Selection
SENTINEL provides explainability through its Variable Selection Network (VSN). During macro geopolitical crises (e.g., supply chain shocks), the model automatically escalates the attention weights of `gpr_zscore` (Geopolitical Risk) over standard temporal signals.

![VSN Importance Weight](results/gkg_ablation/erco_iran_crisis_vsn.png)
*Variable Selection Network weights isolating structural covariates during global shock.*

---

## 🏗️ Technical Architecture

1. **Ingestion Layer**: Asynchronous ELT engines processing over 50 million historic EIA rows into **TimescaleDB**.
2. **Feature Store**: Native SQL aggregations generating complex temporal lags and GKG-linked metadata.
3. **Model Engine**: GPU-accelerated **Temporal Fusion Transformer** (TFT) with custom gradient weighting for high-volatility events.
4. **Validation Suite**: Formal statistical verification (Diebold-Mariano, Wilcoxon) and probabilistic coverage testing.

---

## 📁 Project Portfolio

*   **[Final Evaluation Results](docs/EVALUATION_RESULTS.md)** — Comprehensive metrics for all 12 BAs.
*   **[Model C Rectification](docs/MODEL_C_RECTIFICATION.md)** — Deep dive into the GKG integration logic.
*   **[Project Methodology](PROJECT_CONTEXT.md)** — Detailed mapping of the 3-phase development cycle.

---

## 🚀 Deployment Status
**SENTINEL officially marks 100% completion of its primary phases.** It has achieved all objectives relating to grid data synthesis, GDELT integration, multi-horizon ablation benchmarking, and formalized statistical verification.

---

## 📖 Quick Start (Inference Only)
To execute inference on the validated Model C weights:
```bash
# Activate environment
venv\Scripts\activate

# Run evaluation pipeline
python -m src.models.evaluate
```

*(Note: The `tools/` folder and raw `checkpoints/` are excluded from Version Control to maintain repository performance.)*
