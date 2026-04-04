# SENTINEL ⚡
> **S**upply **E**nergy **N**etwork **T**hreat **I**dentification and **N**ational **E**arly-warning **L**ayer

![SENTINEL Arch](docs/images/arch.png) <!-- Replace with actual banner if available -->

SENTINEL is an advanced, deep-learning powered predictive energy monitoring framework. By fusing massive multi-dimensional sequences of U.S. electrical grid behavior (EIA data) with high-density global geopolitical conflict indicators (GDELT), SENTINEL pioneers a robust Temporal Fusion Transformer (TFT) architecture designed to anticipate structural grid volatility *before* it manifests in weather or seasonal cycles.

---

## 🌟 Key Research Breakthroughs

Conventional deep learning models forecast electrical demand using strictly temporal and meteorological covariates. SENTINEL integrates the **GKG (Global Knowledge Graph)** to construct geopolitical structural covariates. 

Formal statistical evaluation verifies the architectural advantage of the GKG covariates during macro geopolitical crises (e.g., the Hormuz Crisis model slice):
*   **42.4% Reduction in Probabilistic Uncertainty**: Pinball Quantile Loss dropped from 516.6 (Baseline) to 297.6.
*   **Strict Predictive Superiority**: Outperformed NHiTS and DeepAR robustly across both parametric Diebold-Mariano ($p < 0.001$) and non-parametric Wilcoxon Signed-Rank ($p < 0.05$) evaluations.
*   **Sub-5% MAPE Validation**: Maintained exceptionally coherent mean absolute percentage errors on dynamic sliding windows across 12 independent U.S. Balancing Authorities simultaneously.

### 🧠 Interpretable Variable Selection
SENTINEL doesn't just predict; it provides explainability. The Variable Selection Network (VSN) inherent in our engineered TFT proves definitively that out of 40 simultaneous signals, `gpr_zscore` (Geopolitical Risk) and `grid_stress` indexes dominated attention weights during global systemic shocks, explicitly isolating the impact of international unrest on domestic energy infrastructure.

![VSN Importance Weight](results/gkg_ablation/erco_iran_crisis_vsn.png)
*Variable Selection Network weights isolating structural covariates during global shock.*

---

## 🏗️ Architecture Stack

1. **Ingestion Pipeline**: 
    - Pure ELT TimescaleDB pipelines seamlessly processing over 50 million historic EIA rows.
    - Zero-waste automated `ON CONFLICT DO NOTHING` idempotency.
2. **Feature Store**: 
    - Database-native aggregation (SQL CTEs & Window Functions) generating complex temporal lags without Python Pandas bottlenecking.
3. **Model Engine**: 
    - **PyTorch Forecasting (TFT)** natively GPU-accelerated.
    - Custom gradient weighting and batch-sizing specifically sculpted to fit 8GB VRAM budgets while running continuous inference on millions of datapoints.

---

## 🚀 Quick Start (Inference)

```bash
# 1. Clone the repository
git clone <repo-url>
cd SENTINEL

# 2. Activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Connect to your initialized TimescaleDB
copy .env.example .env

# 5. Execute DB-Backed Evaluation
python -m src.models.evaluate
```

---

## 📁 Repository Structure

```text
SENTINEL/
├── src/
│   ├── config/              # Model and Database bindings
│   ├── database/            # Schema generation and SQL views
│   ├── ingestion/           # Asynchronous ELT engines
│   ├── features/            # TimescaleDB feature synthesis
│   └── models/              # PyTorch TFT architectures and ablation metrics
├── checkpoints/             # Trained artifact weights
├── results/                 # Automated output generations and generated graphs
├── docs/                    # Research documentation and methodology mapping
├── docker-compose.yml       # TimescaleDB container infrastructure
└── README.md
```

*(Note: Experimental tooling scripts and large local DB volumes are deliberately detached from Version Control to maintain repo cleanliness).*

---

## 📖 Complete Documentation & State
To explore the exact methodology mapping, hardware limitations addressed, and end-state variables of the system, review the contextual metadata files:
- `PROJECT_CONTEXT.md`
- `AGENTS.md`
- `SCHEMA.md`

## 🤝 Project Completion
**SENTINEL officially marks 100% completion of its primary phases.** It has achieved all objectives relating to grid data synthesis, GDELT integration, multi-horizon ablation benchmarking, and formalized statistical verification.
