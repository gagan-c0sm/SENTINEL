# SENTINEL — Project Context & Decisions Log

> **Purpose:** This file captures all architectural decisions, constraints, and context from the research/planning phase so that any new conversation instance can resume work with minimal information loss.

---

## 1. Project Identity

- **Name:** SENTINEL — **S**upply **E**nergy **N**etwork **T**hreat **I**dentification and **N**ational **E**arly-warning **L**ayer
- **Scope:** National-level (all 65 U.S. Balancing Authorities), 24-hour prediction horizon
- **Type:** Research project with scalability potential
- **Primary data source:** EIA Open Data API (user has API key)
- **Historical window:** 5 years (2021–2026)

---

## 2. Hardware & Compute Constraints

| Component | Spec |
|---|---|
| **RAM** | 16 GB LPDDR5x |
| **GPU** | NVIDIA RTX 4060 (8 GB VRAM) |
| **OS** | Windows |

### Implications for Design

- **All training runs locally** — no cloud compute needed
- **Mixed precision training** (`torch.amp` / `fp16`) required to fit CNN-LSTM in 8 GB VRAM
- **Gradient accumulation** for simulating larger batch sizes
- **Data chunking** — process per-BA or use `dask` if 16 GB RAM is tight with full dataset
- **TimescaleDB/SQLite on disk** — don't hold entire dataset in RAM

### Estimated Training Times (on this hardware)

| Model | Time |
|---|---|
| XGBoost spike classifier | 3–10 min |
| Prophet (per BA) | ~30 sec × 65 BAs = ~30 min |
| LSTM (168h window, 3M rows) | 2–4 hours |
| CNN-LSTM hybrid | 4–6 hours |
| DistilBERT fine-tune | 1–2 hours |

### Training vs Inference

- Training is one-time (then periodic retraining)
- **Inference is near-instant:** full pipeline prediction in < 2 seconds on CPU
- Retraining schedule: XGBoost/LSTM monthly/quarterly, NLP every 3–6 months
- Incremental retraining (warm-start) reduces time to 30–60 min

---

## 3. Data Architecture Decisions

### Ingestion Strategy: Batch + Micro-Batch

| Source | Type | Frequency |
|---|---|---|
| EIA API (historical backfill) | Batch | One-time |
| EIA API (ongoing) | Micro-batch | Every 1–2 hours |
| NOAA / Open-Meteo (historical) | Batch | One-time |
| NOAA / Open-Meteo (ongoing) | Micro-batch | Every 1–6 hours |
| News RSS / APIs | Micro-batch | Every 15–30 min |
| EIA supplementary (prices) | Batch | Daily/weekly |

**No real-time streaming** (no Kafka/Flink). EIA has ~1-day lag; micro-batch with Airflow/Prefect is sufficient.

### ETL vs ELT: **ELT**

- Load raw API responses as-is into staging
- Transform in-place (SQL/pandas) for feature engineering
- Raw data preserved for reprocessing when features evolve

### OLAP vs OLTP: **OLAP-dominant (~90%)**

- Analytical queries (time-range scans, aggregations, ML training reads) dominate
- Minimal OLTP: inserting new hourly data, logging predictions/alerts

### Storage: Three-Layer Architecture (Bronze → Silver → Gold)

```
RAW (Bronze)          CLEAN (Silver)         ANALYTICS (Gold)
─────────────         ──────────────         ────────────────
Raw JSON/API          Validated,             Feature-engineered,
responses stored      deduplicated,          model-ready tables
as-is for replay      typed, indexed         (demand+weather+lags+sentiment)
```

### Database: **TimescaleDB (PostgreSQL extension)**

- Time-series optimized (hypertables, compression, continuous aggregates)
- Handles ~50–100M rows on local hardware
- Single instance for both clean + analytics layers
- No cloud data warehouse needed (Snowflake/BigQuery = overkill)
- Model artifacts stored on local filesystem

---

## 4. Tech Stack Summary

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| ML | PyTorch, scikit-learn, Prophet |
| NLP | Hugging Face Transformers (DistilBERT/FinBERT) |
| Data Pipeline | Airflow or Prefect |
| Database | TimescaleDB (PostgreSQL) |
| Blockchain | Polygon PoS + Solidity (or Hyperledger Fabric) |
| Dashboard | Streamlit / Plotly Dash (prototype) |
| API | FastAPI |
| Containers | Docker + Docker Compose |

---

## 5. Key Architectural Decisions & Rationale

| Decision | Choice | Why |
|---|---|---|
| No real-time streaming | Batch + micro-batch | EIA data has 1-day lag |
| ELT over ETL | ELT | Raw data preservation for research flexibility |
| TimescaleDB over full DW | TimescaleDB | Right-sized, free, local, time-series optimized |
| OLAP-dominant | ~90% analytical queries | ML training and aggregation workloads dominate |
| Bronze/Silver/Gold layers | Three-layer | Standard data lakehouse pattern for research |
| Local-only compute | No cloud | RTX 4060 + 16GB RAM handles everything |
| Blockchain for accountability | Prediction hashing | Not just logging — proves forecast integrity |

---

## 6. Research Novelty (What Makes This Publishable)

1. **Tier 3 predictions** — NLP-driven demand disruption forecasting (no prior work at BA-level scale)
2. **Cascading effects** — End-to-end modeling of global events → fuel prices → generation shifts → grid stress
3. **Cross-BA network risk propagation** — Graph-based stress propagation using interchange data
4. **Blockchain prediction accountability** — On-chain hashing for verifiable, tamper-proof forecasting
5. **National-level multi-BA scope** — Most studies cover single utilities; this covers all 65 BAs

---

## 7. Project Phases (16 weeks)

| Phase | Weeks | Focus |
|---|---|---|
| 1. Data Foundation | 1–3 | Ingestion pipelines, DB schema, cleaning |
| 2. Pattern Analysis | 4–6 | STL, clustering, anomaly detection, Prophet baseline |
| 3. Advanced Models | 7–9 | XGBoost, LSTM, CNN-LSTM, evaluation |
| 4. NLP & Cascading | 10–12 | News pipeline, sentiment, dependency graph, simulation |
| 5. Blockchain | 13–14 | Smart contracts, prediction hashing, integration |
| 6. Dashboard | 15–16 | Visualization, alerts, end-to-end testing |

---

## 8. Related Documents in This Directory

| File | Contents |
|---|---|
| [feasibility_analysis.md](file:///d:/Projects/SENTINEL/feasibility_analysis.md) | EIA data availability assessment, feasibility by framework stage |
| [implementation_plan.md](file:///d:/Projects/SENTINEL/implementation_plan.md) | Full system architecture, module breakdown, tech stack, verification plan |
| [research_depth_analysis.md](file:///d:/Projects/SENTINEL/research_depth_analysis.md) | Prediction tiers, context layers, causality depth analysis, limitations |
| [task.md](file:///d:/Projects/SENTINEL/task.md) | Phase-level task tracker with checkboxes |
| [Research_Depth_Analysis.html](file:///d:/Projects/SENTINEL/Research_Depth_Analysis.html) | Styled HTML of research depth analysis (printable to PDF) |
| **PROJECT_CONTEXT.md** (this file) | All architectural decisions, hardware constraints, and rationale |

---

## 9. Current Status

- **Phase:** Research & planning complete. Ready to begin Phase 1 (Data Foundation).
- **No code written yet.** All files are documentation/planning.
- **Next step:** Set up project structure, install dependencies, build EIA data ingestion pipeline.
