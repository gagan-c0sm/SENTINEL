# CODEBASE_MAP.md — Source Code & Script Reference

> **Purpose:** Complete map of every file in the project for onboarding a new agentic IDE.

---

## Project Root: `d:\Projects\SENTINEL`

### Core Source (`src/`)

| File | Purpose | Status |
|---|---|---|
| `src/__init__.py` | Package root | ✅ |
| **src/config/** | | |
| `src/config/settings.py` | Pydantic settings (EIA key, DB creds, BA list, backfill dates) | ✅ |
| **src/database/** | | |
| `src/database/connection.py` | SQLAlchemy engine factory (`get_engine()`) | ✅ |
| `src/database/init.sql` | Full DDL: all Bronze/Silver/Gold tables, indexes, hypertables, compression policies | ✅ |
| `src/database/setup_nlp_tables.py` | NLP-specific table creation | ✅ |
| **src/ingestion/** | | |
| `src/ingestion/eia_client.py` | EIA API v2 client (paginated fetch) | ✅ |
| `src/ingestion/backfill.py` | Historical EIA data backfill orchestrator | ✅ |
| `src/ingestion/load_csvs.py` | Failsafe CSV upsert for interchange data | ✅ Hardened |
| `src/ingestion/weather_client.py` | Open-Meteo hourly weather client (25 BA coordinates) | ✅ |
| `src/ingestion/backfill_weather.py` | Weather backfill with idempotent staging upsert | ✅ Complete |
| `src/ingestion/load_gdelt.py` | GDELT CSV ingestion (US + Global merge → `raw.gdelt_events_daily`) | ✅ Complete |
| **src/data/** | | |
| `src/data/build_silver.py` | Silver layer: pivots EIA to `clean.demand` + `clean.fuel_mix` | 🔄 Running |
| `src/data/clean.py` | Legacy cleaning script (superseded by `build_silver.py`) | ⚠️ Legacy |
| **src/features/** | | |
| `src/features/build_features.py` | Gold layer: SQL CTEs with window functions → `analytics.features` | ✅ Ready |
| **src/nlp/** | | |
| `src/nlp/resolver.py` | NLP entity resolution utilities | 🚧 Stub |
| `src/nlp/seed_regions.py` | Region mapping seed data | 🚧 Stub |
| **src/blockchain/** | | |
| `src/blockchain/oracle_bridge.py` | Chainlink oracle bridge for on-chain prediction hashing | 🚧 Stub |
| `src/blockchain/contracts/SentinelConsensus.sol` | Solidity smart contract for prediction accountability | 🚧 Draft |
| **src/models/** | Empty — next phase | ⬜ |
| **src/analysis/** | Empty — next phase | ⬜ |
| **src/cascading/** | Empty — next phase | ⬜ |
| **src/dashboard/** | Empty — next phase | ⬜ |

### Root Scripts (temporary/diagnostic)

| File | Purpose | Keep? |
|---|---|---|
| `audit_data.py` | DB integrity audit (row counts, date ranges) | ✅ Useful |
| `force_csvs.py` | Manual CSV force-load for interchange data | ⚠️ One-off |
| `diagnose_interchange.py` | Interchange data debugging | ⚠️ One-off |
| `eda_exploration.py` | EDA / exploratory analysis script | ✅ Useful |
| `probe_bpat.py` | BPAT-specific data probe | ⚠️ One-off |
| `tmp_*.py` | Temporary debug scripts (5 files) | 🗑️ Deletable |

### Data Files

| File | Size | Purpose |
|---|---|---|
| `gdelt.events.csv` | 275KB | US GDELT daily aggregates (16 cols, 1911 rows) |
| `gdelt,global.events.csv` | 309KB | Global GDELT daily aggregates (15 cols, 1911 rows) |
| `.env` | — | API keys and DB credentials |
| `requirements.txt` | — | Python dependencies |
| `docker-compose.yml` | — | TimescaleDB container definition |

---

## Documentation Files (`*.md`)

### Architecture & Design (read these first)
| File | Content |
|---|---|
| `AGENTS.md` | **Universal project state** — current progress, DB schema, critical paths |
| `PROJECT_CONTEXT.md` | Architecture decisions, hardware constraints, tech stack rationale |
| `implementation_plan.md` | Full system design: modules, data flow, prediction pipeline |
| `DATA_MAPPING_PLAN.md` | How EIA + Weather + GDELT join into `analytics.features` |

### Research & Analysis
| File | Content |
|---|---|
| `GDELT_IMPACT_RESEARCH.md` | Proof that GDELT features work with TFT (VSN analysis, ablation design) |
| `GDELT_RESEARCH.md` | GDELT data feasibility analysis |
| `gdelt_research_report.md` | Detailed GDELT event analysis with aggregation strategy |
| `MODEL_ARCHITECTURE_RESEARCH.md` | TFT vs LSTM vs XGBoost comparison |
| `NLP_RESEARCH_PROMPT.md` | NLP pipeline design for news sentiment integration |
| `nlp_integration_architecture.md` | Full NLP → feature pipeline architecture |
| `feasibility_analysis.md` | EIA data availability and feasibility assessment |
| `research_depth_analysis.md` | Prediction tiers, causality depth, limitations |
| `BLOCKCHAIN_DON_ARCHITECTURE.md` | DON (Decentralized Oracle Network) architecture for blockchain |

### Status & Tracking
| File | Content |
|---|---|
| `SCHEMA.md` | Database schema snapshot |
| `STORAGE_LOG.md` | Disk usage tracking |
| `task.md` | Phase-level task tracker |

---

## Execution Commands

```powershell
# All commands use the project venv
$PYTHON = "d:\Projects\SENTINEL\venv\Scripts\python.exe"

# Step 1: GDELT Ingestion (DONE)
& $PYTHON -m src.ingestion.load_gdelt

# Step 2: Silver Layer
& $PYTHON -m src.data.build_silver

# Step 3: Gold Layer (after Silver completes)
& $PYTHON -m src.features.build_features

# Step 4: TFT Training (next phase)
& $PYTHON -m src.models.train_tft
```

## Key Configuration

- **DB**: `postgresql://sentinel:sentinel_dev_2026@localhost:5432/sentinel`
- **25 BAs**: ERCO, PJM, CISO, MISO, NYIS, ISNE, SWPP, SOCO, TVA, DUK, FPL, AECI, AVA, BPAT, LGEE, NEVP, PACE, PACW, PSCO, SC, SCEG, SEC, TAL, TEC, WACM
- **Backfill range**: 2021-01-01 → 2026-03-23
- **GDELT range**: 2021-01-01 → 2026-03-25 (1,910 daily rows)
