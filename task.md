# SENTINEL — Task Tracker

## Phase 1: Data Foundation ✅ COMPLETE
- [x] Set up project structure and dependencies
- [x] Docker + TimescaleDB setup
- [x] Build EIA API data ingestion pipeline (5 years, 25 BAs)
- [x] Build weather data ingestion (Open-Meteo, 5-year hourly backfill)
- [x] Design and create database schema
- [x] Ingest GDELT CSVs (US + Global → `raw.gdelt_events_daily`, 1,910 rows)

## Phase 2: Silver Layer ✅ COMPLETE
- [x] Create `src/data/build_silver.py`
- [x] `clean.demand` — pivoting EIA region data (D/DF/NG/TI → columns)
- [x] `clean.fuel_mix` — pivoting EIA fuel type data (gas_pct, renewable_pct)
- [x] Verify row counts and NULL checks

## Phase 3: Gold Layer (Feature Engineering) ✅ COMPLETE
- [x] Create `src/features/build_features.py` (pure SQL, month-by-month)
- [x] Execute after Silver completes
- [x] Verify `analytics.features` — demand lags, weather, GDELT, spike labels

## Phase 4: Model Training — NOT STARTED
- [ ] TFT (Temporal Fusion Transformer) — primary demand forecaster
- [ ] XGBoost spike classifier
- [ ] Ablation study: Model A (EIA+Weather) vs Model B (+GDELT)
- [ ] Model evaluation (MAPE, RMSE, tail-event F1)

## Phase 5: NLP & Cascading Effects — NOT STARTED
- [ ] News RSS/API ingestion pipeline
- [ ] NLP sentiment analysis pipeline (FinBERT/DistilBERT)
- [ ] Energy source dependency graph (NetworkX)
- [ ] Cascading effect simulation engine

## Phase 6: Dashboard & Integration — NOT STARTED
- [ ] Streamlit monitoring dashboard
- [ ] Alert visualization
- [ ] End-to-end system integration testing

## Deferred: Blockchain
- [ ] Smart contract for prediction hashing (draft at `src/blockchain/`)
- [ ] Prediction accountability system
