# AGENTS.md — Universal Project State

## System Context
**Project**: SENTINEL (Smart Energy Network Technical Intelligence & Neural Event Link)  
**Objective**: 5-year predictive energy monitoring framework combining EIA grid data with geopolitical sentiment.  
**Stack**: TimescaleDB (PostgreSQL 16), Python 3.10+, PyTorch Forecasting (TFT).
**Hardware**: NVIDIA RTX 5060 (8GB VRAM), 32GB RAM.

## Behavioral Guidelines
1. **Idempotency**: All ingestion/cleaning scripts must use `ON CONFLICT DO NOTHING` or equivalent.
2. **Zero-Waste**: Max disk budget is 10GB. Use `TIMESTAMPTZ` and columnar compression.
3. **UTC Everywhere**: All time-series data is stored in **UTC +00:00**.
4. **Python Env**: `c:\Users\sriha\OneDrive\Attachments\Documents\Desktop\Projects\Sentinel\venv\Scripts\python.exe`.
5. **PYTHONPATH**: Must include `c:\Users\sriha\OneDrive\Attachments\Documents\Desktop\Projects\Sentinel` for module imports.

## Current Progress

### Phase 1: Data Foundation ✅ COMPLETE
- [x] **EIA Historical Backfill**: 25/25 BAs completed (Demand D/DF/NG/TI + Fuel Mix + Interchange).
- [x] **Interchange Data**: ~9M rows via failsafe CSVs.
- [x] **Weather Data**: 5-year hourly backfill from Open-Meteo complete for all 25 BAs.
- [x] **GDELT Ingestion**: 1,910 rows loaded into `raw.gdelt_events_daily` (US + Global merged, 2021-01-01 → 2026-03-25).

### Phase 2: Silver Layer ✅ COMPLETE
- [x] `src/data/build_silver.py` created — SQL-based pivot of Bronze → Silver.
- [x] `clean.demand` populated.
- [x] `clean.fuel_mix` populated.

### Phase 3: Gold Layer ✅ COMPLETE
- [x] `src/features/build_features.py` created — pure SQL with CTEs + window functions.
- [x] Executed — produced `analytics.features` (hourly × per-BA) with demand lags, weather, GDELT signals.
- [x] Backfilled Gas Price and Nuclear Outage features.

### Phase 4: Model Training — IN PROGRESS
- [x] TFT (Temporal Fusion Transformer) architecture defined (`TFT_ARCHITECTURE.md`).
- [x] Model pipeline implemented in `src/models/`.
- [🔄] Environment setup and dependency installation.
- [ ] Ablation study: Model A (EIA+Weather) vs Model B (EIA+Weather+GDELT).
- [ ] XGBoost spike classifier.

## Critical Paths
- **Database**: `sentinel` on `localhost:5432` (Size: ~5.6 GB).
- **Transfer**: See `db_transfer_guide.md` for `pg_dump` instructions.
- **Environment**: `PYTHONPATH` must include `c:/Users/sriha/OneDrive/Attachments/Documents/Desktop/Projects/Sentinel`.
- **Venv**: `c:\Users\sriha\OneDrive\Attachments\Documents\Desktop\Projects\Sentinel\venv\Scripts\python.exe`.
- **Next Task**: Run smoke test for TFT model.

## Database Schema Overview

### Bronze (raw)
| Table | Rows | Grain |
|---|---|---|
| `raw.eia_region_data` | ~4.5M | hourly × BA × type(D/DF/NG/TI) |
| `raw.eia_fuel_type_data` | ~7.2M | hourly × BA × fueltype |
| `raw.eia_interchange_data` | ~9M | hourly × BA-pair |
| `raw.weather_hourly` | ~1.1M | hourly × BA |
| `raw.gdelt_events_daily` | 1,910 | daily (US + Global merged) |

### Silver (clean)
| Table | Grain | Key Columns |
|---|---|---|
| `clean.demand` | hourly × BA | demand_mw, forecast_mw, generation_mw, interchange_mw |
| `clean.fuel_mix` | hourly × BA | coal/gas/nuclear/oil/solar/hydro/wind_mw, gas_pct, renewable_pct |

### Gold (analytics)
| Table | Grain | Key Columns |
|---|---|---|
| `analytics.features` | hourly × BA | demand_mw (target), temporal, demand lags, weather, supply, GDELT signals, spike labels |

## Key Design Decisions
1. **TFT over LSTM** — Variable Selection Networks handle noisy GDELT features safely.
2. **GDELT as daily broadcast** — National-level events broadcast to all BAs per hour; TFT learns per-BA sensitivity via fuel mix interaction.
3. **Pure SQL feature engineering** — Window functions for lags/rolling stats run inside TimescaleDB, no pandas overhead.
4. **geo_risk_index formula**: `(severe_conflict/event_count)*10 + (oil_region_events/global_events)*10`.
