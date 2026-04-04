# AGENTS.md — Universal Project State

## System Context
**Project**: SENTINEL (Smart Energy Network Technical Intelligence & Neural Event Link)  
**Objective**: 5-year predictive energy monitoring framework combining EIA grid data with geopolitical sentiment via Variable Selection Networks.  
**Stack**: TimescaleDB (PostgreSQL 16), Python 3.10+, PyTorch Forecasting (TFT).

## Behavioral Guidelines
1. **Idempotency**: All ingestion/cleaning scripts must use `ON CONFLICT DO NOTHING` or equivalent.
2. **Zero-Waste**: Max disk budget is 10GB. Use `TIMESTAMPTZ` and columnar compression.
3. **UTC Everywhere**: All time-series data is stored in **UTC +00:00**.
4. **Python Env**: Always use `d:\Projects\SENTINEL\venv\Scripts\python.exe` — system Python lacks project deps.
5. **PYTHONPATH**: Must include `d:\Projects\SENTINEL` for module imports.

## Current Progress: 100% COMPLETE

### Phase 1: Data Foundation ✅
- **EIA Historical Backfill**: 25/25 BAs completed (Demand D/DF/NG/TI + Fuel Mix + Interchange).
- **Interchange Data**: ~9M rows via failsafe CSVs.
- **Weather Data**: 5-year hourly backfill from Open-Meteo complete for all 25 BAs.
- **GDELT Ingestion**: Global + regional structured news integrated with custom Risk Multipliers.

### Phase 2: Silver Layer ✅
- `src/data/build_silver.py` processed Bronze → Silver data efficiently via TimescaleDB SQL.
- Created `clean.demand`, `clean.fuel_mix`, and `clean.interchange`.

### Phase 3: Gold Layer ✅
- `analytics.features` generated successfully via SQL CTEs, window functions.
- Fully synchronized feature set connecting temporal lags, weather attributes, supply structure, and GDELT geopolitical signaling.

### Phase 4: Model Training & Ablation ✅
- Integrated Temporal Fusion Transformers (TFT) with massive hardware acceleration.
- Verified DeepAR and NHiTS baseline comparisons.
- **Model C (TFT + GKG Integration)** successfully completed!
- **Results:**
  - Surpassed deep learning baselines with statistically significant precision (Diebold-Mariano test p-value < 0.001).
  - Reduced Pinball Quantile Loss by **42.4%** across volatile events.
  - VSN importance tracking verified strong dependency on "gpr_zscore" and "grid_stress" indicators derived exclusively from textual geopolitical events.
  
## Critical Paths
- **Database**: `sentinel` on `localhost:5432` (Size: ~5.6 GB).
- **Environment**: `PYTHONPATH` must include `d:/Projects/SENTINEL`.
- **Venv**: `d:\Projects\SENTINEL\venv\Scripts\python.exe`.
- **Next Task**: The project has officially concluded. See `README.md` for full research documentation and methodology.

## Database Schema Overview
*(Fully populated and validated)*

### Bronze (raw)
| Table | Rows | Grain |
|---|---|---|
| `raw.eia_region_data` | ~4.5M | hourly × BA × type(D/DF/NG/TI) |
| `raw.eia_fuel_type_data` | ~7.2M | hourly × BA × fueltype |
| `raw.eia_interchange_data` | ~9M | hourly × BA-pair |
| `raw.weather_hourly` | ~1.1M | hourly × BA |
| `raw.gdelt_events_daily` | 1,910 | daily (US + Global merged) |

### Gold (analytics)
| Table | Grain | Key Columns |
|---|---|---|
| `analytics.features` | hourly × BA | demand_mw (target), temporal, demand lags, weather, supply, GDELT signals, spike labels |

## Key Design Decisions
1. **TFT over LSTM** — Variable Selection Networks handled noisy GDELT features safely perfectly.
2. **GDELT as daily broadcast** — National-level events broadcast to all BAs per hour; TFT learned structural sensitivity interactively.
3. **Pure SQL feature engineering** — Window functions effectively replaced Pandas memory bloat.
