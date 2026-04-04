# Regional Data Mapping — Feature Engineering Plan

## Goal
Join all three data sources (EIA, Weather, GDELT) into a single model-ready feature table at **hourly × per-BA** granularity, stored in `analytics.features`.

## Data Source Summary

| Source | Granularity | Key | Rows | Status |
|---|---|---|---|---|
| EIA Interchange | Hourly × BA-pair | `period, fromba, toba` | 9M | ✅ In DB |
| EIA Region (Demand) | Hourly × BA | `period, respondent, type` | 4.5M | ✅ In DB |
| EIA Fuel Type | Hourly × BA | `period, respondent` | 7.2M | ✅ In DB |
| Weather | Hourly × BA | `period, ba_code` | ~1.1M | 🔄 Backfilling |
| GDELT Events | **Daily × National** | `SQLDATE` | 1,911 | ✅ CSV ready |

## The Mapping Challenge

```
EIA + Weather  →  Hourly × 25 BAs  →  PERFECT ALIGNMENT ✅
GDELT Events   →  Daily × 1 National →  MISALIGNMENT ❌
```

GDELT has **two mismatches**: daily (not hourly) and national (not per-BA).

## Proposed Join Architecture

### Step 1: EIA Pivot (Bronze → Silver)
Pivot `raw.eia_region_data` from long-format (`type` column = D/DF/NG/TI) into wide-format:

```
clean.demand:
  period (hourly) | ba_code | demand_mw | forecast_mw | generation_mw | interchange_mw
```

### Step 2: Weather Join (1:1 match)
Direct inner join on `(period, ba_code)` — same granularity:

```
clean.demand ⟕ raw.weather_hourly ON (period, ba_code)
```

### Step 3: GDELT Expansion (Daily → Hourly, National → Per-BA)

**Time expansion**: Convert daily SQLDATE to a date, then join to each hour of that day via:
```sql
ON DATE(eia.period) = gdelt.event_date
```
All 24 hours of a day get the same GDELT features (valid — geopolitical events don't vary hourly).

**Regional strategy**: GDELT stays **national** (same values for all BAs within the same hour).

> [!NOTE]
> This is correct. A sanctions event affects the entire US energy grid, not just one BA. The TFT model learns per-BA sensitivity to national shocks through the interaction of GDELT features with BA-specific demand patterns. Texas (ERCO, gas-dependent) reacts differently to a gas pipeline event than Washington (BPAT, hydro-dominated) — the model captures this via attention heads, not via pre-filtering the data.

### Step 4: Lag Features
Add temporal context so the model sees history:

| Feature | Lag Windows |
|---|---|
| `demand_mw` | 1h, 24h, 168h (1 week) |
| `temperature_2m` | 1h, 24h |
| `avg_goldstein` | 1d, 3d, 7d rolling avg |
| `crisis_event_count` | 1d, 3d, 7d rolling sum |

### Final Output Schema: `analytics.features`

| Column | Source | Type |
|---|---|---|
| `period` | EIA | TIMESTAMPTZ (PK) |
| `ba_code` | EIA | VARCHAR(10) (PK) |
| `demand_mw` | EIA region | FLOAT |
| `forecast_mw` | EIA region | FLOAT |
| `generation_mw` | EIA region | FLOAT |
| `interchange_mw` | EIA region | FLOAT |
| `temperature_2m` | Weather | FLOAT |
| `humidity_2m` | Weather | FLOAT |
| `wind_speed_10m` | Weather | FLOAT |
| `cloud_cover` | Weather | FLOAT |
| `shortwave_radiation` | Weather | FLOAT |
| `precipitation` | Weather | FLOAT |
| `avg_goldstein` | GDELT (national) | FLOAT |
| `min_goldstein` | GDELT (national) | FLOAT |
| `std_goldstein` | GDELT (national) | FLOAT |
| `avg_tone` | GDELT (national) | FLOAT |
| `event_count` | GDELT (national) | INT |
| `severe_conflict_count` | GDELT (national) | INT |
| `crisis_event_count` | GDELT (national) | INT |
| `demand_lag_1h` | Derived | FLOAT |
| `demand_lag_24h` | Derived | FLOAT |
| `demand_lag_168h` | Derived | FLOAT |
| `temp_lag_24h` | Derived | FLOAT |
| `goldstein_3d_avg` | Derived | FLOAT |
| `crisis_7d_sum` | Derived | INT |
| `hour_of_day` | Derived | INT (0-23) |
| `day_of_week` | Derived | INT (0-6) |
| `month` | Derived | INT (1-12) |

**Total: ~1.1M rows × 28 columns per BA = ~27.5M rows for all 25 BAs**

## Implementation Files

| File | Purpose |
|---|---|
| [NEW] `src/features/build_features.py` | Main pipeline: reads Bronze, produces Gold |
| [MODIFY] `src/database/init.sql` | Add `analytics.features` table DDL |
| [NEW] `src/ingestion/load_gdelt.py` | Ingest GDELT CSV into `raw.gdelt_events_daily` |

## Verification Plan
- Row count: `analytics.features` should have ~25 × 43,800 = ~1.1M rows per BA
- NULL check: no NULLs in core columns (demand, temperature, goldstein)
- Date range: 2021-01-01 to 2026-03-23 for all 25 BAs
- Spot-check: Feb 2021 Texas freeze should show temperature drop + demand spike + goldstein drop
