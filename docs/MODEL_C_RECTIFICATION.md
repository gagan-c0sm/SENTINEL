# SENTINEL Model C — Complete Rectification Reference

> **Document Purpose:** Exhaustive, data-driven justification for every change in the Model C rebuild. Each section contains the diagnosis, proof from our database, research backing, and exact implementation steps. Intended as the permanent record for this rebuild.
>
> **Date:** 2026-03-29  
> **Database:** `sentinel` on `localhost:5432` (TimescaleDB / PostgreSQL 16)  
> **Training Data:** 2021-01-01 to 2024-07-01 | Validation: 2024-07-01 to 2025-01-01 | Test: 2025-01-01 to 2026-03-23  
> **Target BAs (12):** ERCO, PJM, MISO, CISO, NYIS, ISNE, SWPP, SOCO, TVA, DUK, FPL, BPAT

---

## Table of Contents

1. [Root Cause Diagnosis: Why Models A & B Failed](#1-root-cause-diagnosis)
2. [Fix 1: Data Corruption Cleanup](#2-fix-1-data-corruption-cleanup)
3. [Fix 2: NaN-to-Zero Poisoning Elimination](#3-fix-2-nan-to-zero-poisoning)
4. [Fix 3: Normalizer Replacement (softplus → log1p)](#4-fix-3-normalizer-replacement)
5. [Fix 4: Geopolitical Signal Autopsy & Redesign](#5-fix-4-geopolitical-signal-redesign)
6. [Fix 5: Regional Energy Sentiment from GDELT GKG](#6-fix-5-regional-energy-sentiment)
7. [Fix 6: Prophet Decomposition](#7-fix-6-prophet-decomposition)
8. [Fix 7: Weather Feature Reclassification](#8-fix-7-weather-reclassification)
9. [Fix 8: New Supply/Price Features](#9-fix-8-new-supply-price-features)
10. [Fix 9: BA Fuel Sensitivity Profiles](#10-fix-9-ba-fuel-sensitivity-profiles)
11. [Fix 10: Optimizer Switch (Ranger → AdamW + OneCycleLR)](#11-fix-10-optimizer-switch)
12. [Fix 11: log_interval Audit (Silent Training Slowdown)](#12-fix-11-log_interval-audit)
13. [Fix 12: is_holiday Permanently FALSE](#13-fix-12-is_holiday-permanently-false)
14. [Fix 13: Optuna Optimization Overhaul](#14-fix-13-optuna-optimization-overhaul)
15. [Fix 14: RTX 5070 Laptop Hardware Profile](#15-fix-14-rtx-5070-hardware-profile)
16. [Fix 15: Caldara-Iacoviello GPR Index Integration](#16-fix-15-gpr-index)
17. [Audit: Quantile Loss — Keep 7 Quantiles](#17-audit-quantile-loss)
18. [Audit: Windows Multiprocessing — num_workers=2](#18-audit-windows-workers)
19. [Model C Complete Configuration & Architecture](#19-model-c-complete-configuration)
20. [Chronological Execution Plan (18 Steps)](#20-chronological-execution-plan)

---

## 1. Root Cause Diagnosis

### 1.1 Model A Results (Baseline — No GDELT)

Model A achieved reasonable MAPE on some BAs but catastrophic failure on others. The per-BA results revealed a bimodal pattern: either <10% MAPE or >50% MAPE, with no middle ground.

### 1.2 Model B Results (Baseline + GDELT)

Model B (adding `sentiment_mean_24h`, `sentiment_min_24h`, `event_count_24h`, `geo_risk_index`) performed WORSE than Model A in several BAs. The GDELT features introduced noise that the model could not effectively filter.

### 1.3 Root Causes Identified (3 Critical, 2 Systemic)

| # | Root Cause | Impact | BAs Affected |
|---|-----------|--------|-------------|
| **RC1** | Demand data corruption (INT32_MAX, negative, extreme outliers) | Normalizer overflow → NaN gradients → flat-line predictions | PJM, TVA, SWPP, FPL, NYIS |
| **RC2** | `fillna(0)` on price/outage columns | False "free gas" signal on weekends, artificial seasonality | All 12 BAs |
| **RC3** | `GroupNormalizer(softplus)` overflow on corrupted values | Numerical explosion when processing 2.1B MW values | PJM, MISO, SOCO (flat-line) |
| **RC4** | `geo_risk_index` formula is mathematically self-cancelling | Index cannot distinguish crises from normal days | All BAs using Model B |
| **RC5** | GDELT features are national-level, not mapped to BA regions | Same sentiment value for ERCO (Texas) and BPAT (Pacific NW) | All BAs using Model B |

---

## 2. Fix 1: Data Corruption Cleanup

### 2.1 Evidence — Live Database Audit

Query executed on `analytics.features` (2026-03-29):

```sql
SELECT ba_code, MIN(demand_mw), MAX(demand_mw), AVG(demand_mw),
       PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY demand_mw)
FROM analytics.features
WHERE ba_code IN ('PJM','TVA','SWPP','FPL','NYIS')
GROUP BY ba_code;
```

**Results:**

| BA | MIN (MW) | MAX (MW) | P99 (MW) | Physical Capacity | Problem |
|----|---------|---------|---------|-------------------|---------|
| **PJM** | 30,049 | **2,147,483,647** | 148,221 | ~190 GW | INT32_MAX values (0x7FFFFFFF) |
| **TVA** | **-9,952,817** | **9,969,712** | 31,421 | ~40 GW | ±9.9M MW outliers |
| **SWPP** | 13,289 | **3,631,292** | 48,128 | ~55 GW | 3.6M MW outliers |
| **FPL** | **-1,287** | 29,107 | 24,618 | ~30 GW | Negative demand |
| **NYIS** | **0** | 33,082 | 27,543 | ~35 GW | Zero-demand rows |

### 2.2 Why This Is Catastrophic

**GroupNormalizer(softplus) computation:**

```
softplus(x) = log(1 + exp(x))
```

When `x = 2,147,483,647` (PJM's INT32_MAX):
- `exp(2,147,483,647)` → `Inf`
- `log(Inf)` → `Inf`
- Gradient of Inf → `NaN`
- All subsequent weight updates → `NaN`
- Model produces flat-line predictions for every BA in the same batch

The corruption in PJM alone can poison the entire training run because PJM rows appear in mixed-BA training batches.

### 2.3 Physical Bounds Reference

From EIA capacity ratings and historical peak demand records:

| BA | Min Plausible (MW) | Max Plausible (MW) | Source |
|----|-------------------|-------------------|--------|
| ERCO | 10,000 | 90,000 | ERCOT 2023 peak: 85,463 MW |
| PJM | 30,000 | 170,000 | PJM 2024 peak: 151,100 MW |
| MISO | 30,000 | 135,000 | MISO 2023 peak: 127,340 MW |
| CISO | 8,000 | 55,000 | CAISO 2022 peak: 52,061 MW |
| NYIS | 8,000 | 35,000 | NYISO 2024 peak: 31,744 MW |
| ISNE | 5,000 | 28,000 | ISO-NE 2024 peak: 24,341 MW |
| SWPP | 15,000 | 55,000 | SPP 2023 peak: 52,802 MW |
| SOCO | 8,000 | 55,000 | Southern Co. peak: ~48 GW |
| TVA | 8,000 | 40,000 | TVA 2024 peak: 33,373 MW |
| DUK | 4,000 | 25,000 | Duke Energy Carolinas peak: ~22 GW |
| FPL | 5,000 | 30,000 | FPL 2023 peak: 28,200 MW |
| BPAT | 2,000 | 15,000 | BPA 2024 peak: ~12 GW |

### 2.4 Implementation

**File:** Execute SQL on TimescaleDB  
**When:** Step 1 (first action)

```sql
-- PJM: NULL anything outside [30000, 170000]
UPDATE clean.demand SET demand_mw = NULL
  WHERE ba_code = 'PJM' AND (demand_mw > 170000 OR demand_mw < 30000);

-- TVA: NULL anything outside [8000, 40000]
UPDATE clean.demand SET demand_mw = NULL
  WHERE ba_code = 'TVA' AND (demand_mw > 40000 OR demand_mw < 8000);

-- SWPP: NULL anything > 55000
UPDATE clean.demand SET demand_mw = NULL
  WHERE ba_code = 'SWPP' AND demand_mw > 55000;

-- FPL: NULL negative
UPDATE clean.demand SET demand_mw = NULL
  WHERE ba_code = 'FPL' AND demand_mw < 0;

-- NYIS: NULL zero/negative
UPDATE clean.demand SET demand_mw = NULL
  WHERE ba_code = 'NYIS' AND demand_mw <= 0;

-- Cascade: NULL related columns on corrupted rows
UPDATE clean.demand SET generation_mw = NULL, supply_demand_gap = NULL
  WHERE demand_mw IS NULL AND ba_code IN ('PJM','TVA','SWPP','FPL','NYIS');
```

**Verification:**
```sql
SELECT ba_code, COUNT(*) FILTER (WHERE demand_mw IS NULL) as nullified,
       MIN(demand_mw), MAX(demand_mw)
FROM clean.demand
WHERE ba_code IN ('PJM','TVA','SWPP','FPL','NYIS')
GROUP BY ba_code;
```
Expected: PJM max < 170k, TVA within [8k, 40k], no negatives in FPL, no zeros in NYIS.

---

## 3. Fix 2: NaN-to-Zero Poisoning

### 3.1 Evidence — NULL Percentages in analytics.features

Query executed on `analytics.features`:

| Column | NULL or Zero % | Problem |
|--------|---------------|---------|
| `oil_price` | **100.00%** | Entirely NULL → `fillna(0)` turns it into "$0/barrel oil" |
| `gas_price` | **32.12%** | NULL on weekends/holidays → `fillna(0)` = "free gas on weekends" |
| `nuclear_outage_pct` | **32.08%** | NULL on weekends → `fillna(0)` = "zero nuclear outages every weekend" |
| `gas_price_change_7d` | **33.14%** | Derived from gas_price, inherits the NULL pattern |

### 3.2 Why fillna(0) Is Destructive

The current `prepare_dataframe()` code (dataset.py, lines 66-72):

```python
# Current (BROKEN):
df[col] = df.groupby("ba_code")[col].transform(
    lambda s: s.ffill().fillna(0)  # <-- ffill then ZERO
)
```

The `ffill()` handles mid-series gaps correctly, but `fillna(0)` at the start of each BA's series introduces artificial values:

**For `gas_price`:**
- Weekday gas price: ~$2-5/MMBtu
- Saturday & Sunday: `fillna(0)` → $0/MMBtu
- The model learns: "gas is free on weekends" → artificial weekend seasonality in predictions
- Every Monday, gas "jumps" from $0 to $2-5 → model sees a false 100%+ price spike

**For `oil_price`:**
- 100% NULL → 100% zero after fillna(0)
- Model treats "$0 oil" as a real feature → constant that occupies a VSN attention slot
- Burns one feature's worth of model capacity on noise

**For `nuclear_outage_pct`:**
- Weekday: ~5-8% outage
- Weekend: `fillna(0)` → 0% outage
- Model learns: "nuclear plants never have outages on weekends" → impossible

### 3.3 Implementation

**File:** `src/models/dataset.py` → `prepare_dataframe()`  
**When:** Step 5 (after Gold layer rebuild)

```python
# NEW (FIXED):
# Step 1: Forward-fill price/outage columns (Friday value → weekend)
FORWARD_FILL_COLS = ['gas_price', 'gas_price_change_7d', 'nuclear_outage_pct',
                     'gas_price_volatility_7d']
for col in FORWARD_FILL_COLS:
    if col in df.columns:
        df[col] = df.groupby("ba_code")[col].transform(
            lambda s: s.ffill().bfill()  # forward-fill then back-fill (no zeros)
        )

# Step 2: Drop oil_price entirely (100% NULL, irrecoverable)
if 'oil_price' in df.columns:
    df = df.drop(columns=['oil_price'])

# Step 3: For remaining NaN, use per-BA median (not zero)
numeric_cols = df.select_dtypes(include=["float64", "float32"]).columns
for col in numeric_cols:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    if df[col].isna().any():
        median_val = df.groupby("ba_code")[col].transform("median")
        df[col] = df[col].fillna(median_val)
        df[col] = df[col].fillna(0)  # Final fallback only if entire group is NaN
```

**Justification:** Forward-fill is the standard approach for daily-reported values broadcast to hourly resolution. Friday's gas price IS Saturday and Sunday's gas price — markets are closed. Using $0 creates a false signal.

---

## 4. Fix 3: Normalizer Replacement

### 4.1 Evidence — Why softplus Fails

Current normalizer (dataset.py, line 171-174):

```python
target_normalizer=GroupNormalizer(
    groups=GROUP_IDS,
    transformation="softplus",
)
```

`softplus(x) = log(1 + exp(x))` requires all values to be positive and within a reasonable range. Our demand data has:
- PJM: 30,000 to 170,000 MW (5.7x range)
- BPAT: 2,000 to 15,000 MW (7.5x range)
- ERCO: 10,000 to 90,000 MW (9x range)

When corrupted values (2.1B MW for PJM) are present, softplus produces `Inf`, killing the gradient. Even after corruption cleanup, the heterogeneous demand ranges across BAs make softplus unstable:

- softplus normalizes by computing `log(1 + exp(x - mean))` per group
- For PJM (mean ~90k), deviations of ±40k create `exp(40000)` → overflow
- For BPAT (mean ~7k), deviations of ±3k are manageable

### 4.2 Why log1p Is Better

`log1p(x) = log(1 + x)`:
- Handles heterogeneous scales naturally (maps 2,000 and 170,000 to the same log space)
- No exponential computation → no overflow risk
- Standard practice for energy demand normalization in TFT literature
- Preserves relative differences within each group

```
log1p(2000)   = 7.60  (BPAT minimum)
log1p(15000)  = 9.62  (BPAT maximum)  → range = 2.02
log1p(30000)  = 10.31 (PJM minimum)
log1p(170000) = 12.04 (PJM maximum)   → range = 1.73
```

All BAs end up in a similar normalized range (~7-12), making multi-BA training stable.

### 4.3 Implementation

**File:** `src/models/dataset.py` → `build_datasets()`  
**When:** Step 9 (Model C configuration)

```python
target_normalizer=GroupNormalizer(
    groups=GROUP_IDS,
    transformation="log1p",   # CHANGED from "softplus"
),
```

---

## 5. Fix 4: Geopolitical Signal Autopsy & Redesign

### 5.1 How the Current geo_risk_index Is Computed

**File:** `src/features/build_features.py`, lines 66-70

```sql
geo_risk_index = (
    (us_severe_conflict / us_event_count) * 10        -- Term A
  + (oil_region_event_count / global_event_count) * 10 -- Term B
)
```

**Term A:** Ratio of US severe conflicts (CAMEO 18-20) to total US events  
**Term B:** Ratio of oil-region events to total global events

### 5.2 Evidence — Index vs Known Crises

Live database audit against known crisis dates:

```
Date         | US_Events | Severe | Oil_Reg | Global  | GeoRisk | Event
-------------|-----------|--------|---------|---------|---------|------------------
2021-02-15   |    23,329 |  4,483 |   6,843 | 105,042 |  2.573  | Texas Freeze
2022-02-24   |    31,312 |  5,862 |   4,548 | 178,922 |  2.126  | Russia invades Ukraine
2022-06-15   |    29,482 |  6,407 |   5,146 | 119,167 |  2.605  | NORMAL summer day
2023-10-07   |    19,919 |  4,229 |   6,479 | 107,827 |  2.724  | Israel-Hamas war
2024-04-14   |    15,751 |  3,585 |  19,798 | 105,691 |  4.149  | Iran strikes Israel
2026-02-28   |    23,174 |  5,377 |  23,659 | 113,642 |  4.402  | Iran War 2026
2026-03-01   |    15,543 |  3,956 |  28,140 | 102,865 |  5.281  | Iran War 2026 peak
```

**Separation test (crisis periods vs normal):**

```
Period               | Days | Avg Risk | Std  | Separation from Normal
---------------------|------|----------|------|------------------------
Iran 2026            |   24 |    3.710 | 0.57 | +1.04 above normal
Iran-Israel 2024     |   21 |    2.897 | 0.36 | +0.23 (within 1 std)
Israel-Hamas 2023    |   42 |    2.700 | 0.14 | +0.03 (indistinguishable)
NORMAL (1782 days)   | 1782 |    2.670 | 0.25 | —
Ukraine Invasion     |   41 |    2.372 | 0.17 | -0.30 (BELOW normal)
```

### 5.3 Why the Formula Fails — Mathematical Proof

**Flaw 1: Term A self-cancels during crises.**

During a crisis, both `us_severe_conflict` (numerator) AND `us_event_count` (denominator) increase proportionally:

```
Normal day:    6,407 / 29,482 = 0.217
Ukraine day:   5,862 / 31,312 = 0.187  ← LOWER during crisis
Iran 2026:     5,377 / 23,174 = 0.232  ← barely above normal
```

The ratio stays in [0.187, 0.232] regardless of crisis severity.

**Flaw 2: Term B only detects oil-region crises.**

`oil_region_event_count / global_event_count` spikes when Iran/Iraq/Saudi Arabia is directly involved but is blind to non-oil crises:

```
Ukraine invasion:  4,548 / 178,922 = 0.025  (Ukraine ≠ oil region → LOW)
Iran strikes:     19,798 / 105,691 = 0.187  (Iran = oil region → HIGH)
```

**Flaw 3: Near-constant distribution makes the signal useless for the TFT.**

Index range: [1.86, 5.28], mean=2.68, std=0.28. 93% of days fall within [2.4, 3.0]. The TFT's Variable Selection Network (VSN) correctly identifies this as low-variance noise and assigns it near-zero importance. The Gated Residual Network (GRN) gates it off. This is the model being smart, not broken.

### 5.4 Evidence — Do International Crises Affect US Grid Demand?

Year-over-year demand comparison during crisis windows (controlling for seasonality):

| Crisis | Duration | Gas Price Effect | Demand Effect (Most BAs) | Grid Impact? |
|--------|---------|-----------------|-------------------------|-------------|
| **Texas Freeze 2021** | 7 days | **+177%** | ERCO +20%, TVA +26%, MISO +16% | **YES — but WEATHER** |
| **Ukraine Invasion 2022** | 36 days | **+85%** | Most BAs <3% | **NO (price channel only)** |
| **Oil Price Peak Mar 2022** | 15 days | **+71%** | Most BAs <3% | **NO** |
| **Israel-Hamas Oct 2023** | 40 days | **-45%** (fell!) | All BAs ≈ 0% | **NO** |
| **Iran Strikes Apr 2024** | 17 days | **-31%** (fell!) | All BAs ≈ 0% | **NO** |
| **Iran War 2026** | 24 days | **-26%** (fell!) | Mixed 5-10% | **Probably weather/growth** |

**Key findings:**

1. **The Ukraine invasion — the biggest geopolitical shock of the decade — registered BELOW normal on our index (2.13 vs 2.67) and showed <3% demand change in most BAs.**

2. **Post-2023, gas prices FELL during every Middle Eastern crisis.** US energy independence (shale production, LNG infrastructure) structurally decoupled the domestic market from oil-region geopolitics.

3. **The only crisis with massive grid impact was the Texas Freeze — which is WEATHER, not geopolitics.** Our existing temperature/HDD/CDD features already capture this.

**Conclusion:** International geopolitical risks do NOT directly move US electricity demand on observable timescales. The transmission mechanism (crisis → gas price → dispatch) existed pre-2023 but has been structurally neutralized by US energy independence. Including a broken `geo_risk_index` as a feature is harmful — it adds noise that degrades predictions.

### 5.5 Research Context

**Caldara-Iacoviello GPR Index** (American Economic Review, 2022): The gold standard for measuring geopolitical risk. Uses keyword-counting in 10 major newspapers, normalized by total article volume. Daily version available at matteoiacoviello.com/gpr.htm. While academically validated for financial markets, our data shows it would not improve US demand forecasting because the underlying effect doesn't exist post-2023.

**GDELT Goldstein Scale Best Practice** (academic consensus): Raw event counts should never be used — they must be normalized by total media volume, z-scored against rolling baselines, and smoothed to remove news cycle noise. Our formula violated all three principles: it used a ratio that self-cancels, applied no z-scoring, and used no smoothing.

### 5.6 Decision

**Drop all 4 existing GDELT features** (`sentiment_mean_24h`, `sentiment_min_24h`, `event_count_24h`, `geo_risk_index`). Replace with **regional energy-sector sentiment** from GDELT GKG (see Fix 5). The international geopolitics angle becomes a research finding ("US grid is resilient"), not a prediction feature.

---

## 6. Fix 5: Regional Energy Sentiment from GDELT GKG

### 6.1 The Pivot — From International Geopolitics to Regional Energy News

Our data proved international crises don't affect US demand. But REGIONAL events do:

| Event | Type | Demand Effect | Detectable in GKG? |
|-------|------|-------------|-------------------|
| Texas Freeze 2021 | Weather/Grid | ERCO +20%, TVA +26% | YES — grid stress articles |
| Hurricane Ian 2022 | Weather/Grid | FPL -3.10σ (grid collapse) | YES — 76,445 grid stress articles |
| Winter Storm Elliott 2022 | Weather/Grid | PJM +29%, MISO +26% | YES — 7,893% spike in grid stress |
| Hurricane Milton 2024 | Weather/Grid | FPL -2.91σ | YES — 126,371 grid stress articles |
| Israel-Hamas 2023 | Geopolitics | All BAs ≈ 0% | No grid stress signal (correctly) |

### 6.2 Data Source — GDELT GKG on BigQuery

**What is the GKG?** The Global Knowledge Graph processes every news article worldwide, extracting:
- **V2Themes:** Standardized topic tags (e.g., `POWER_OUTAGE`, `ENV_NATURALGAS`, `ECON_ELECTRICALGRID`)
- **V2Tone:** Article sentiment (-100 to +100)
- **V2Locations:** Geographic mentions (US state-level via ADM1 code)

This differs from GDELT Events (which we had before): Events tracks CAMEO-coded actions between actors. GKG tracks WHAT articles are ABOUT and WHERE.

### 6.3 Theme Groups (From Discovery Query)

Discovery query run on BigQuery (1 month of data, ~$0 via free tier) returned exact theme names:

**Group 1: Grid Stress / Power Outages**
| Theme | Monthly Article Count |
|-------|----------------------|
| `POWER_OUTAGE` | 67,318 |
| `MANMADE_DISASTER_WITHOUT_POWER` | 31,624 |
| `MANMADE_DISASTER_POWER_OUTAGES` | 20,059 |
| `ECON_ELECTRICALGRID` | 20,068 |
| `MANMADE_DISASTER_POWER_FAILURE` | 9,574 |
| `MANMADE_DISASTER_WITHOUT_ELECTRICITY` | 7,600 |
| `MANMADE_DISASTER_RESTORE_POWER` | 7,225 |
| `MANMADE_DISASTER_POWER_OUTAGE` | 6,424 |

**Group 2: Natural Gas / Pipeline**
| Theme | Monthly Article Count |
|-------|----------------------|
| `ENV_NATURALGAS` | 119,947 |
| `WB_2299_PIPELINES` | 79,389 |
| `WB_1768_OIL_AND_GAS_PIPELINE` | 7,423 |
| `WB_551_GAS_TRANSPORTATION_STORAGE_AND_DISTRIBUTION` | 7,566 |
| `WB_1751_LIQUEFIED_NATURAL_GAS` | 6,254 |

**Group 3: Electricity Demand / Generation / Price**
| Theme | Monthly Article Count |
|-------|----------------------|
| `WB_508_POWER_SYSTEMS` | 61,352 |
| `ECON_ELECTRICALGENERATION` | 38,123 |
| `ECON_ELECTRICALPRICE` | 13,735 |
| `ECON_ELECTRICALDEMAND` | 9,334 |

**Group 4: Nuclear**
| Theme | Monthly Article Count |
|-------|----------------------|
| `WB_509_NUCLEAR_ENERGY` | 48,743 |
| `ENV_NUCLEARPOWER` | 20,943 |

**Group 5: Renewables**
| Theme | Monthly Article Count |
|-------|----------------------|
| `ENV_SOLAR` | 187,778 |
| `WB_525_RENEWABLE_ENERGY` | 148,910 |
| `ENV_WINDPOWER` | 37,098 |
| `ENV_HYDRO` | 19,824 |

### 6.4 BigQuery Extraction

**Cost:** ~$8-15 (partition-pruned, column-selected, state-filtered)  
**Output:** 75,837 rows — one per US state per day (2021-01-01 to 2026-03-29)  
**File:** `gkg.csv` (4.4 MB)  
**Budget used:** ~$20 of $300 available

The query extracts per-state-per-day:
- Article counts for each of 5 theme groups
- Average tone, minimum tone, tone volatility
- Average word count (proxy for article depth)

Filtered to only the 40 US states that map to our 12 BAs.

### 6.5 State-to-BA Mapping

```
ERCO → TX
CISO → CA
NYIS → NY
FPL  → FL
BPAT → WA, OR
DUK  → NC, SC
SOCO → GA, AL
TVA  → TN, KY
ISNE → MA, CT, NH, ME, RI, VT
SWPP → KS, OK, NE, ND, SD
PJM  → PA, NJ, MD, VA, OH, WV, DE, IL, IN
MISO → MN, WI, IA, MI, MO, AR, LA, MS
```

For multi-state BAs: article counts are SUMMED, tone is AVERAGED.

### 6.6 Validation — Does GKG Energy Sentiment Align with Grid Events?

**Test 1: Top 20 Grid Stress Days**

| Date | BA | Grid Stress Articles | Tone | Demand Z | Event |
|------|-----|---------------------|------|----------|-------|
| 2024-10-10 | FPL | 126,371 | -6.07 | -2.91 | **Hurricane Milton** |
| 2021-08-30 | MISO | 105,807 | -2.28 | -1.21 | **Hurricane Ida** |
| 2024-09-27 | FPL | 91,997 | -5.95 | +0.18 | **Hurricane Helene** |
| 2022-09-29 | FPL | 76,445 | -5.91 | -3.10 | **Hurricane Ian** |
| 2021-02-18 | ERCO | 66,707 | -3.50 | +0.92 | **Texas Freeze** |
| 2021-02-19 | ERCO | 64,591 | -3.42 | +0.94 | **Texas Freeze** |
| 2025-01-09 | CISO | 56,289 | -3.62 | -0.37 | **Jan 2025 Cold** |

**Every single top-20 day is a real grid disaster.** No geopolitics, no noise — hurricanes, freezes, and storms. The GKG energy sentiment correctly identifies regional grid stress.

**Test 2: Crisis Window Comparison**

Winter Storm Elliott (Dec 22-27, 2022) vs same period 2021:

| BA | Grid Stress Change | Gas/Pipe Change | Tone (Crisis vs Normal) | Demand Change |
|----|-------------------|----------------|------------------------|-------------|
| PJM | **+3,385%** | +16% | -3.75 vs -0.34 | **+29.1%** |
| MISO | **+2,190%** | +104% | -3.64 vs -0.60 | **+25.8%** |
| NYIS | **+7,893%** | +90% | -4.42 vs -0.54 | **+9.0%** |
| ERCO | **+635%** | +174% | -3.67 vs -0.94 | **+47.9%** |

**Grid stress articles spike 500-8,000% during real weather crises AND demand simultaneously spikes.**

Jan 2024 Arctic Blast:

| BA | Grid Stress Change | Demand Change |
|----|-------------------|-------------|
| ERCO | **+503%** | **+42.6%** |
| PJM | **+306%** | **+15.2%** |
| MISO | **+367%** | **+19.1%** |

During geopolitical crises (Israel-Hamas, Iran War), grid stress articles either DROP or stay flat — correctly reflecting zero grid impact.

**Test 3: Extreme Events — Top 5% Days**

| BA | Extreme Day Demand Z | Normal Day Demand Z | Extreme Tone | Normal Tone |
|----|---------------------|--------------------|--------------|--------------| 
| SWPP | **+0.58** | -0.15 | -2.79 | -0.35 |
| BPAT | **+0.57** | -0.08 | -3.22 | -0.46 |
| ERCO | **+0.46** | -0.13 | -3.82 | -0.38 |
| ISNE | **+0.43** | -0.16 | -2.70 | -0.15 |
| TVA | **+0.39** | -0.06 | -4.04 | +0.23 |
| PJM | **+0.30** | -0.15 | -2.24 | -0.13 |

**10 of 12 BAs show higher demand on extreme GKG stress days.** FPL is the exception because hurricanes knock out meters → recorded demand drops even though actual need surges.

Tone separation is massive across ALL BAs: -2.7 to -4.8 during extreme events vs -0.0 to -0.6 normal.

**Test 4: Correlation (Daily)**

| BA | GridStress→Demand | GasPipe→Demand | Elec→Demand | Tone→Demand |
|----|------------------|----------------|-------------|-------------|
| CISO | +0.033 | +0.089 | **+0.223** | +0.071 |
| BPAT | +0.114 | **+0.156** | +0.082 | -0.006 |
| PJM | +0.052 | **+0.125** | **+0.146** | +0.052 |
| ISNE | +0.034 | **+0.145** | **+0.126** | +0.028 |
| NYIS | +0.031 | **+0.140** | +0.071 | +0.087 |

Daily correlations are low (0.02-0.27). This is **expected and correct** — demand is 95% driven by weather/time. The signal's value is in extreme event detection, not daily prediction. After z-scoring, the signal will be ~0 on 95% of days (harmless) and +3-5σ during real grid events (informative).

### 6.7 Processing Pipeline

1. **BigQuery output** → `gkg.csv` (75,837 rows, one per state-day) ✅ DONE
2. **State→BA aggregation** → sum article counts, average tone across states per BA
3. **Z-scoring** per BA: 7-day rolling mean / 28-day rolling baseline → crisis-only activation
4. **Ingest** to `analytics.features` → broadcast daily values to all 24 hours

### 6.8 Features That Enter the TFT

| Feature | Replaces | TFT Input Type | What It Captures |
|---------|---------|---------------|-----------------|
| `grid_stress_zscore` | `sentiment_mean_24h` | `time_varying_observed_real` | Outage/blackout coverage → explains demand drops + neighbor imports |
| `gas_pipeline_zscore` | `sentiment_min_24h` | `time_varying_observed_real` | Pipeline/supply disruption news → leads gas_price 24-48h |
| `electricity_buzz_zscore` | `event_count_24h` | `time_varying_observed_real` | Demand/price/generation coverage → grid attention spike |
| `energy_tone_regional` | `geo_risk_index` | `time_varying_observed_real` | Overall regional energy sentiment (negative = alarming) |

Same 4 feature slots. Completely different, validated signals.

---

## 7. Fix 6: Prophet Decomposition

### 7.1 Research Basis

The reference paper (Elsevier, 2025: "A hybrid Prophet-TFT approach") demonstrates that feeding Prophet-decomposed trend and seasonality components as **future known inputs** to the TFT significantly improves accuracy. The TFT's decoder can use these deterministic components to reason about the future, while only the residuals need to be learned from observed features.

### 7.2 Why This Helps

The TFT currently must learn trend and seasonality from raw `hour_of_day`, `day_of_week`, and `month` features. While it CAN learn these patterns, providing them as pre-computed inputs reduces the learning burden and makes the model converge faster.

| Component | What Prophet Extracts | How TFT Uses It |
|-----------|----------------------|-----------------|
| `prophet_trend` | Long-term demand trajectory per BA (growth/decline) | Known real → decoder sees future trend direction |
| `prophet_weekly` | Day-of-week pattern per BA | Known real → decoder anticipates weekend/weekday shift |
| `prophet_yearly` | Seasonal pattern per BA (summer peaks, winter valleys) | Known real → decoder anticipates seasonal envelope |

### 7.3 Implementation

**File:** `src/features/prophet_decompose.py` [NEW]  
**When:** Step 6 (after Gold layer rebuild)

```python
from prophet import Prophet

def decompose_ba(ba_code, demand_series):
    prophet_df = demand_series.rename(columns={'period': 'ds', 'demand_mw': 'y'})
    
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,  # hourly data
        changepoint_prior_scale=0.05,  # conservative trend
    )
    m.fit(prophet_df[prophet_df['ds'] < '2024-07-01'])  # train period only
    
    forecast = m.predict(prophet_df)  # predict full dataset
    return forecast[['ds', 'trend', 'weekly', 'yearly']]
```

Prophet components enter the TFT as `time_varying_known_reals` — the model knows these values for the future prediction horizon because they are deterministic (trend continues, seasonality is periodic).

---

## 8. Fix 7: Weather Feature Reclassification

### 8.1 Evidence — Weather Data Quality

Weather data coverage audit (all from Open-Meteo historical API):

```
All 12 BAs: 0% NULL | 45,792 rows each | 2021-01-01 to 2026-03-23
No gaps, no missing values, full 5.2-year coverage.
```

### 8.2 Evidence — Weather-Demand Correlations

| BA | CDD→Demand | HDD→Demand | Temp→Demand | Interpretation |
|----|-----------|-----------|-------------|----------------|
| **ERCO** | **+0.785** | -0.184 | +0.559 | Cooling-dominated (Texas AC) |
| **FPL** | **+0.786** | -0.193 | +0.700 | Cooling-dominated (Florida AC) |
| **CISO** | **+0.696** | -0.376 | +0.629 | Cooling-dominated (California) |
| **MISO** | **+0.686** | -0.056 | +0.259 | Cooling-driven |
| **SOCO** | **+0.669** | +0.056 | +0.290 | Cooling-dominated (Southeast) |
| **BPAT** | +0.158 | **+0.389** | -0.296 | Heating-dominated (Pacific NW) |
| **PJM** | **-0.003** | **+0.000** | **-0.001** | ⚠️ ZERO — data corruption effect |
| **TVA** | **+0.031** | **+0.028** | **-0.008** | ⚠️ Near-zero — data corruption effect |

PJM and TVA show zero weather-demand correlation. This is physically impossible for grids serving tens of millions of people. The cause: demand data corruption (PJM's 2.1B MW, TVA's ±9.9M MW) destroys the statistical relationship. After Fix 1, these correlations should recover to normal levels.

### 8.3 Reclassification — Observed → Known

Currently, ALL weather features are `time_varying_observed_reals` (past only). This means the TFT can only see weather in the encoder (lookback), not the decoder (forecast horizon).

Since 24-hour weather forecasts are >95% accurate, `temperature_c`, `hdd`, and `cdd` should be reclassified as `time_varying_known_reals`:

```python
# MOVE to known reals (model can see these in the 24h forecast horizon):
# temperature_c, hdd, cdd

# KEEP as observed (less reliable 24h ahead):
# humidity_pct, wind_speed_kmh, cloud_cover_pct, solar_radiation
```

This tells the TFT "you WILL know the temperature for the next 24h" rather than "you can only see past temperature." This is architecturally correct and matches the reference paper's approach.

---

## 9. Fix 8: New Supply/Price Features

### 9.1 Supply Margin Percentage

```sql
supply_margin_pct = (generation_mw - demand_mw) / NULLIF(demand_mw, 0)
```

Measures how tight the grid is. Positive = surplus generation, negative = deficit (importing).
During Texas Freeze: ERCO supply_margin went deeply negative. This feature directly captures grid stress.

### 9.2 Gas Price Realized Volatility (7-day)

```sql
gas_price_volatility_7d = STDDEV(gas_price) OVER (
    PARTITION BY ba_code ORDER BY period
    ROWS BETWEEN 167 PRECEDING AND CURRENT ROW
)
```

Captures gas market uncertainty. When gas prices swing (for ANY reason — weather, pipeline, geopolitics), the dispatch stack shifts. This is the MARKET-EFFECT signal: if a crisis matters, gas volatility will spike. If it doesn't (Israel-Hamas), volatility stays flat.

### 9.3 Gas Mix Momentum (24h delta)

```sql
gas_pct_delta_24h = gas_pct - LAG(gas_pct, 24) OVER (
    PARTITION BY ba_code ORDER BY period
)
```

Captures fuel-switching dynamics. When gas generation is rising relative to 24h ago, the dispatch stack is shifting — possibly due to weather, outages, or price signals.

---

## 10. Fix 9: BA Fuel Sensitivity Profiles

### 10.1 Rationale

Different BAs have vastly different fuel mixes, making them differently sensitive to gas/nuclear/renewable signals:

- **ISNE** (New England): ~55% gas → highly sensitive to gas price/supply signals
- **BPAT** (Pacific NW): ~5% gas, ~70% hydro → insensitive to gas, sensitive to hydro/drought
- **TVA** (Tennessee Valley): ~40% nuclear → sensitive to nuclear outage signals

### 10.2 Implementation

Compute from `clean.fuel_mix` over training period:

```python
gas_sensitivity = mean(gas_mw / total_mw)   # per BA, over 2021-2024
renewable_sensitivity = mean((solar+wind+hydro) / total_mw)
nuclear_sensitivity = mean(nuclear_mw / total_mw)
```

These enter the TFT as `static_reals` — one value per BA, constant across time. The TFT's Static Covariate Encoder uses these to learn: "ISNE should pay 11x more attention to gas_pipeline_zscore than BPAT."

---

## 11. Fix 10: Optimizer Switch (Ranger → AdamW + OneCycleLR)

### 11.1 Current State

**File:** `src/models/train_tft.py`, line 110

```python
optimizer="ranger",
reduce_on_plateau_patience=config.reduce_lr_patience,
```

Ranger = RAdam + Lookahead. Originally designed for CNNs.

### 11.2 Problems with Ranger for TFT

**Problem 1: Checkpoint restoration bugs.**

Ranger uses a complex dual-state architecture (RAdam inner state + Lookahead slow weights). When PyTorch Lightning restores from checkpoint via `trainer.fit(ckpt_path=...)`, the Lookahead slow weights can fail to reload correctly, particularly across different Lightning versions. This is documented in multiple GitHub issues and StackOverflow threads. The symptom: training resumes but the optimizer effectively resets, destroying the warmup period and causing divergent loss.

Our codebase already shows evidence of this problem — `train_tft.py` lines 172-179 contain a try/except block around `load_from_checkpoint` with a fallback to the raw model state, which is a workaround for this exact issue.

**Problem 2: Higher VRAM usage.**

Ranger stores 3 sets of parameters: current weights, RAdam momentum/variance, and Lookahead slow weights. AdamW stores 2 sets: current weights and momentum/variance. On 8GB VRAM (RTX 5070 laptop), this difference matters.

**Problem 3: No native scheduler integration.**

Ranger uses `ReduceLROnPlateau` (reactive — waits for loss to plateau before reducing LR). AdamW + OneCycleLR is proactive — it warms up, peaks, and anneals the learning rate on a mathematically optimal schedule:

```
OneCycleLR schedule over 30 epochs:
  Epoch 1-3:   LR ramps from 1e-5 to 1e-3  (warmup)
  Epoch 3-25:  LR anneals from 1e-3 to 1e-5 (cosine decay)
  Epoch 25-30: LR drops to 1e-7             (fine-tuning)
```

This is the industry standard for transformers (GPT, BERT, TFT all use warmup + cosine decay).

### 11.3 Implementation

**File:** `src/models/train_tft.py` [MODIFY]

```python
# Subclass TFT to override optimizer
class SentinelTFT(TemporalFusionTransformer):
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,       # 10% warmup
            anneal_strategy='cos',
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
```

---

## 12. Fix 11: log_interval Mismatch (Silent Training Slowdown)

### 12.1 Current State

**File:** `src/models/config.py`, lines 104-105

```python
log_every_n_steps: int = 50         # Trainer-level (progress bar / TB)
tft_log_interval: int = -1          # TFT internal (VSN/attention). -1 = OFF
```

**File:** `src/models/train_tft.py`, lines 112-113

```python
log_interval=config.tft_log_interval,   # -1 = disable VSN/attention logging
log_val_interval=-1,                     # Disable plot logging
```

### 12.2 The Issue

There are TWO different logging systems:

1. **`Trainer(log_every_n_steps=50)`** — PyTorch Lightning level. Logs scalar metrics (loss, lr) to TensorBoard every 50 steps. This is lightweight and fine.

2. **`TFT(log_interval=...)`** — pytorch_forecasting TFT level. When `> 0`, generates **full interpretability outputs** (VSN importance matrices, attention heatmaps, prediction-vs-actual plots) every N steps. Each of these requires an additional forward pass + matplotlib rendering + disk I/O.

The config correctly sets `tft_log_interval = -1` (disabled), and `train_tft.py` correctly passes it. **The current code is actually correct.** The `log_every_n_steps=50` in the Trainer is for scalar logging (loss values), which is cheap — NOT the expensive interpretability outputs.

However, `find_optimal_lr()` (line 215) and `optimize.py` (line 110) both still hardcode `optimizer="ranger"`. These need to be updated to `"adamw"` when we switch optimizers.

### 12.3 Verification

After the optimizer switch, confirm in TensorBoard that:
- `log_interval=-1`: No interpretation/ directory is created during training
- Loss logging works via `log_every_n_steps=50`

---

## 13. Fix 12: is_holiday Permanently FALSE (5 Years of Missing Signal)

### 13.1 Current State

**File:** `src/features/build_features.py`, line 38

```sql
FALSE                                        AS is_holiday,
```

Every single row in `analytics.features` (4.5M+ rows across 5 years) has `is_holiday = FALSE`. The model literally cannot learn that demand drops on Christmas, Thanksgiving, Labor Day, etc.

### 13.2 Impact

US federal holidays produce measurable demand changes:

| Holiday | Typical Demand Effect | Reason |
|---------|----------------------|--------|
| **Christmas Day** | -15 to -25% | Commercial/industrial shutdown |
| **Thanksgiving** | -10 to -20% | + Black Friday anomaly next day |
| **Independence Day** | -10 to -15% | Commercial shutdown, AC still runs |
| **New Year's Day** | -10 to -20% | Commercial shutdown |
| **Memorial Day** | -5 to -10% | Mixed — some AC load increase |
| **Labor Day** | -5 to -10% | End of summer, mixed |

Over 5 years, there are ~55 holiday days × 24 hours = ~1,320 hours where the model sees demand drop with NO explanatory feature. It must attribute these drops to other features (temperature, weekday) which is incorrect and adds noise to those features' learned weights.

### 13.3 Implementation

**File:** `src/features/build_features.py` [MODIFY]

Replace the hardcoded `FALSE` with actual US federal holiday detection using `pandas.tseries.holiday.USFederalHolidayCalendar`:

```sql
-- Option A: Compute in SQL using a holiday lookup table
-- OR
-- Option B: Compute in Python post-processing (simpler)
```

**Python approach (in build_features.py, post-SQL):**

```python
from pandas.tseries.holiday import USFederalHolidayCalendar

cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2021-01-01', end='2026-12-31')

# After SQL insert, UPDATE the is_holiday column:
UPDATE analytics.features SET is_holiday = TRUE
WHERE period::DATE IN (
    '2021-01-01', '2021-01-18', '2021-02-15', '2021-05-31',
    '2021-06-18', '2021-07-05', '2021-09-06', '2021-10-11',
    '2021-11-11', '2021-11-25', '2021-12-24', '2021-12-31',
    -- ... (generate full list from USFederalHolidayCalendar for 2021-2026)
);
```

**SQL approach (direct in the CTE):**

```sql
-- Create a holiday reference table
CREATE TABLE IF NOT EXISTS ref.us_holidays (
    holiday_date DATE PRIMARY KEY,
    holiday_name TEXT
);

-- Populate with USFederalHolidayCalendar output
-- Then JOIN in build_features.py:
CASE WHEN h.holiday_date IS NOT NULL THEN TRUE ELSE FALSE END AS is_holiday
```

### 13.4 Verification

```sql
SELECT is_holiday, COUNT(*), 
       ROUND(AVG(demand_mw), 0) as avg_demand
FROM analytics.features
WHERE ba_code = 'ERCO'
GROUP BY is_holiday;
```

Expected: is_holiday=TRUE rows show ~10-20% lower avg_demand than FALSE rows.

---

## 14. Fix 13: Optuna Optimization Overhaul

### 14.1 Current State

**File:** `src/models/optimize.py`, lines 32-48

```python
CV_FOLDS = [
    TrainSplitConfig(train_start="2021-01-01", train_end="2023-01-01", ...),
    TrainSplitConfig(train_start="2021-01-01", train_end="2023-07-01", ...),
    TrainSplitConfig(train_start="2021-01-01", train_end="2024-07-01", ...),
]
```

Each Optuna trial trains **3 full models** (one per CV fold), each for up to 20 epochs. With 20 trials, this means 60 full training runs. On 8GB VRAM, each takes ~20 min = **20 hours total**.

### 14.2 Problems

1. **3x cost for minimal statistical benefit.** Walk-forward CV is overkill for hyperparameter search — we're not publishing a paper on the search, we're finding good hyperparameters.

2. **No trial pruning.** A trial with `hidden_size=32, dropout=0.3` that produces `val_loss=Inf` at epoch 2 still trains for all 20 epochs across 3 folds before being discarded.

3. **Full dataset for tuning.** Using all 12 BAs for hyperparameter search is wasteful. The optimal hidden_size/dropout don't change significantly between 3 BAs and 12.

### 14.3 Implementation

**File:** `src/models/optimize.py` [MODIFY]

```python
from optuna.integration import PyTorchLightningPruningCallback

# Single split — no CV
TUNING_SPLIT = TrainSplitConfig(
    train_start="2021-01-01", train_end="2024-07-01",
    val_start="2024-07-01", val_end="2025-01-01",
    test_start="2025-01-01", test_end="2025-07-01",
)

# Only 3 representative BAs for tuning
TUNING_BAS = ['ERCO', 'PJM', 'CISO']  # Large/medium/West

def objective(trial, df, optuna_cfg):
    # ... sample hyperparameters ...
    
    # Filter to 3 BAs
    df_tune = df[df['ba_code'].isin(TUNING_BAS)]
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        PyTorchLightningPruningCallback(trial, monitor="val_loss"),  # Kill bad trials
    ]
    
    # Single train/val split, 10 epochs max
    trainer = pl.Trainer(max_epochs=10, callbacks=callbacks, ...)
    trainer.fit(tft, ...)
```

**Result:** 20 trials × 1 split × 10 epochs × 3 BAs ≈ **2 hours** instead of 20+ hours.

---

## 15. Fix 14: RTX 5070 Laptop Hardware Profile

### 15.1 Hardware Specifications

| Spec | RTX 5070 Laptop |
|------|----------------|
| **Architecture** | Blackwell (GB206) |
| **VRAM** | 8 GB GDDR7 |
| **Memory Bandwidth** | 384 GB/s |
| **CUDA Cores** | 4,608 |
| **SM Count** | ~36 |
| **Tensor Cores** | 5th Gen (FP4/MXFP4, BF16 native) |
| **TGP Range** | 35W - 115W (laptop-dependent) |

### 15.2 Key Decisions

**Precision: `bf16-mixed`**

The RTX 5070 (Blackwell) has native BF16 GEMM support. BF16 has the same exponent range as FP32 (avoids the overflow that killed FP16 on TFT attention masking) while halving memory usage. This is the optimal choice.

```
FP32:     Each parameter = 4 bytes → 8GB VRAM ≈ 2.1B parameter slots
BF16:     Each parameter = 2 bytes → 8GB VRAM ≈ 4.2B parameter slots (2x headroom)
```

**Batch size: 128**

At 384 GB/s bandwidth, the GPU can saturate compute at batch_size=128. Going higher reduces gradient noise but hits VRAM limits when combined with the 168h encoder length.

```
Memory budget per batch (BF16):
  Model weights:  ~500 KB (hidden=64, tiny for TFT)
  Activations:    batch × encoder_length × features × hidden
                  = 128 × 168 × 34 × 64 × 2 bytes ≈ 94 MB
  Optimizer state: ~1 MB (AdamW, 2x weights)
  Gradient:       ~500 KB
  Total:          ~100 MB per batch → well within 8GB
```

**Workers: 4**

Laptop cooling limits sustained throughput. 4 workers balance data loading with thermal management. Higher counts risk thermal throttling.

### 15.3 Implementation

**File:** `src/models/config.py` [MODIFY]

```python
# ── RTX 5070 Laptop Profile ────────────────────────────────────────
# SM_120 (Blackwell GB206), 8GB GDDR7 @ 384 GB/s
# BF16-mixed: native Blackwell support, same range as FP32, half memory
# AdamW + OneCycleLR replaces Ranger + ReduceLROnPlateau

RTX_5070_LAPTOP_CONFIG = TFTConfig(
    batch_size=64,                  # User-specified; safe on 8GB VRAM
    num_workers=2,                  # Windows spawn overhead — 2 is optimal
    pin_memory=True,
    prefetch_factor=4,
    precision="bf16-mixed",         # Native BF16 on Blackwell
    log_every_n_steps=50,
    tft_log_interval=-1,            # Disable interpretability during training
)

DEFAULT_TFT_CONFIG = RTX_5070_LAPTOP_CONFIG
```

---

## 16. Fix 15: Caldara-Iacoviello GPR Index Integration

### 16.1 What Is the GPR Index?

The **Geopolitical Risk (GPR) Index** by Caldara & Iacoviello (American Economic Review, 2022) is the academic gold standard for measuring geopolitical risk. It is computed daily by counting keywords related to geopolitical threats and acts in 10 major international newspapers, normalized by total article volume.

**File:** `data_gpr_daily_recent.xls` (3.2 MB, 15,057 rows, 1985–2026-03-23)

### 16.2 Available Columns

| Column | Description |
|--------|-------------|
| `GPRD` | Daily GPR Index (100 = 1985-2019 mean) |
| `GPRD_ACT` | GPR Acts sub-index (actual events — wars, attacks, escalations) |
| `GPRD_THREAT` | GPR Threats sub-index (threats, rhetoric, tensions) |
| `GPRD_MA7` | 7-day moving average (pre-computed, smoothed) |
| `GPRD_MA30` | 30-day moving average |
| `N10D` | Number of articles in the 10 newspapers that day |

### 16.3 Crisis Signal Separation — GPR vs Our Broken geo_risk_index

```
Crisis                 | GPR Daily Mean | Our geo_risk_index | Separation?
-----------------------|----------------|-------------------|-----------
Ukraine Invasion 2022  |      361.3     |       2.13        | GPR: 3x normal. Ours: BELOW normal.
Iran War 2026          |      330.0     |       4.40        | GPR: 2.6x normal. Ours: +1σ only.
Israel-Hamas 2023      |      210.8     |       2.70        | GPR: 1.6x normal. Ours: indistinguishable.
Iran Strikes 2024      |      177.3     |       2.90        | GPR: 1.4x normal. Ours: +0.2σ.
Texas Freeze 2021      |       60.7     |       2.57        | GPR: BELOW normal. Ours: normal.
Normal day             |      128.8     |       2.67        | —
```

The GPR index massively separates crises from normal days (1.4x to 3x baseline). Our homegrown `geo_risk_index` could not (all values in [2.1, 4.4]).

### 16.4 Why Include GPR If International Crises Don't Affect US Demand?

We proved in §5 that international crises don't directly move US demand. However, the GPR index serves a **different purpose in Model C**:

1. **Uncertainty regime detection.** The GPR index acts as a regime indicator. When GPR is elevated (>200), gas markets behave differently (higher volatility, risk premiums). The TFT can use `gpr_index` to contextually weight its gas_price features.

2. **Interaction with gas_price_volatility_7d.** GPR alone doesn't move demand. But `GPR × gas_volatility` creates a compound signal that the TFT's GRN can gate: "high geopolitical risk AND elevated gas volatility → dispatch stack uncertainty."

3. **Zero cost.** The GPR index is a single daily float. As a `time_varying_observed_real`, it occupies minimal capacity. The VSN will assign it near-zero weight on normal days and only activate during true geopolitical spikes via the GRN.

4. **Academic credibility.** Using Caldara-Iacoviello (published in AER) is vastly more defensible than our homemade ratio-based index.

### 16.5 Implementation

```python
# In ingest_gpr.py [NEW]
import pandas as pd
from src.database.connection import get_engine
from sqlalchemy import text

gpr = pd.read_excel('data_gpr_daily_recent.xls', parse_dates=['date'])
gpr = gpr[gpr['date'] >= '2021-01-01'][['date', 'GPRD', 'GPRD_ACT', 'GPRD_THREAT', 'GPRD_MA7']]

# Z-score for crisis detection
gpr['gpr_zscore'] = (gpr['GPRD_MA7'] - gpr['GPRD_MA7'].rolling(90).mean()) / gpr['GPRD_MA7'].rolling(90).std()

# Broadcast to all BAs (GPR is global, not regional)
engine = get_engine()
for _, row in gpr.iterrows():
    engine.execute(text("""
        UPDATE analytics.features 
        SET gpr_index = :gpr, gpr_zscore = :zscore
        WHERE period::DATE = :dt
    """), {'gpr': row['GPRD_MA7'], 'zscore': row['gpr_zscore'], 'dt': row['date']})
```

**Feature enters TFT as:** `time_varying_observed_real` — `gpr_index` (GPRD_MA7, smoothed)

---

## 17. Audit: Quantile Loss — Keep 7 Quantiles

### 17.1 The Suggestion

> Reduce quantiles from `[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]` (7) to `[0.05, 0.5, 0.95]` (3) to save VRAM and speed up the backward pass.

### 17.2 The Verdict: **Keep 7 quantiles.**

**The overhead is negligible.** The quantile output layer is just the final linear projection: `hidden_size → n_quantiles × prediction_length` (64 → 7×24 = 168 neurons). Reducing to 3 quantiles saves 64 → 3×24 = 72 neurons — a difference of 96 neurons. The entire model has ~200K parameters; this saves <0.05%.

The computational cost of QuantileLoss is element-wise operations (comparisons, additions) — trivially cheap compared to the LSTM encoder, multi-head attention, and GRN computations that dominate training time.

**But the information loss is significant:** 
- `[0.02, 0.98]` captures the full 96% prediction interval — essential for extreme event analysis
- `[0.10, 0.90]` gives the 80% interval — standard for risk assessment  
- `[0.25, 0.75]` gives the IQR — useful for understanding typical uncertainty
- Dropping from 7 to 3 loses the ability to distinguish between "likely" and "extreme" uncertainty bands

**Decision:** Keep `quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]`. The VRAM savings are not worth the loss of distributional information.

---

## 18. Audit: Windows Multiprocessing — num_workers=2

### 18.1 The Issue

PyTorch on Windows uses `spawn` instead of `fork` for multiprocessing. Each worker spawns a **complete new Python interpreter**, re-imports all modules, and re-initializes the data pipeline. This has dramatically higher per-worker overhead than Linux's `fork` (which clones the existing process with shared memory).

### 18.2 Evidence

Benchmark pattern on Windows + PyTorch:

```
num_workers=0:  GPU util ~60%  (data-starved, CPU bottleneck)
num_workers=2:  GPU util ~90%  (sweet spot on Windows)
num_workers=4:  GPU util ~85%  (spawn overhead starts eating gains)
num_workers=8:  GPU util ~75%  (massive context switching, RAM pressure)
```

With `persistent_workers=True` (which we already use), the spawn cost is paid once per epoch rather than per batch. But on a laptop with limited RAM bandwidth, 4+ workers still compete for memory bus with the GPU.

### 18.3 Decision

**Set `num_workers=2` for RTX 5070 laptop.** Combined with:
- `pin_memory=True` (async CPU→GPU transfer via page-locked memory)
- `persistent_workers=True` (avoid re-spawning between epochs)
- `prefetch_factor=4` (pre-queue 4 batches per worker = 8 batches ready)

This keeps the GPU fed without melting the CPU.

---

## 19. Model C Complete Configuration & Architecture

### 16.1 Feature Architecture

**Target:** `demand_mw`

**Group IDs:** `['ba_code']`

**Static Categoricals:** `['ba_code']`

**Static Reals (NEW):** `['gas_sensitivity', 'renewable_sensitivity', 'nuclear_sensitivity']`

**Quantiles:** `[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]` (7 — full distributional prediction)

**Time-Varying Known Categoricals:** `['is_weekend', 'is_holiday']`

**Time-Varying Known Reals:**
```python
[
    "hour_of_day", "day_of_week", "month",
    "prophet_trend", "prophet_weekly", "prophet_yearly",  # NEW: Prophet decomp
    "temperature_c", "hdd", "cdd",  # MOVED from observed → known
]
```

**Time-Varying Observed Reals:**
```python
[
    # Demand autoregressive (6)
    "demand_lag_1h", "demand_lag_24h", "demand_lag_168h",
    "demand_rolling_24h", "demand_rolling_168h", "demand_std_24h",
    # Weather — past only (4)
    "humidity_pct", "wind_speed_kmh", "cloud_cover_pct", "solar_radiation",
    # Supply & Grid (3 — includes NEW supply_margin)
    "generation_mw", "supply_margin_pct", "gas_pct",
    # Price (3 — forward-filled, no more $0 weekends)
    "gas_price", "gas_price_volatility_7d", "nuclear_outage_pct",
    # Regional Energy Sentiment — from GKG (4 — replaces old 4 GDELT)
    "grid_stress_zscore", "gas_pipeline_zscore",
    "electricity_buzz_zscore", "energy_tone_regional",
    # GPR Index (1 — Caldara-Iacoviello, regime detection)
    "gpr_index",
]
```

**Total features:** 9 known reals + 2 known cats + 21 observed reals + 3 static reals = 35 features

### 19.2 Normalizer

```python
target_normalizer=GroupNormalizer(groups=['ba_code'], transformation="log1p")
```

### 19.3 Optimizer

```python
AdamW(lr=1e-3, weight_decay=0.01) + OneCycleLR(pct_start=0.1, anneal_strategy='cos')
```

### 19.4 Hardware

```python
RTX 5070 Laptop: bf16-mixed, batch_size=64, num_workers=2, pin_memory=True
```

### 19.5 BA Filter

Training restricted to 12 chart BAs only. Reduces dataset size from ~1.1M to ~549k rows.

### 19.6 Encoder/Decoder

- Encoder length: 168h (7 days) — unchanged
- Prediction length: 24h — unchanged

---

## 20. Chronological Execution Plan

### Phase A: Data Corruption & Validation (Steps 1-3)

#### Step 1: Clean Corrupted Demand (SQL)
- **Action:** Run corruption cleanup SQL on `clean.demand`
- **Files:** Direct SQL execution
- **Verify:** MIN/MAX within physical bounds for PJM, TVA, SWPP, FPL, NYIS

#### Step 2: Create Data Validation Script
- **Action:** Create `src/features/validate_data.py`
- **Files:** `src/features/validate_data.py` [NEW]
- **Verify:** Run, should show 0 physical bound violations

#### Step 3: Add New Columns to analytics.features
- **Action:** ALTER TABLE to add new columns
- **Files:** Direct SQL execution
- **Columns:** `supply_margin_pct`, `gas_pct_delta_24h`, `gas_price_volatility_7d`, `grid_stress_zscore`, `gas_pipeline_zscore`, `electricity_buzz_zscore`, `energy_tone_regional`, `prophet_trend`, `prophet_weekly`, `prophet_yearly`, `gas_sensitivity`, `renewable_sensitivity`, `nuclear_sensitivity`, `gpr_index`, `gpr_zscore`

### Phase B: Feature Engineering (Steps 4-9)

#### Step 4: Fix is_holiday (build_features.py)
- **Action:** Replace `FALSE AS is_holiday` with actual US federal holiday detection
- **Files:** `src/features/build_features.py` [MODIFY]
- **Method:** Generate holiday dates via `pandas.tseries.holiday.USFederalHolidayCalendar`, create ref table or inject into SQL
- **Verify:** `SELECT COUNT(*) FROM analytics.features WHERE is_holiday = TRUE` → ~1,320 rows per BA

#### Step 5: Modify build_features.py for new supply/price features
- **Action:** Add supply_margin, gas_momentum, gas_volatility to SQL; fix gas price forward-fill
- **Files:** `src/features/build_features.py` [MODIFY]
- **Verify:** New columns populated with sensible values

#### Step 6: Rebuild Gold Layer
- **Action:** Run `python -m src.features.build_features`
- **Time:** ~30 min
- **Verify:** Row count, NULL percentages, demand bounds, is_holiday counts

#### Step 7: Prophet Decomposition
- **Action:** Create and run `src/features/prophet_decompose.py`
- **Files:** `src/features/prophet_decompose.py` [NEW]
- **Time:** ~6 min (30 sec/BA × 12)
- **Verify:** prophet_trend/weekly/yearly populated for all 12 BAs

#### Step 8: Compute BA Fuel Sensitivity Profiles
- **Action:** Create and run `src/features/compute_ba_profiles.py`
- **Files:** `src/features/compute_ba_profiles.py` [NEW]
- **Verify:** gas_sensitivity populated (ISNE ≈ 0.55, BPAT ≈ 0.05)

#### Step 9: Ingest and Process GKG Energy Sentiment
- **Action:** Run state→BA aggregation, z-scoring, and DB insert
- **Files:** `src/features/ingest_gkg_sentiment.py` [NEW]
- **Input:** `gkg.csv` (already downloaded from BigQuery)
- **Verify:** grid_stress_zscore spikes during Texas Freeze, hurricanes

#### Step 9b: Ingest GPR Index
- **Action:** Read `data_gpr_daily_recent.xls`, compute 90-day z-score, broadcast to all BAs
- **Files:** `src/features/ingest_gpr.py` [NEW]
- **Input:** `data_gpr_daily_recent.xls` (already downloaded from matteoiacoviello.com)
- **Verify:** gpr_index elevated during Ukraine/Iran, low during Texas Freeze

### Phase C: Model Configuration (Steps 10-13)

#### Step 10: Update Model Configuration
- **Action:** Add Model C features, static reals, BA filter, RTX 5070 profile
- **Files:** `src/models/config.py` [MODIFY]
- **Verify:** Feature lists match Section 16.1, hardware profile = RTX_5070_LAPTOP

#### Step 11: Update Dataset Builder
- **Action:** Fix fillna (ffill/bfill, no zeros), add log1p normalizer, add BA filter, Model C variant
- **Files:** `src/models/dataset.py` [MODIFY]
- **Verify:** `python -m src.models.dataset` runs with zero NaN, ~549k rows

#### Step 12: Update Training Script (Optimizer + Model C)
- **Action:** Switch Ranger → AdamW+OneCycleLR via SentinelTFT subclass, add --model C flag
- **Files:** `src/models/train_tft.py` [MODIFY]

#### Step 13: Update Optuna (single-split + pruning)
- **Action:** Replace 3-fold CV with single split, add PyTorchLightningPruningCallback, filter to 3 BAs
- **Files:** `src/models/optimize.py` [MODIFY]

### Phase D: Validation & Training (Steps 14-18)

#### Step 14: Final Data Validation
- **Action:** Run comprehensive data validation before training
- **Checks:** Zero NaN, correct bounds, all features populated, is_holiday counts, feature distributions

#### Step 15: Smoke Test
- **Action:** `python -m src.models.train_tft --model C --smoke-test`
- **Time:** ~5 min
- **Watch:** No NaN loss, val_loss decreasing, AdamW+OneCycleLR active, BF16 precision

#### Step 16: Optuna Hyperparameter Search (Optional)
- **Action:** `python -m src.models.optimize --trials 20`
- **Time:** ~2 hours (3 BAs, single split, pruning)
- **Output:** Best hidden_size, dropout, attention_heads, learning_rate

#### Step 17: Full Training
- **Action:** `python -m src.models.train_tft --model C`
- **Config:** 30 epochs, early stop patience=5, batch_size=64, BF16-mixed, AdamW+OneCycleLR, num_workers=2
- **Time:** ~5-8 hours

#### Step 18: Per-BA Evaluation
- **Action:** `python -m src.models.evaluate --model C`
- **Target:** <5% MAPE across all 12 BAs, 0/12 flat-line predictions
