# SCHEMA.md — Database Schema Reference

> **Database**: `sentinel` on `localhost:5432`  
> **Engine**: TimescaleDB (PostgreSQL 16)  
> **Schemas**: `raw` (Bronze), `clean` (Silver), `analytics` (Gold)

---

## 1. RAW Schema (Bronze Layer)

### `raw.eia_region_data` (Hypertable, ~4.5M rows)
| Column | Type | Notes |
|---|---|---|
| id | BIGSERIAL | Auto-increment |
| period | TIMESTAMPTZ | PK (hypertable key) |
| respondent | VARCHAR(10) | BA code (e.g., ERCO) |
| respondent_name | VARCHAR(200) | |
| type | VARCHAR(5) | D, DF, NG, TI |
| type_name | VARCHAR(200) | |
| value | DOUBLE PRECISION | MW |
| units | VARCHAR(20) | |
| ingested_at | TIMESTAMPTZ | |

### `raw.eia_fuel_type_data` (Hypertable, ~7.2M rows)
| Column | Type | Notes |
|---|---|---|
| id | BIGSERIAL | |
| period | TIMESTAMPTZ | PK |
| respondent | VARCHAR(10) | BA code |
| respondent_name | VARCHAR(200) | |
| fueltype | VARCHAR(10) | COL, NG, NUC, OIL, SUN, WAT, WND, OTH, BAT, GEO, PS, etc. |
| type_name | VARCHAR(200) | |
| value | DOUBLE PRECISION | MW |
| units | VARCHAR(20) | |
| ingested_at | TIMESTAMPTZ | |

### `raw.eia_interchange_data` (Hypertable, ~9M rows)
| Column | Type | Notes |
|---|---|---|
| id | BIGSERIAL | |
| period | TIMESTAMPTZ | PK |
| respondent | VARCHAR(10) | |
| fromba / toba | VARCHAR(10) | BA-pair |
| value | DOUBLE PRECISION | MW |
| units | VARCHAR(20) | |

### `raw.weather_hourly` (Hypertable, ~1.1M rows)
| Column | Type | Notes |
|---|---|---|
| period | TIMESTAMPTZ | PK |
| ba_code | VARCHAR(10) | PK |
| temperature_2m | DOUBLE PRECISION | °C |
| relative_humidity_2m | DOUBLE PRECISION | % |
| wind_speed_10m | DOUBLE PRECISION | km/h |
| cloud_cover | DOUBLE PRECISION | % |
| shortwave_radiation | DOUBLE PRECISION | W/m² |
| precipitation | DOUBLE PRECISION | mm |

### `raw.gdelt_events_daily` (1,910 rows)
| Column | Type | Notes |
|---|---|---|
| event_date | DATE | PK |
| us_event_count | INTEGER | US daily event total |
| us_avg_goldstein / min / max / std | DOUBLE PRECISION | US conflict scale |
| us_avg_tone / min_tone | DOUBLE PRECISION | Media sentiment |
| us_total_mentions / articles | BIGINT | Volume metrics |
| us_severe_conflict / moderate / tension / cooperation | INTEGER | Severity tiers |
| us_very_negative / crisis_events | INTEGER | |
| global_event_count | INTEGER | Worldwide total |
| global_avg_goldstein / min / std | DOUBLE PRECISION | |
| global_avg_tone | DOUBLE PRECISION | |
| global_total_articles | BIGINT | |
| oil_region_event_count | INTEGER | Middle East / oil producers |
| oil_region_avg_goldstein | DOUBLE PRECISION | |
| global_severe_conflict / moderate / crisis / very_negative | INTEGER | |

---

## 2. CLEAN Schema (Silver Layer)

### `clean.demand` (Hypertable, hourly × 25 BAs)
| Column | Type | Notes |
|---|---|---|
| period | TIMESTAMPTZ | PK |
| ba_code | VARCHAR(10) | PK |
| demand_mw | DOUBLE PRECISION | Pivoted from type='D' |
| forecast_mw | DOUBLE PRECISION | From type='DF' |
| generation_mw | DOUBLE PRECISION | From type='NG' |
| interchange_mw | DOUBLE PRECISION | From type='TI' |
| demand_forecast_gap | DOUBLE PRECISION | demand - forecast |
| supply_demand_gap | DOUBLE PRECISION | generation - demand |
| is_interpolated | BOOLEAN | Gap-fill flag |

### `clean.fuel_mix` (Hypertable, hourly × 25 BAs)
| Column | Type | Notes |
|---|---|---|
| period | TIMESTAMPTZ | PK |
| ba_code | VARCHAR(10) | PK |
| coal_mw / gas_mw / nuclear_mw / oil_mw | DOUBLE PRECISION | By fuel type |
| solar_mw / hydro_mw / wind_mw / other_mw | DOUBLE PRECISION | |
| total_mw | DOUBLE PRECISION | Sum of all fuels |
| gas_pct | DOUBLE PRECISION | % of total from natural gas |
| renewable_pct | DOUBLE PRECISION | % from solar + wind + hydro |

---

## 3. ANALYTICS Schema (Gold Layer)

### `analytics.features` (Hypertable, hourly × 25 BAs)
| Column | Type | Source |
|---|---|---|
| period | TIMESTAMPTZ | PK |
| ba_code | VARCHAR(10) | PK |
| **demand_mw** | DOUBLE PRECISION | **TARGET** |
| hour_of_day / day_of_week / month | SMALLINT | Derived |
| is_weekend / is_holiday | BOOLEAN | Derived |
| demand_lag_1h / 24h / 168h | DOUBLE PRECISION | Window function |
| demand_rolling_24h / 168h | DOUBLE PRECISION | Window function |
| demand_std_24h | DOUBLE PRECISION | Window function |
| temperature_c / humidity_pct / wind_speed_kmh | DOUBLE PRECISION | Weather join |
| cloud_cover_pct / solar_radiation | DOUBLE PRECISION | Weather join |
| hdd / cdd | DOUBLE PRECISION | Heating/cooling degree days |
| generation_mw / supply_demand_gap | DOUBLE PRECISION | clean.demand |
| gas_pct / renewable_pct | DOUBLE PRECISION | clean.fuel_mix |
| interchange_mw | DOUBLE PRECISION | clean.demand |
| gas_price / oil_price / gas_price_change_7d | DOUBLE PRECISION | Not yet populated |
| nuclear_outage_pct | DOUBLE PRECISION | Not yet populated |
| sentiment_mean_24h / min_24h | DOUBLE PRECISION | GDELT us_avg/min_goldstein |
| event_count_24h | INTEGER | GDELT us_event_count |
| geo_risk_index | DOUBLE PRECISION | Composite: severe_conflict + oil_region tension |
| is_spike | BOOLEAN | demand > 2σ from 24h rolling mean |
| spike_magnitude | DOUBLE PRECISION | Z-score of deviation |

### `analytics.predictions`
| Column | Type | Notes |
|---|---|---|
| target_period | TIMESTAMPTZ | Prediction target hour |
| ba_code | VARCHAR(10) | |
| model_name | VARCHAR(50) | e.g., 'tft_v1', 'xgb_spike' |
| predicted_demand | DOUBLE PRECISION | |
| actual_demand | DOUBLE PRECISION | (filled after observation) |
| confidence_lower / upper | DOUBLE PRECISION | 80% CI |
| mape | DOUBLE PRECISION | |
| prediction_hash / tx_hash | VARCHAR(66) | For blockchain accountability |
