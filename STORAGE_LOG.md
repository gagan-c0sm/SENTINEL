# STORAGE_LOG.md — Disk Usage Tracker

## Current Usage Summary
- **Total DB Size**: **4,639 MB (~4.6 GB)**
- **Budget**: 10.0 GB
- **Remaining**: **5.4 GB**

## Volume Breakdown
| Table | Rows | Size |
|---|---|---|
| `raw.eia_interchange_data` | 9,030,809 | 32 kB* |
| `raw.eia_fuel_type_data` | 7,190,244 | 24 kB* |
| `raw.eia_region_data` | 4,566,274 | 24 kB* |
| **Total Captured Rows** | **20,787,327** | |

*\*Note: TimescaleDB internal storage metrics for hypertables can appear low if querying parent relation; actual data is in chunks within the 4.6GB total.*

## Orphaned Staging Tables (Requires Nuke)
The following tables are left from previous ingestion crashes and are occupying several gigabytes:
`temp_csv_AECI`, `temp_csv_AVA`, `temp_csv_CISO`, `temp_csv_ERCO`, `temp_csv_ISNE`, `temp_csv_BPAT`, `temp_csv_FPL`, `temp_csv_NEVP`, etc.

## Growth Projection
$$V_{total} = (4.6\text{ GB}) + (1.2\text{ GB GDELT}) \approx 5.8\text{ GB}$$
We are **well under** the 10GB limit for Phase 1 and 2.

## Cleanliness Status
- **Checkpoint**: Strictly 25/25 BAs completed.
- **Deduplication**: `ON CONFLICT DO NOTHING` applied to all master tables.
