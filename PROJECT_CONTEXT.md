# SENTINEL — Project Context & Decisions Log

> **Purpose:** This file captures all architectural decisions, constraints, and context from the research/planning phase, transitioning into the finalized state of the network.

---

## 1. Project Identity

- **Name:** SENTINEL — **S**upply **E**nergy **N**etwork **T**hreat **I**dentification and **N**ational **E**arly-warning **L**ayer
- **Scope:** National-level (multi-BA), 24-hour prediction horizon
- **Type:** Research project concluding with multi-horizon ablation benchmarking.
- **Primary data source:** EIA Open Data API + GDELT + Open-Meteo
- **Historical window:** 5 years (2021–2026)

---

## 2. Hardware & Compute Constraints

| Component | Spec | Outcome
|---|---|---|
| **RAM** | 16 GB LPDDR5x | Efficient SQL ingestion prevented memory bloat perfectly. |
| **GPU** | NVIDIA RTX 4060 (8 GB VRAM) | Successfully supported PyTorch Forecasting (TFT) with `batch_size=256` |
| **OS** | Windows | Required optimized file-handling mapping for PyTorch. |

### Inference
- Inference is near-instant: full pipeline prediction runs in < 20 seconds even for multi-BA batches on DB feature loads.

---

## 3. Data Architecture Decisions

### Storage: Three-Layer Architecture (Bronze → Silver → Gold)

```
RAW (Bronze)          CLEAN (Silver)         ANALYTICS (Gold)
─────────────         ──────────────         ────────────────
Raw JSON/API          Validated,             Feature-engineered,
responses stored      deduplicated,          model-ready tables
as-is for replay      typed, indexed         (demand+weather+lags+sentiment)
```

### Database: **TimescaleDB (PostgreSQL)**

- Handled ~50–100M rows on local hardware elegantly.
- Feature aggregations generated utilizing efficient native window functions instead of Pandas.

---

## 4. Tech Stack Summary

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML | PyTorch Forecasting, PyTorch Lightning |
| Deep Learning Focus | Temporal Fusion Transformers (TFT), NHiTS, DeepAR |
| Data Pipeline | SQL-Driven Timescale pipelines |
| Database | TimescaleDB 16 (PostgreSQL) |

---

## 5. Research Novelty (What Makes This Publishable)

1. **Tier 3 predictions** — NLP-driven and global GDELT events successfully mapped directly to the Variable Selection Network of a Temporal Fusion Transformer.
2. **Quantifiable Uncertainty Restructuring** — Pinball score quantitively improved by 42.4% during systemic shock events against baselines, demonstrating that Global Geopolitics strongly govern grid stability beyond weather/seasonal cycles.
3. **Formal Verification** — Confirmed outperformance against robust classical DL paradigms via Diebold-Mariano ($p<0.001$).
4. **National-level mapping** — Intermingled 12 major balancing authorities seamlessly.

---

## 6. Project Phases (Conclusion)

- **Phase 1-3 (Data Infrastructure)**: 100% COMPLETE. Database size ~5.6GB.
- **Phase 4-5 (Advanced Modeling)**: 100% COMPLETE. DeepAR, NHiTS, and TFT successfully evaluated.
- **Phase 6 (Statistical Verification)**: 100% COMPLETE. Formal evaluation complete.

---

## 7. Current Status

- **Phase:** Project formally concluded. Research analysis generated.
- **Results Generation:** The repository contains ablation shootout graphs, sliding window prediction intervals, VSN interpretation figures natively integrated for markdown presentation.
- **Git State:** Final push stage initiated. Tooling directory stripped from tracker to save space and organize output safely.
