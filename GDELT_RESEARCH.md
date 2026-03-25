# SENTINEL — Geopolitical Data Source Research Report

## Executive Summary

GDELT is **feasible but requires surgical precision** to avoid drowning in irrelevant data. The key insight from research is that GDELT has **two separate tables** that serve different purposes for SENTINEL, and we should use **both** strategically. However, there are also **strong alternatives** that may be simpler to implement.

---

## 1. GDELT Architecture (What We're Working With)

GDELT 2.0 lives entirely on **Google BigQuery** (free tier: **1 TB scanned/month**, 10 GB storage).

### Two Tables, Two Purposes

| Table | Purpose | Size | Best For |
|---|---|---|---|
| `gdelt-bq.gdeltv2.events` | "Who did what to whom" — coded activities | ~63M events/year (~60GB/year) | Detecting **sanctions, protests, military actions** near energy infrastructure |
| `gdelt-bq.gdeltv2.gkg` | Global Knowledge Graph — themes, sentiment, tone | ~266M articles, 2.65+ TB | Detecting **energy-specific news themes** (oil, gas, pipeline, electricity) with built-in sentiment |

---

## 2. GKG V2Themes (The Better Source for Energy)

Energy-relevant theme strings include:
```
ENV_OIL, ENV_NATURALGAS, ENV_COAL, ENV_NUCLEAR, ENV_SOLAR, ENV_WIND,
INFRASTRUCTURE, ENERGY, ELECTRICITY, PIPELINE, POWER_GRID,
SANCTIONS, TRADE_DISPUTE, WEATHER_HURRICANE, WEATHER_WINTER_STORM
```

### Storage Estimation

$$V_{compressed} = (50\text{MB} + 1.2\text{GB}) \times 0.1 \approx 125\text{MB after TimescaleDB compression}$$

---

## 3. The GoldsteinScale — GDELT's Built-In "Crisis Meter"

The Events table includes a `GoldsteinScale` column: a **-10 to +10 float** that measures the theoretical impact of an event on country stability. This is essentially a **pre-built geopolitical risk index**.

---

## 4. Alternatives

- **NewsAPI.ai**: AI-powered news, pre-clustered energy events, simple REST API.
- **NewsData.io**: Real-time + historical news, built-in sentiment.
- **VADER + Google News**: Free, DIY sentiment analysis on RSS feeds (limited history).

---

## 5. Recommended Strategy

- **GDELT GKG** for historical training.
- **NewsAPI.ai** for real-time triggers.
- **VADER** for local zero-cost sentiment analysis.
