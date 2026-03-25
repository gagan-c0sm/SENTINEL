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

### How CAMEO Codes Work (Events Table)

CAMEO uses a **hierarchical numbering system** with 20 root codes. Energy events don't have their own code — you must filter by **action type** + **geography**:

| Root Code | Category | Energy Relevance |
|---|---|---|
| **14** | Protest | Workers striking at refineries, pipeline protests |
| **17** | Coerce | Trade pressure, energy export threats |
| **18** | Assault | Attacks on infrastructure |
| **19** | Fight | Military conflict disrupting supply chains |
| **20** | Mass Violence | War zones affecting oil/gas production |
| **11** | Disapprove | Government rejecting energy deals |
| **12** | Reject | Sanctions on energy exports |
| **16** | Reduce Relations | Diplomatic fallout affecting trade |

> [!IMPORTANT]
> CAMEO codes classify **actions**, not **topics**. A protest at a pipeline (code 14) looks identical to a protest at a school. You **must** combine CAMEO filtering with GKG theme filtering or keyword matching to isolate energy-relevant events.

### How GKG V2Themes Work (The Better Source for Energy)

The GKG table tags every news article with machine-extracted **themes**. Energy-relevant theme strings include:

```
ENV_OIL, ENV_NATURALGAS, ENV_COAL, ENV_NUCLEAR, ENV_SOLAR, ENV_WIND,
INFRASTRUCTURE, ENERGY, ELECTRICITY, PIPELINE, POWER_GRID,
SANCTIONS, TRADE_DISPUTE, WEATHER_HURRICANE, WEATHER_WINTER_STORM
```

This is **dramatically more useful** than raw CAMEO codes because you can directly ask: "Give me every article about oil/gas/pipeline/electricity in the US with its sentiment score."

---

## 2. Optimal BigQuery Filtering Strategy

### The "Surgical Query" Approach

```sql
-- Events Table: US-only conflict/protest events (2021-2026)
SELECT 
    SQLDATE, Actor1Name, Actor2Name,
    EventCode, EventRootCode, GoldsteinScale, AvgTone,
    ActionGeo_Lat, ActionGeo_Long, SOURCEURL
FROM `gdelt-bq.gdeltv2.events`
WHERE ActionGeo_CountryCode = 'US'
  AND EventRootCode IN ('14','17','18','19','20')  -- conflict-related
  AND CAST(SQLDATE AS INT64) >= 20210101
```

```sql
-- GKG Table: US energy-themed articles with sentiment
SELECT 
    DATE, V2Themes, V2Tone, DocumentIdentifier,
    V2Locations, V2EnhancedDates
FROM `gdelt-bq.gdeltv2.gkg`
WHERE V2Locations LIKE '%US%'
  AND (V2Themes LIKE '%ENERGY%'
       OR V2Themes LIKE '%ENV_OIL%'
       OR V2Themes LIKE '%ENV_NATURALGAS%'
       OR V2Themes LIKE '%PIPELINE%'
       OR V2Themes LIKE '%ELECTRICITY%'
       OR V2Themes LIKE '%POWER_GRID%')
  AND _PARTITIONTIME >= '2021-01-01'
```

### Storage Estimation

$$V_{events} = \frac{63M \times 5\text{ years} \times 0.02 \text{ (US filter)} \times 0.05 \text{ (CAMEO filter)}}{1} \approx 315K \text{ rows} \approx 50\text{MB raw}$$

$$V_{gkg} = \frac{266M \times 5\text{ years} \times 0.15 \text{ (US)} \times 0.03 \text{ (energy themes)}}{1} \approx 6M \text{ rows} \approx 1.2\text{GB raw}$$

$$V_{compressed} = (50\text{MB} + 1.2\text{GB}) \times 0.1 \approx 125\text{MB after TimescaleDB compression}$$

> [!TIP]
> With aggressive filtering + compression, the entire 5-year GDELT dataset for US energy events fits in **~125MB**. Well within the 10GB budget.

### BigQuery Cost

- Events query: ~3GB scanned per run (well under 1TB free tier)
- GKG query: ~50-100GB scanned per run (need to partition by date carefully)
- **Monthly budget**: ~4-5 GKG queries/month safely within free tier

---

## 3. The GoldsteinScale — GDELT's Built-In "Crisis Meter"

The Events table includes a `GoldsteinScale` column: a **-10 to +10 float** that measures the theoretical impact of an event on country stability:

| Score | Meaning | Example |
|---|---|---|
| **+10** | Extreme cooperation | Major trade deal signed |
| **+5** | Positive diplomatic action | Energy partnership announced |
| **0** | Neutral | Routine government statement |
| **-5** | Threats, sanctions | Oil export sanctions imposed |
| **-10** | Extreme conflict | Military attack on infrastructure |

This is essentially a **pre-built geopolitical risk index** — exactly what SENTINEL needs without computing it ourselves.

---

## 4. Feasibility Verdict & Alternatives

### GDELT: Verdict = FEASIBLE (with caveats)

| Pro | Con |
|---|---|
| Free (BigQuery free tier) | Requires Google Cloud account setup |
| 5+ years of historical data | GKG queries can be expensive (high scan volume) |
| Pre-built sentiment (`AvgTone`) and crisis score (`GoldsteinScale`) | CAMEO codes don't directly tag "energy" — need theme filtering |
| Updated every 15 minutes | No direct REST API — must go through BigQuery |
| Academic gold standard for geopolitical research | Data quality varies (machine-coded, not human-verified) |

### Alternative 1: NewsAPI.ai (Formerly Event Registry) — STRONG

| Feature | Detail |
|---|---|
| **What it does** | AI-powered news aggregation with **built-in event detection, clustering, and sentiment analysis** |
| **Energy focus** | Can filter by energy concepts, companies, and supply chain keywords natively |
| **API** | Simple REST API — no BigQuery setup needed |
| **Historical data** | Archives going back to 2014 |
| **Free tier** | 2,000 article lookups/month (limited but usable for daily aggregation) |
| **Key advantage** | Events are **pre-clustered** — "Texas pipeline explosion" groups all related articles automatically |

> [!NOTE]
> NewsAPI.ai is the **simplest path** to production-ready geopolitical data. It eliminates the BigQuery dependency entirely and returns pre-structured JSON with sentiment scores.

### Alternative 2: NewsData.io — GOOD

| Feature | Detail |
|---|---|
| **What it does** | Real-time + historical news from 92,000+ sources |
| **Free tier** | 200 credits/day (reset daily) |
| **Sentiment** | Built-in sentiment field (positive/negative/neutral) |
| **Historical** | Back to January 2018 |
| **Limitation** | Less sophisticated event clustering than NewsAPI.ai |

### Alternative 3: VADER + Google News RSS — FREE (DIY)

| Feature | Detail |
|---|---|
| **What it does** | Scrape Google News RSS for energy keywords, run VADER sentiment locally |
| **Cost** | Completely free |
| **Effort** | Medium — requires building the scraper and NLP pipeline |
| **Quality** | Lower than pre-built APIs but fully controllable |
| **Historical** | Limited — RSS only provides ~30 days of history |

---

## 5. Recommended Strategy for SENTINEL

### Hybrid Approach (Best of Both Worlds)

```
Layer 1: GDELT GKG (BigQuery)
├── Monthly batch pull of US energy-themed articles
├── Extract: date, tone, themes, GoldsteinScale
├── Aggregate into hourly/daily buckets
└── Store in TimescaleDB as "geopolitical_sentiment" hypertable

Layer 2: NewsAPI.ai (REST API)  
├── Daily real-time pull of energy crisis events
├── Pre-clustered events with sentiment
├── Use as "breaking event" trigger
└── Store in TimescaleDB as "breaking_events" table

Layer 3: VADER (Local NLP)
├── Run sentiment analysis on any raw text we collect
├── No API dependency  
├── Backup sentiment scorer
└── Used in feature engineering pipeline
```

### Why This Works

- **GDELT** gives us the **5-year historical depth** needed for model training (backtesting the TFT against past crises)
- **NewsAPI.ai** gives us the **real-time event stream** needed for live predictions
- **VADER** gives us **zero-cost sentiment** we can run on any text without API limits

---

## 6. Implementation Priority

| Priority | Source | Effort | Value |
|---|---|---|---|
| **P0** | GDELT GKG via BigQuery (historical backfill) | 1-2 days | Training data for TFT model |
| **P1** | NewsAPI.ai (real-time events) | 0.5 days | Live prediction triggers |
| **P2** | VADER sentiment (local NLP) | Already available | Backup sentiment engine |
