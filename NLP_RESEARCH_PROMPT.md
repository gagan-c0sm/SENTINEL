# NLP Research Prompt for SENTINEL Team Member

> **Copy everything below the line into a new conversation with your AI assistant.**
> This gives them full project context so they can help you research and build the NLP pipeline.

---

# START OF PROMPT

## Project Context: SENTINEL

I'm working on a research project called **SENTINEL** (Supply Energy Network Threat Identification and National Early-warning Layer). It's a predictive energy monitoring framework for the U.S. power grid that predicts demand spikes 24 hours ahead using historical data, weather, and geopolitical/news signals.

**My role on the team:** I'm responsible for the **NLP & Geopolitical Intelligence Pipeline** — the module that ingests energy news, performs sentiment analysis, extracts events, and generates a geopolitical risk score that feeds into the demand prediction models.

**My hardware:** I have a lightweight notebook (no GPU, ~8–16 GB RAM). All model training will be done by a teammate with a GPU (RTX 5060). I write the code, push to GitHub, and they run the training. I can run inference on CPU with small models.

**Tech stack:** Python 3.11+, PyTorch, Hugging Face Transformers, pandas, NetworkX. Database is TimescaleDB (PostgreSQL).

---

## What I Need You to Research (In Detail)

### 1. NLP Model Selection for Energy News Sentiment

I need to classify energy news articles into sentiment (positive/neutral/negative impact on energy supply) and extract relevant entities (fuel types, regions, event types).

**Research the following models and compare them for my use case:**

1. **FinBERT** (ProsusAI/finbert) — Financial sentiment model. Can it transfer to energy domain? What's the domain gap?
2. **DistilBERT** (distilbert-base-uncased) — Lighter, faster. How much accuracy do we lose vs full BERT? Can it fit in 8GB VRAM for fine-tuning?
3. **ClimateBERT** (climatebert/distilroberta-base-climate-f) — Climate-specific. Does it understand energy supply chain language?
4. **RoBERTa-base** — Stronger than BERT for classification. VRAM requirements for fine-tuning?
5. **Zero-shot classification** (facebook/bart-large-mnli) — Can we skip fine-tuning entirely and use zero-shot with energy-specific labels? What's the accuracy trade-off?

**For each model, I need:**
- Parameter count and model size (MB)
- VRAM required for fine-tuning (batch size 16 and 32)
- VRAM required for inference
- Can it run inference on CPU? How slow?
- Pre-training domain (finance, climate, general) and relevance to energy news
- Expected accuracy for sentiment classification given our domain
- Recommended fine-tuning approach (full fine-tune vs. LoRA/QLoRA vs. adapter layers)

### 2. Fine-Tuning Data: Where Do I Get Labeled Energy News?

I need labeled training data for energy news sentiment. Research:

- Does a labeled energy news sentiment dataset exist? (Kaggle, HuggingFace datasets, academic papers)
- If not, how many articles do I need to manually label for fine-tuning? (minimum viable, ideal)
- Can I use **FinBERT on financial energy headlines** as a starting point and then fine-tune on a small manually labeled energy-specific set? (transfer learning strategy)
- What annotation scheme should I use? Binary (positive/negative)? Ternary (positive/neutral/negative)? Or multi-label (supply disruption, price impact, policy change, demand effect)?
- Tools for quick annotation: Label Studio, Prodigy, or simpler approaches?

### 3. Named Entity Recognition (NER) for Energy Articles

I need to extract structured entities from news text:

- **Fuel types:** oil, natural gas, coal, solar, wind, nuclear
- **Regions:** Middle East, Gulf of Mexico, Texas, specific countries
- **Events:** sanctions, pipeline explosion, embargo, hurricane, shutdown, strike
- **Commodities:** Brent crude, WTI, Henry Hub, LNG

**Research:**
- Can I use spaCy NER with custom entity types? What's the training requirement?
- Would a pre-trained NER model + rule-based post-processing (regex + keyword lists) be simpler and good enough?
- Is there a model already trained on commodity/energy NER?
- How do I handle entity linking (mapping "Strait of Hormuz" → oil supply chokepoint → affects X% of global oil)?

### 4. Event Classification Taxonomy

I need to classify detected events into categories that map to energy impact. Propose a taxonomy:

**Level 1: Event Type**
- Supply Disruption
- Price Shock
- Policy Change
- Infrastructure Event
- Weather Event
- Demand Event

**For each category:**
- What sub-categories exist?
- What is the expected energy impact direction (positive/negative for supply)?
- What is the typical time lag between the news event and its effect on the U.S. electricity grid?
- What BAs (Balancing Authorities) would be most affected?

### 5. Time Synchronization: How News Events Align With Hourly Energy Data

This is CRITICAL and requires careful design. Our energy data is hourly timestamps (UTC). News articles have publication timestamps. The question is: **how do we align them?**

**Research and propose solutions for:**

1. **Publication time vs. event time:** An article published at 2 PM about an event that happened at 8 AM — which timestamp do we use?
2. **Time lag modeling:** If oil prices spike today due to a geopolitical event, when does it affect:
   - Oil spot prices? (same day)
   - Natural gas spot prices? (1–3 days)
   - Electricity generation mix? (3–7 days)
   - BA-level demand patterns? (indirect, potentially weeks)
3. **Rolling sentiment windows:** Should we compute sentiment as:
   - Point-in-time (sentiment at hour X)?
   - Rolling average (mean sentiment over last 24h/48h/7d)?
   - Exponentially decaying (recent news weighted more)?
   - Multiple windows simultaneously (24h, 48h, 7d as separate features)?
4. **Time zone handling:** EIA data is in Eastern Time. News is in various time zones. How do we normalize?
5. **Feature engineering for the prediction model:** The ML models (XGBoost, LSTM) consume tabular features per hour. How do we convert a variable number of articles per hour into a fixed-size feature vector? Options:
   - Average sentiment score per hour
   - Max negative sentiment per hour
   - Count of high-impact articles per hour
   - Separate features per event category
   - Embedding-based aggregation

### 6. News Source Selection and Ingestion Pipeline

**Research these news sources and rank them by value:**

| Source | Type | Cost | Evaluate for: relevance, latency, coverage |
|---|---|---|---|
| EIA "Today in Energy" | RSS | Free | Official U.S. energy commentary |
| S&P Global Commodity Insights | RSS | Free (headlines) | Oil/gas price signals |
| Reuters Energy | RSS/API | Free (headlines) | Breaking energy news |
| NewsAPI.ai | REST API | Free tier (100 req/day) | Broad geopolitical |
| GDELT Project | Open data | Free | Pre-scored global events |
| Google News RSS | RSS | Free | Broad coverage |
| Energy Information Administration press releases | Web scrape | Free | U.S. policy |

**For each source:**
- What's the typical article delay (how fast after an event is it published)?
- What format does it return (title only? full text? summary?)
- Rate limits and API constraints?
- How many articles/day on average?
- Example API call or RSS URL

**Pipeline architecture questions:**
- Should I use `feedparser` for RSS or a unified API aggregator?
- How do I avoid duplicate articles across sources?
- Should I store raw article text or just extracted features?
- What's the storage estimate for 1–2 years of news articles?

### 7. Geopolitical Risk Index Construction

I need to construct a single numerical score (0–100) per hour that represents the current energy-related geopolitical risk level. This score becomes a feature in the prediction models.

**Research how to construct this:**

1. What existing geopolitical risk indices exist? (Caldara & Iacoviello GPR Index, BlackRock Geopolitical Risk Indicator)
2. How do they compute their scores?
3. Can I build a simplified version using:
   - Volume of negative-sentiment energy news (count per 24h window)
   - Weighted by entity severity (nuclear > oil > gas > coal)
   - Weighted by geographic proximity to U.S. energy supply chains
   - Decaying over time (event 6 hours ago matters more than 48 hours ago)
4. How should I validate this index? (backtest against historical energy price movements)
5. Should the index be global or per-BA? (e.g., a gas supply disruption only affects gas-heavy BAs)

### 8. Cascading Effect Simulation: My Part

I'm also responsible for the **Energy Source Dependency Graph** and **Cascading Effect Simulation**. These are pure Python/NetworkX (no GPU needed).

**Research:**

1. **Dependency Graph:**
   - Each BA has a fuel mix (e.g., ERCOT: 48% gas, 25% wind, 10% nuclear, 8% solar, 9% coal)
   - This data comes from EIA Table 2 (`electricity/rto/fuel-type-data/data/`)
   - How should I model this as a graph? Bipartite graph (BAs ↔ fuel types)? Weighted edges?

2. **Cascading Simulation:**
   - When NLP detects an event (e.g., "oil supply disruption"), how do I simulate the cascade:
     ```
     Event detected → which fuel affected? → which BAs depend on that fuel?
     → how much generation is at risk? → can interchange from neighbors compensate?
     → final risk score per BA
     ```
   - What simulation approach? Monte Carlo? Deterministic rule-based? Bayesian network?
   - How do I estimate "generation at risk" from a fuel disruption? (e.g., if gas prices double, what % of gas generation becomes uneconomic?)

3. **NetworkX graph structure:**
   - What graph type (DiGraph, MultiGraph)?
   - Node attributes (BA properties: capacity, peak demand, fuel mix)
   - Edge attributes (interchange capacity between BAs)
   - What algorithms to use for risk propagation? (PageRank? Shortest path? Cascade models from network science?)

### 9. Integration Points With Other Team Members

**I need to deliver the following interfaces to the team:**

1. **To Person A (XGBoost/LSTM trainer):**
   - A function `get_nlp_features(ba_code, start_time, end_time)` that returns a DataFrame with hourly NLP features:
     ```
     timestamp | sentiment_mean_24h | sentiment_min_24h | event_count | supply_disruption_flag | geopolitical_risk_index
     ```
   - This gets joined with EIA + weather features for model training

2. **To Person B (CNN-LSTM trainer):**
   - Same interface, but also:
   - A function `get_news_embeddings(start_time, end_time)` that returns dense vector representations of recent news (for the CNN to learn from)

3. **To the Dashboard (Day 5):**
   - A function `get_recent_alerts()` returning current high-risk events with human-readable descriptions
   - A function `get_risk_map()` returning per-BA risk scores for map visualization

**Questions to answer:**
- What should the NLP feature schema look like? (exact column names, data types, update frequency)
- How do I handle time periods with no news? (return zeros? NaN? Last known value?)
- What's the latency requirement? (can NLP processing take 30 seconds? 5 minutes?)

### 10. Deliverables and Timeline

I have 5 days total. Here's what I need to deliver each day:

- **Day 1:** News ingestion pipeline (RSS + API client, article storage, deduplication)
- **Day 2:** NLP pipeline structure (preprocessing, model loading, sentiment inference, entity extraction)
- **Day 3:** Geopolitical risk index logic, event classification, supply disruption flagging  
- **Day 4:** Dependency graph (NetworkX), cascading effect simulation engine, alert generation
- **Day 5:** Streamlit dashboard (full build: charts, maps, alerts, predictions display)

For each day, give me:
- Exact Python files to create and what goes in each
- Function signatures (input/output types)
- Which libraries to import
- Any data files I need

### Summary of What I Need From You

1. **Model comparison table** (FinBERT vs DistilBERT vs ClimateBERT vs RoBERTa vs zero-shot)
2. **Fine-tuning data strategy** (where to get labeled data, how much, annotation scheme)
3. **NER approach** (model-based vs rule-based, entity taxonomy)
4. **Event classification taxonomy** with time-lag estimates
5. **Time synchronization design** (how to align news timestamps with hourly energy data)
6. **News source evaluation** with API/RSS details
7. **Geopolitical risk index formula**
8. **Dependency graph + cascading simulation architecture**
9. **Integration interface specification** (exact function signatures and data schemas)
10. **Day-by-day file-level implementation plan**

Be as detailed and technical as possible. Include code snippets, exact library versions, and specific model names from HuggingFace. I need to go from zero to working NLP pipeline in 4 days (Day 5 is dashboard).

# END OF PROMPT
