# SENTINEL — Predictive Energy Monitoring Framework

## Quick Start

```bash
# 1. Clone and enter project
git clone <repo-url>
cd SENTINEL

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
copy .env.example .env
# Edit .env and add your EIA API key

# 5. Start database
docker compose up -d

# 6. Run database migrations
python -m src.database.migrate

# 7. Backfill historical data (runs 1-2 hours)
python -m src.ingestion.backfill
```

## Project Structure

```
SENTINEL/
├── docker-compose.yml       # TimescaleDB container
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── src/
│   ├── config/              # Settings and configuration
│   ├── database/            # Schema, migrations, connection
│   ├── ingestion/           # EIA, weather, news data pipelines
│   ├── analysis/            # Pattern analysis, anomaly detection
│   ├── features/            # Feature engineering pipeline
│   ├── models/              # XGBoost, TFT, LSTM, Prophet
│   ├── nlp/                 # News sentiment, entity extraction
│   ├── cascading/           # Dependency graph, risk simulation
│   └── dashboard/           # Streamlit visualization
├── data/                    # Local data files (gitignored)
├── models/                  # Trained model artifacts (gitignored)
└── tests/                   # Test suite
```

## Team

- **Person A**: Data Foundation + Models Lead (RTX 4060)
- **Person B**: GPU Training Lead (RTX 5060)
- **Person C**: NLP + Dashboard Lead (lightweight notebook)
