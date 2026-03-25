"""
SENTINEL Configuration Module
Loads settings from .env file and provides typed configuration.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from functools import lru_cache


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # ── EIA API ─────────────────────────────────────────────────────
    eia_api_key: str = Field(..., description="EIA Open Data API key")
    eia_base_url: str = Field(
        default="https://api.eia.gov/v2",
        description="EIA API v2 base URL",
    )

    # ── Database ────────────────────────────────────────────────────
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=5432)
    db_name: str = Field(default="sentinel")
    db_user: str = Field(default="sentinel")
    db_password: str = Field(default="sentinel_dev_2026")

    # ── Open-Meteo ──────────────────────────────────────────────────
    openmeteo_base_url: str = Field(
        default="https://api.open-meteo.com/v1",
        description="Open-Meteo API base URL (no key needed)",
    )

    # ── News APIs ───────────────────────────────────────────────────
    newsapi_key: str = Field(default="", description="NewsAPI.ai key (optional)")
    gdelt_base_url: str = Field(
        default="https://api.gdeltproject.org/api/v2",
        description="GDELT Project API base URL",
    )

    # ── Logging ─────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    # ── Derived Properties ──────────────────────────────────────────
    @property
    def database_url(self) -> str:
        """SQLAlchemy connection string."""
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def async_database_url(self) -> str:
        """Async SQLAlchemy connection string."""
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# ── Constants ───────────────────────────────────────────────────────

# EIA data types
EIA_DATA_TYPES = {
    "D": "Demand",
    "DF": "Day-ahead demand forecast",
    "NG": "Net generation",
    "TI": "Total interchange",
}

# Key Balancing Authorities (covering major U.S. regions)
KEY_BALANCING_AUTHORITIES = {
    "ERCO": "Electric Reliability Council of Texas (ERCOT)",
    "PJM": "PJM Interconnection",
    "CISO": "California Independent System Operator (CAISO)",
    "MISO": "Midcontinent Independent System Operator",
    "NYIS": "New York Independent System Operator (NYISO)",
    "ISNE": "ISO New England",
    "SWPP": "Southwest Power Pool",
    "SOCO": "Southern Company",
    "TVA": "Tennessee Valley Authority",
    "DUK": "Duke Energy Carolinas",
    "FPL": "Florida Power & Light",
    "AECI": "Associated Electric Cooperative Inc.",
    "AVA": "Avista Corporation",
    "BPAT": "Bonneville Power Administration",
    "LGEE": "Louisville Gas and Electric / Kentucky Utilities",
    "NEVP": "Nevada Power",
    "PACE": "PacifiCorp East",
    "PACW": "PacifiCorp West",
    "PSCO": "Public Service Company of Colorado",
    "SC": "South Carolina Public Service Authority",
    "SCEG": "Dominion Energy South Carolina",
    "SEC": "Seminole Electric Cooperative",
    "TAL": "City of Tallahassee",
    "TEC": "Tampa Electric Company",
    "WACM": "Western Area Power Administration - Colorado/Missouri",
}

# All 65 BAs — will be populated from EIA API on first run
ALL_BALANCING_AUTHORITIES = {}

# Fuel types returned by EIA
FUEL_TYPES = {
    "COL": "Coal",
    "NG": "Natural Gas",
    "NUC": "Nuclear",
    "OIL": "Petroleum",
    "SUN": "Solar",
    "WAT": "Hydro",
    "WND": "Wind",
    "OTH": "Other",
    "UNK": "Unknown",
}

# Data extraction date ranges
BACKFILL_START_DATE = "2021-01-01T00"
BACKFILL_END_DATE = "2026-03-23T00"

# EIA API pagination
EIA_PAGE_SIZE = 5000
EIA_REQUEST_DELAY = 0.5  # seconds between requests to avoid rate limiting
