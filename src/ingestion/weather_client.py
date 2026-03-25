"""
SENTINEL — Open-Meteo API Client
Fetches hourly historical weather data for the geographical centers
of each Balancing Authority. No API key required for the free Archive API.
"""

import time
import requests
import pandas as pd
from typing import Dict, Tuple, List
from loguru import logger
from datetime import datetime
from tqdm import tqdm

from src.config.settings import PROJECT_ROOT


# Approximate geographical coordinates (lat, lon) for the centers of major BAs
BA_COORDINATES = {
    "ERCO": (31.9686, -99.9018),  # Texas
    "PJM": (40.7128, -74.0060),   # Represents broader PJM network area
    "CISO": (36.7783, -119.4179), # California
    "MISO": (40.0000, -89.0000),  # Midwest
    "NYIS": (43.0000, -75.0000),  # New York
    "ISNE": (42.0000, -71.5000),  # New England
    "SWPP": (38.0000, -97.0000),  # Southwest Power Pool
    "SOCO": (33.0000, -86.0000),  # Southern Company
    "TVA": (36.0000, -86.0000),   # Tennessee Valley Authority
    "DUK": (35.0000, -80.0000),   # Duke Energy
    "FPL": (27.0000, -81.0000),   # Florida Power & Light
    "AECI": (38.0000, -92.0000),  # Missouri area
    "AVA": (47.0000, -117.0000),  # Washington/Idaho
    "BPAT": (45.0000, -120.0000), # Bonneville Power Admin
    "LGEE": (38.0000, -85.0000),  # Kentucky
    "NEVP": (39.0000, -115.0000), # Nevada
    "PACE": (40.0000, -111.0000), # Utah area
    "PACW": (44.0000, -120.0000), # Oregon area
    "PSCO": (39.0000, -105.0000), # Colorado
    "SC": (34.0000, -80.0000),    # South Carolina
    "SCEG": (34.0000, -81.0000),  # SC Dominion
    "SEC": (28.0000, -82.0000),   # Seminole FL
    "TAL": (30.4383, -84.2807),   # Tallahassee
    "TEC": (27.9506, -82.4572),   # Tampa
    "WACM": (40.0000, -105.0000), # Colorado/Missouri
}


class WeatherClient:
    """Client for Open-Meteo Historical Weather API."""

    def __init__(self):
        # Open-Meteo Historical API URL
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        self.session = requests.Session()
        
    def fetch_historical_weather(self, ba_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch hourly weather variables for a given Balancing Authority.
        Open-Meteo allows fetching up to several years in a single request!
        """
        if ba_code not in BA_COORDINATES:
            logger.warning(f"No coordinates defined for BA: {ba_code}")
            return pd.DataFrame()

        lat, lon = BA_COORDINATES[ba_code]
        
        # We fetch the exact features needed by our model (TFT & XGBoost)
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date[:10],  # Format: YYYY-MM-DD
            "end_date": end_date[:10],
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "cloud_cover",
                "shortwave_radiation",
                "precipitation"
            ],
            "timezone": "UTC"
        }

        try:
            # Add a small delay to respect free tier limits (max 10,000 requests/day)
            time.sleep(1) 
            
            response = self.session.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 429:
                logger.warning("Rate limit hit. Waiting 60s...")
                time.sleep(60)
                response = self.session.get(self.base_url, params=params, timeout=30)
                
            response.raise_for_status()
            data = response.json()
            
            # Convert JSON response to Pandas DataFrame
            hourly = data.get("hourly", {})
            if not hourly:
                return pd.DataFrame()
                
            df = pd.DataFrame(hourly)
            df = df.rename(columns={"time": "period"})
            df["period"] = pd.to_datetime(df["period"], utc=True)
            df["ba_code"] = ba_code
            df["latitude"] = lat
            df["longitude"] = lon
            
            return df

        except Exception as e:
            logger.error(f"Failed to fetch weather for {ba_code}: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Quick Test
    client = WeatherClient()
    print("Testing Open-Meteo API via ERCOT (Texas) coordinates...")
    df = client.fetch_historical_weather("ERCO", "2024-01-01", "2024-01-07")
    print(f"✅ Downloaded {len(df)} hourly weather records for ERCOT.")
    print(df.head())
