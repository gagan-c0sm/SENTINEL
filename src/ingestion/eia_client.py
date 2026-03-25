"""
SENTINEL — EIA Open Data API Client
Handles all communication with the EIA v2 API including pagination,
rate limiting, retries, and raw data storage.
"""

import time
import json
import requests
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from loguru import logger

from src.config import get_settings
from src.config.settings import (
    EIA_PAGE_SIZE,
    EIA_REQUEST_DELAY,
    EIA_DATA_TYPES,
    FUEL_TYPES,
    PROJECT_ROOT,
)


class EIAClient:
    """Client for the EIA Open Data API v2."""

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.eia_api_key
        self.base_url = settings.eia_base_url
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

        # Create raw data directory for bronze layer backups
        self.raw_dir = PROJECT_ROOT / "data" / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _make_request(
        self,
        endpoint: str,
        params: dict,
        max_retries: int = 5,
    ) -> dict:
        """
        Make a single API request with retry logic.

        Args:
            endpoint: API endpoint path (e.g., 'electricity/rto/region-data/data/')
            params: Query parameters
            max_retries: Number of retries on failure

        Returns:
            JSON response dict
        """
        url = f"{self.base_url}/{endpoint}"
        params["api_key"] = self.api_key

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=60)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited — back off exponentially
                    wait_time = (2 ** attempt) * 2
                    logger.warning(
                        f"Rate limited (429). Waiting {wait_time}s... "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                elif response.status_code == 503:
                    # Service unavailable — retry
                    wait_time = (2 ** attempt) * 3
                    logger.warning(
                        f"Service unavailable (503). Waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"API error {response.status_code}: {response.text[:500]}"
                    )
                    if attempt == max_retries - 1:
                        response.raise_for_status()
                    time.sleep(2)

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                time.sleep(5)
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries})")
                time.sleep(10)

        raise RuntimeError(f"Failed after {max_retries} retries: {endpoint}")

    def fetch_paginated(
        self,
        endpoint: str,
        params: dict,
        description: str = "Fetching",
        save_raw: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch all pages of data from an EIA API endpoint.

        Handles pagination automatically and returns a single DataFrame.
        Optionally saves raw JSON responses to disk (Bronze layer).

        Args:
            endpoint: API endpoint path
            params: Base query parameters (offset/length are added automatically)
            description: Progress bar description
            save_raw: Whether to save raw JSON to data/raw/

        Returns:
            DataFrame with all fetched rows
        """
        all_data = []
        offset = 0
        total_rows = None

        # Set up pagination params
        params["length"] = EIA_PAGE_SIZE
        params["sort[0][column]"] = "period"
        params["sort[0][direction]"] = "asc"

        # First request to get total count
        params["offset"] = offset
        response = self._make_request(endpoint, params.copy())

        response_data = response.get("response", {})
        total_rows = int(response_data.get("total", 0))
        data = response_data.get("data", [])

        if total_rows == 0:
            logger.info(f"No data found for {description}")
            return pd.DataFrame()

        all_data.extend(data)

        # Save raw response
        if save_raw:
            self._save_raw_response(endpoint, params, response, page=0)

        # Paginate through remaining data
        pbar = tqdm(
            total=total_rows,
            desc=description,
            initial=len(data),
            unit="rows",
        )

        while len(all_data) < total_rows:
            offset += EIA_PAGE_SIZE
            params["offset"] = offset

            # Rate limit delay
            time.sleep(EIA_REQUEST_DELAY)

            response = self._make_request(endpoint, params.copy())
            data = response.get("response", {}).get("data", [])

            if not data:
                break

            all_data.extend(data)
            pbar.update(len(data))

            if save_raw:
                self._save_raw_response(
                    endpoint, params, response, page=offset // EIA_PAGE_SIZE
                )

        pbar.close()
        logger.info(f"Fetched {len(all_data):,} rows for {description}")

        return pd.DataFrame(all_data)

    def _save_raw_response(
        self, endpoint: str, params: dict, response: dict, page: int
    ):
        """Save raw API response to disk (Bronze layer)."""
        # Create filename from endpoint and page
        safe_endpoint = endpoint.replace("/", "_").strip("_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_endpoint}_page{page}_{timestamp}.json"
        filepath = self.raw_dir / filename

        with open(filepath, "w") as f:
            json.dump(
                {"endpoint": endpoint, "params": params, "response": response},
                f,
                indent=2,
            )

    # ═══════════════════════════════════════════════════════════════════
    # Convenience methods for each EIA data table
    # ═══════════════════════════════════════════════════════════════════

    def fetch_region_data(
        self,
        respondent: Optional[str] = None,
        data_type: Optional[str] = None,
        start: str = "2021-01-01T00",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch hourly demand/forecast/generation/interchange data.
        (EIA Table 1: electricity/rto/region-data/data/)

        Args:
            respondent: BA code (e.g., 'ERCO'). None = all BAs.
            data_type: 'D' (demand), 'DF' (forecast), 'NG' (generation), 'TI' (interchange)
            start: Start datetime
            end: End datetime

        Returns:
            DataFrame with columns: period, respondent, respondent-name, type, type-name, value
        """
        params = {
            "frequency": "hourly",
            "data[0]": "value",
            "start": start,
        }
        if end:
            params["end"] = end
        if respondent:
            params["facets[respondent][]"] = respondent
        if data_type:
            params["facets[type][]"] = data_type

        type_label = EIA_DATA_TYPES.get(data_type, data_type or "all")
        ba_label = respondent or "all BAs"
        description = f"Region data: {type_label} — {ba_label}"

        return self.fetch_paginated(
            "electricity/rto/region-data/data/",
            params,
            description=description,
        )

    def fetch_fuel_type_data(
        self,
        respondent: Optional[str] = None,
        fueltype: Optional[str] = None,
        start: str = "2021-01-01T00",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch hourly generation by fuel type.
        (EIA Table 2: electricity/rto/fuel-type-data/data/)
        """
        params = {
            "frequency": "hourly",
            "data[0]": "value",
            "start": start,
        }
        if end:
            params["end"] = end
        if respondent:
            params["facets[respondent][]"] = respondent
        if fueltype:
            params["facets[fueltype][]"] = fueltype

        fuel_label = FUEL_TYPES.get(fueltype, fueltype or "all fuels")
        ba_label = respondent or "all BAs"
        description = f"Fuel type: {fuel_label} — {ba_label}"

        return self.fetch_paginated(
            "electricity/rto/fuel-type-data/data/",
            params,
            description=description,
        )

    def fetch_interchange_data(
        self,
        respondent: Optional[str] = None,
        start: str = "2021-01-01T00",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch hourly interchange between BAs.
        (EIA Table 3: electricity/rto/interchange-data/data/)
        """
        params_base = {
            "frequency": "hourly",
            "data[0]": "value",
            "start": start,
        }
        if end:
            params_base["end"] = end
            
        ba_label = respondent or "all BAs"
        df_list = []

        if respondent:
            # 1. Fetch exports (where this BA is the source)
            p1 = params_base.copy()
            p1["facets[fromba][]"] = respondent
            df_from = self.fetch_paginated(
                "electricity/rto/interchange-data/data/",
                p1,
                description=f"Interchange Exports: {ba_label}",
                save_raw=False  # Avoid double-saving duplicates if we want raw data later
            )
            df_list.append(df_from)

            # 2. Fetch imports (where this BA is the destination)
            p2 = params_base.copy()
            p2["facets[toba][]"] = respondent
            df_to = self.fetch_paginated(
                "electricity/rto/interchange-data/data/",
                p2,
                description=f"Interchange Imports: {ba_label}",
                save_raw=False
            )
            df_list.append(df_to)
            
        else:
            # Fetch all global interchanges (no facet)
            df_all = self.fetch_paginated(
                "electricity/rto/interchange-data/data/",
                params_base,
                description=f"Interchange: {ba_label}",
            )
            df_list.append(df_all)

        if not df_list:
            return pd.DataFrame()
            
        combined = pd.concat([d for d in df_list if not d.empty], ignore_index=True) if df_list else pd.DataFrame()
        if not combined.empty:
            combined = combined.drop_duplicates()
            
        return combined

    def fetch_gas_prices(
        self,
        start: str = "2021-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch natural gas spot prices (Henry Hub).
        (EIA Table 4: natural-gas/pri/fut/data/)
        """
        params = {
            "frequency": "daily",
            "data[0]": "value",
            "start": start,
        }
        if end:
            params["end"] = end

        return self.fetch_paginated(
            "natural-gas/pri/fut/data/",
            params,
            description="Natural gas prices (Henry Hub)",
        )

    def fetch_oil_prices(
        self,
        start: str = "2021-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch crude oil spot prices (WTI Cushing, Brent).
        (EIA Table 5: petroleum/pri/spt/data/)
        """
        params = {
            "frequency": "daily",
            "data[0]": "value",
            "start": start,
        }
        if end:
            params["end"] = end

        return self.fetch_paginated(
            "petroleum/pri/spt/data/",
            params,
            description="Crude oil prices (WTI/Brent)",
        )

    def fetch_nuclear_outages(
        self,
        start: str = "2021-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch daily nuclear outage data.
        (EIA Table 6: nuclear-outages/us-nuclear-outages/data/)
        """
        params = {
            "frequency": "daily",
            "data[0]": "capacity",
            "data[1]": "outage",
            "data[2]": "percentOutage",
            "start": start,
        }
        if end:
            params["end"] = end

        return self.fetch_paginated(
            "nuclear-outages/us-nuclear-outages/data/",
            params,
            description="Nuclear outages",
        )

    def list_balancing_authorities(self) -> pd.DataFrame:
        """Fetch the list of all available Balancing Authorities."""
        params = {}
        response = self._make_request(
            "electricity/rto/region-data/facet/respondent/", params
        )
        facets = response.get("response", {}).get("facets", [])
        return pd.DataFrame(facets)


if __name__ == "__main__":
    # Quick test — fetch one page of ERCOT demand data
    client = EIAClient()

    print("Testing EIA API connection...")
    try:
        bas = client.list_balancing_authorities()
        print(f"✅ Found {len(bas)} Balancing Authorities")
        print(bas.head())
    except Exception as e:
        print(f"❌ API test failed: {e}")
