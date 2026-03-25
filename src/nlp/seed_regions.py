"""
SENTINEL — Seed NLP Mappings
Populates the clean.ba_region_mapping table with core mapping data.
"""

from sqlalchemy import text
from loguru import logger
from src.database.connection import get_engine

SEED_DATA = [
    {
        "ba_code": "ERCO",
        "ba_name": "Electric Reliability Council of Texas",
        "state_codes": ["TX"],
        "region_names": ["Texas", "ERCOT", "West Texas", "Gulf Coast", "Houston", "Dallas", "Austin"],
        "fuel_profile": '{"NG": 0.45, "WND": 0.25, "COL": 0.15, "NUC": 0.10, "SUN": 0.05}'
    },
    {
        "ba_code": "PJM",
        "ba_name": "PJM Interconnection",
        "state_codes": ["PA", "NJ", "MD", "DE", "OH", "VA", "WV", "IL", "IN", "KY", "MI", "NC", "TN"],
        "region_names": ["PJM", "Mid-Atlantic", "Appalachia", "Pennsylvania", "New Jersey", "Ohio", "Virginia"],
        "fuel_profile": '{"NG": 0.40, "NUC": 0.33, "COL": 0.20, "WND": 0.04, "SUN": 0.02}'
    },
    {
        "ba_code": "CISO",
        "ba_name": "California Independent System Operator",
        "state_codes": ["CA"],
        "region_names": ["California", "CAISO", "Los Angeles", "San Francisco", "Silicon Valley"],
        "fuel_profile": '{"NG": 0.40, "SUN": 0.25, "WAT": 0.15, "WND": 0.10, "NUC": 0.08}'
    },
    {
        "ba_code": "MISO",
        "ba_name": "Midcontinent Independent System Operator",
        "state_codes": ["MN", "WI", "MI", "IA", "IL", "IN", "MO", "AR", "MS", "LA"],
        "region_names": ["Midwest", "MISO", "Michigan", "Minnesota", "Wisconsin", "Louisiana", "Gulf Coast"],
        "fuel_profile": '{"COL": 0.45, "NG": 0.30, "WND": 0.15, "NUC": 0.08}'
    },
    {
        "ba_code": "NYIS",
        "ba_name": "New York Independent System Operator",
        "state_codes": ["NY"],
        "region_names": ["New York", "NYISO", "NYC", "Long Island", "Upstate New York"],
        "fuel_profile": '{"NG": 0.45, "NUC": 0.25, "WAT": 0.20, "WND": 0.05}'
    }
]

def seed_regions():
    logger.info("Seeding mapping table...")
    engine = get_engine()
    with engine.begin() as conn:
        for data in SEED_DATA:
            query = text("""
                INSERT INTO clean.ba_region_mapping 
                (ba_code, ba_name, state_codes, region_names, fuel_profile) 
                VALUES (:ba_code, :ba_name, :state_codes, :region_names, :fuel_profile)
                ON CONFLICT (ba_code) DO NOTHING
            """)
            conn.execute(query, data)
    logger.info(f"✅ Seeding complete. Inserted {len(SEED_DATA)} core BA mappings.")

if __name__ == "__main__":
    seed_regions()
