"""
SENTINEL — NLP Region Resolver
Maps unstructured geographic text (e.g., 'West Texas') to EIA Balancing Authority codes (e.g., 'ERCO').
Uses an in-memory inverted index for O(1) latency, with fuzzy matching fallback.
Designed specifically for high-throughput batch NLP workloads.
"""

from typing import List, Set, Dict
from loguru import logger
from rapidfuzz import fuzz
from sqlalchemy import text

from src.database.connection import get_engine

class RegionResolver:
    """Resolves geographic entities and commodities to Balancing Authority codes."""
    
    def __init__(self):
        self._region_to_bas: Dict[str, Set[str]] = {}
        self._fuel_profiles: Dict[str, Dict] = {}
        self._all_names: List[str] = []
        self._load_mapping_from_db()

    def _load_mapping_from_db(self):
        """Loads the entire ba_region_mapping table into an inverted index in RAM."""
        try:
            engine = get_engine()
            with engine.connect() as conn:
                result = conn.execute(text("SELECT ba_code, region_names, fuel_profile FROM clean.ba_region_mapping"))
                rows = result.fetchall()
                
            for row in rows:
                ba_code = row[0]
                region_names = row[1] or []
                fuel_profile = row[2] or {}
                
                self._fuel_profiles[ba_code] = fuel_profile
                
                for name in region_names:
                    key = name.lower()
                    if key not in self._region_to_bas:
                        self._region_to_bas[key] = set()
                    self._region_to_bas[key].add(ba_code)
                    
            self._all_names = list(self._region_to_bas.keys())
            logger.info(f"RegionResolver loaded {len(self._all_names)} regional keywords targeting {len(self._fuel_profiles)} BAs.")
            
        except Exception as e:
            logger.error(f"Failed to load BA mappings: {e}")

    def resolve(self, entities: List[str], is_commodity: bool = False, fuel: str = None) -> List[str]:
        """
        Maps a list of extracted NER entities to a distinct list of BA codes.
        
        Args:
            entities: List of location names (e.g. ['Texas', 'Houston'])
            is_commodity: True if the article is about a global energy market event
            fuel: The fuel type abbreviation if known (e.g., 'NG' for Natural Gas)
        
        Returns:
            List of affected BA codes.
        """
        matched = set()
        
        # 1. Direct O(1) lookup
        for entity in entities:
            key = entity.lower().strip()
            if key in self._region_to_bas:
                matched.update(self._region_to_bas[key])
            else:
                # 2. Fuzzy match fallback
                for name in self._all_names:
                    if fuzz.ratio(key, name) > 85:
                        matched.update(self._region_to_bas[name])
                        break
        
        # 3. Handle global market shocks based on fuel dependency
        if not matched and is_commodity and fuel:
            for ba, profile in self._fuel_profiles.items():
                if profile.get(fuel, 0.0) > 0.30:  # If BA relies > 30% on this fuel
                    matched.add(ba)
                    
        return list(matched)

if __name__ == "__main__":
    # Test instantiation
    resolver = RegionResolver()
