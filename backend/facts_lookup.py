"""Lookup ground truth facts from CSV/Chroma."""

from __future__ import annotations

import pandas as pd
from typing import Dict, Any, Optional

from . import config
from .chroma_setup import init_chroma_client

# Cache for CSV data
_csv_cache: Optional[pd.DataFrame] = None


def _load_csv_cache() -> pd.DataFrame:
    """Load and cache the CSV data."""
    global _csv_cache
    if _csv_cache is None:
        _csv_cache = pd.read_csv(config.PLANETS_CSV_PATH)
    return _csv_cache


def _normalize_planet_name(name: str) -> str:
    """Normalize planet name to match CSV column names."""
    name_lower = name.lower().strip()
    # Handle common variations
    if name_lower == "moon":
        return "Moon"
    return name_lower.capitalize()


def get_ground_truth_facts(planet_name: str) -> Dict[str, Any]:
    """
    Return canonical facts for this planet from planets_full_clean.csv.
    
    Returns normalized dict with same keys as extract_planet_facts:
    {
      "planet": "Mars",
      "number_of_moons": 2,
      "surface_pressure_bars": 0.01,
      "mean_temperature_celsius": -65,
      ...
    }
    """
    if not planet_name:
        return {}
    
    df = _load_csv_cache()
    planet_normalized = _normalize_planet_name(planet_name)
    
    # Check if planet exists in CSV
    if planet_normalized not in df.columns:
        return {"planet": planet_name}
    
    facts: Dict[str, Any] = {"planet": planet_normalized}
    
    # Extract relevant fields
    for _, row in df.iterrows():
        parameter = str(row["Parameter"]).lower()
        value = row[planet_normalized]
        
        if pd.isna(value):
            continue
        
        # Map CSV parameters to normalized fields
        if "moon" in parameter:
            try:
                facts["number_of_moons"] = int(float(str(value)))
            except (ValueError, TypeError):
                pass
        
        elif "temperature" in parameter and "mean" in parameter:
            try:
                facts["mean_temperature_celsius"] = float(str(value))
            except (ValueError, TypeError):
                pass
        
        elif "pressure" in parameter and "surface" in parameter:
            try:
                # CSV has pressure in bars
                facts["surface_pressure_bars"] = float(str(value).replace(",", ""))
            except (ValueError, TypeError):
                pass
        
        elif parameter == "gravity":
            try:
                facts["gravity_mps2"] = float(str(value))
            except (ValueError, TypeError):
                pass
        
        elif "orbital period" in parameter:
            try:
                facts["orbital_period_days"] = float(str(value).replace(",", ""))
            except (ValueError, TypeError):
                pass
        
        elif "distance" in parameter and "sun" in parameter:
            try:
                # CSV has distance in 10^6 km
                distance_10e6km = float(str(value).replace(",", "").replace("*", ""))
                facts["distance_from_sun_km"] = distance_10e6km * 1_000_000
            except (ValueError, TypeError):
                pass
        
        elif parameter == "mass":
            try:
                # CSV has mass in 10^24 kg
                facts["mass_10e24kg"] = float(str(value))
            except (ValueError, TypeError):
                pass
        
        elif parameter == "diameter":
            try:
                facts["diameter_km"] = float(str(value).replace(",", ""))
            except (ValueError, TypeError):
                pass
    
    return facts


def get_ground_truth_from_chroma(planet_name: str) -> Dict[str, Any]:
    """
    Alternative: Get facts from Chroma metadata.
    Falls back to CSV if Chroma lookup fails.
    """
    try:
        client = init_chroma_client()
        collection = client.get_collection(config.CHROMA_COLLECTIONS.planets_facts)
        
        # Query for this planet
        results = collection.get(
            where={"planet": planet_name},
            limit=1,
        )
        
        if results["metadatas"]:
            metadata = results["metadatas"][0]
            facts: Dict[str, Any] = {"planet": planet_name}
            
            # Extract numeric fields from metadata
            for key, value in metadata.items():
                if key == "planet":
                    continue
                try:
                    if "moon" in key.lower():
                        facts["number_of_moons"] = int(float(str(value)))
                    elif "temperature" in key.lower():
                        facts["mean_temperature_celsius"] = float(str(value))
                    elif "pressure" in key.lower():
                        facts["surface_pressure_bars"] = float(str(value))
                    elif "gravity" in key.lower():
                        facts["gravity_mps2"] = float(str(value))
                    elif "orbital" in key.lower() and "period" in key.lower():
                        facts["orbital_period_days"] = float(str(value))
                except (ValueError, TypeError):
                    continue
            
            return facts
    except Exception:
        pass
    
    # Fallback to CSV
    return get_ground_truth_facts(planet_name)

