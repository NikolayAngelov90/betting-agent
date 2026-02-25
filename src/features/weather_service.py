"""Weather features via Open-Meteo free API (no API key required)."""

import json
import urllib.request
import urllib.parse
from datetime import date
from typing import Optional, Tuple, Dict

from src.utils.logger import get_logger

logger = get_logger()

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# In-memory caches (persist for the lifetime of the process)
_GEO_CACHE: Dict[str, Optional[Tuple[float, float]]] = {}
_WEATHER_CACHE: Dict[str, Dict] = {}

# Default values returned when weather data is unavailable
_DEFAULTS = {
    "weather_temp_c": 12.0,
    "weather_wind_kmh": 10.0,
    "weather_precip_mm": 0.0,
    "weather_is_raining": 0,
    "weather_is_windy": 0,
    "weather_available": 0,
}


def _geocode(city: str) -> Optional[Tuple[float, float]]:
    """Return (latitude, longitude) for a city name. Results cached."""
    key = city.lower().strip()
    if key in _GEO_CACHE:
        return _GEO_CACHE[key]
    try:
        params = urllib.parse.urlencode({"name": city, "count": 1, "format": "json"})
        url = f"{_GEOCODING_URL}?{params}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        results = data.get("results", [])
        if results:
            lat = float(results[0]["latitude"])
            lon = float(results[0]["longitude"])
            _GEO_CACHE[key] = (lat, lon)
            return (lat, lon)
    except Exception as exc:
        logger.debug(f"Geocoding failed for '{city}': {exc}")
    _GEO_CACHE[key] = None
    return None


def _fetch_daily_weather(lat: float, lon: float, match_date: date) -> Dict:
    """Fetch daily weather summary for coordinates and date from Open-Meteo."""
    cache_key = f"{lat:.3f},{lon:.3f},{match_date}"
    if cache_key in _WEATHER_CACHE:
        return _WEATHER_CACHE[cache_key]

    date_str = match_date.strftime("%Y-%m-%d")
    try:
        params = urllib.parse.urlencode({
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,precipitation_sum,windspeed_10m_max",
            "timezone": "auto",
            "start_date": date_str,
            "end_date": date_str,
            "format": "json",
        })
        url = f"{_FORECAST_URL}?{params}"
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())

        daily = data.get("daily", {})
        temps = daily.get("temperature_2m_max", [None])
        precips = daily.get("precipitation_sum", [None])
        winds = daily.get("windspeed_10m_max", [None])

        temp = float(temps[0]) if temps and temps[0] is not None else 12.0
        precip = float(precips[0]) if precips and precips[0] is not None else 0.0
        wind = float(winds[0]) if winds and winds[0] is not None else 10.0

        result = {
            "weather_temp_c": round(temp, 1),
            "weather_wind_kmh": round(wind, 1),
            "weather_precip_mm": round(precip, 1),
            "weather_is_raining": 1 if precip > 1.0 else 0,
            "weather_is_windy": 1 if wind > 30.0 else 0,
            "weather_available": 1,
        }
        _WEATHER_CACHE[cache_key] = result
        return result
    except Exception as exc:
        logger.debug(f"Weather fetch failed for ({lat:.3f},{lon:.3f},{date_str}): {exc}")
        result = dict(_DEFAULTS)
        _WEATHER_CACHE[cache_key] = result
        return result


class WeatherService:
    """Fetches match-day weather features using the Open-Meteo free API.

    No API key is required. Geocoding and weather data are both cached
    in-process to avoid redundant HTTP calls for the same venue/date.

    Returned feature keys:
        weather_temp_c      — max temperature (°C)
        weather_wind_kmh    — max wind speed (km/h)
        weather_precip_mm   — total precipitation (mm)
        weather_is_raining  — 1 if precip > 1 mm, else 0
        weather_is_windy    — 1 if wind > 30 km/h, else 0
        weather_available   — 1 if data was successfully fetched, else 0
    """

    def get_match_weather(self, venue: Optional[str], match_date: date) -> Dict:
        """Return weather features for a venue and date.

        Falls back to neutral defaults when the venue cannot be geocoded or
        when the Open-Meteo API is unavailable.

        Args:
            venue: Stadium/city name (e.g. "Old Trafford" or "Manchester").
                   If None or empty, defaults are returned immediately.
            match_date: Date of the match.
        """
        if not venue:
            return dict(_DEFAULTS)

        coords = _geocode(venue)
        if coords is None and " " in venue:
            # Try just the first token as a city name fallback
            first = venue.split()[0]
            if len(first) >= 3:
                coords = _geocode(first)

        if coords is None:
            return dict(_DEFAULTS)

        lat, lon = coords
        return _fetch_daily_weather(lat, lon, match_date)
