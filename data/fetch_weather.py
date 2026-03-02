"""
data/fetch_weather.py
Pulls game-time weather for outdoor MLB parks using WeatherAPI.
Sign up (free tier) at https://www.weatherapi.com
Wind blowing out to CF at 15+ mph is a meaningful O/U signal.
"""

import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

WEATHER_API_KEY  = os.getenv("WEATHER_API_KEY", "")
WEATHER_BASE_URL = "https://api.weatherapi.com/v1"

# Coordinates for outdoor MLB parks
PARK_COORDS = {
    "BOS": {"lat": 42.3467, "lon": -71.0972, "cf_bearing": 95},   # CF bearing in degrees
    "CHC": {"lat": 41.9484, "lon": -87.6553, "cf_bearing": 5},
    "CIN": {"lat": 39.0979, "lon": -84.5067, "cf_bearing": 30},
    "CLE": {"lat": 41.4962, "lon": -81.6853, "cf_bearing": 285},
    "COL": {"lat": 39.7559, "lon": -104.9942,"cf_bearing": 5},
    "DET": {"lat": 42.3390, "lon": -83.0485, "cf_bearing": 340},
    "KC":  {"lat": 39.0517, "lon": -94.4803, "cf_bearing": 5},
    "LAD": {"lat": 34.0739, "lon": -118.2400,"cf_bearing": 5},
    "MIA": {"lat": 25.7781, "lon": -80.2197, "cf_bearing": 310},
    "NYM": {"lat": 40.7571, "lon": -73.8458, "cf_bearing": 5},
    "NYY": {"lat": 40.8296, "lon": -73.9262, "cf_bearing": 5},
    "OAK": {"lat": 37.7516, "lon": -122.2005,"cf_bearing": 340},
    "PHI": {"lat": 39.9061, "lon": -75.1665, "cf_bearing": 5},
    "PIT": {"lat": 40.4469, "lon": -80.0057, "cf_bearing": 355},
    "SD":  {"lat": 32.7076, "lon": -117.1568,"cf_bearing": 330},
    "SF":  {"lat": 37.7786, "lon": -122.3893,"cf_bearing": 80},
    "SEA": {"lat": 47.5915, "lon": -122.3325,"cf_bearing": 15},
    "STL": {"lat": 38.6226, "lon": -90.1928, "cf_bearing": 5},
    "TB":  {"lat": 27.7682, "lon": -82.6534, "cf_bearing": 340},
    "TEX": {"lat": 32.7513, "lon": -97.0832, "cf_bearing": 5},
    "TOR": {"lat": 43.6414, "lon": -79.3894, "cf_bearing": 5},
    "WSH": {"lat": 38.8730, "lon": -77.0074, "cf_bearing": 5},
}


def fetch_game_weather(team_abb: str, game_datetime: str) -> dict:
    """
    Fetch weather forecast for a specific park at game time.

    Returns dict with:
    - temp_f: temperature in Fahrenheit
    - wind_mph: wind speed
    - wind_dir_deg: wind direction in degrees
    - wind_out_to_cf: component of wind blowing out to CF (positive = out)
    - precip_chance: chance of precipitation %
    - condition: weather condition text
    """
    coords = PARK_COORDS.get(team_abb)
    if not coords:
        return _default_weather()

    if not WEATHER_API_KEY:
        return _mock_weather(team_abb)

    try:
        url = f"{WEATHER_BASE_URL}/forecast.json"
        params = {
            "key":  WEATHER_API_KEY,
            "q":    f"{coords['lat']},{coords['lon']}",
            "days": 2,
            "aqi":  "no",
            "alerts": "no",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Find the closest forecast hour to game time
        target_hour = pd.to_datetime(game_datetime).hour
        forecast_day = data["forecast"]["forecastday"][0]

        # Try today first, then tomorrow
        for fday in data["forecast"]["forecastday"]:
            game_date = pd.to_datetime(game_datetime).strftime("%Y-%m-%d")
            if fday["date"] == game_date:
                forecast_day = fday
                break

        hours = forecast_day.get("hour", [])
        closest_hour = min(hours, key=lambda h: abs(pd.to_datetime(h["time"]).hour - target_hour))

        wind_mph = closest_hour.get("wind_mph", 0)
        wind_dir_deg = closest_hour.get("wind_degree", 0)
        cf_bearing = coords.get("cf_bearing", 0)

        # Calculate component of wind blowing out to center field
        # Positive = blowing out (increases offense), Negative = blowing in
        import math
        angle_diff = (wind_dir_deg - cf_bearing + 180) % 360 - 180
        wind_out_to_cf = wind_mph * math.cos(math.radians(angle_diff))

        return {
            "temp_f":          closest_hour.get("temp_f", 72),
            "wind_mph":        wind_mph,
            "wind_dir_deg":    wind_dir_deg,
            "wind_out_to_cf":  round(wind_out_to_cf, 1),
            "precip_chance":   closest_hour.get("chance_of_rain", 0),
            "condition":       closest_hour.get("condition", {}).get("text", ""),
            "humidity":        closest_hour.get("humidity", 50),
        }

    except Exception as e:
        print(f"  Weather fetch failed for {team_abb}: {e}")
        return _default_weather()


def fetch_weather_for_slate(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch weather for all outdoor games in today's schedule.
    Merges weather back into schedule DataFrame.
    """
    from config import OUTDOOR_PARKS

    weather_rows = []
    for _, game in schedule_df.iterrows():
        home_abb = game.get("home_team_abb", "")
        is_outdoor = home_abb in OUTDOOR_PARKS
        game_dt = game.get("game_datetime", "")

        if is_outdoor and game_dt:
            w = fetch_game_weather(home_abb, game_dt)
        else:
            w = _default_weather()
            w["is_dome"] = True

        w["game_pk"] = game.get("game_pk")
        weather_rows.append(w)

    weather_df = pd.DataFrame(weather_rows)
    return schedule_df.merge(weather_df, on="game_pk", how="left")


def _default_weather() -> dict:
    """Neutral weather (dome or data unavailable)."""
    return {
        "temp_f":          72,
        "wind_mph":        0,
        "wind_dir_deg":    0,
        "wind_out_to_cf":  0,
        "precip_chance":   0,
        "condition":       "Indoor/Dome",
        "humidity":        50,
        "is_dome":         False,
    }


def _mock_weather(team_abb: str) -> dict:
    np.random.seed(hash(team_abb) % 1000)
    wind_mph = np.random.uniform(0, 18)
    wind_dir = np.random.uniform(0, 360)
    cf_bearing = PARK_COORDS.get(team_abb, {}).get("cf_bearing", 5)
    import math
    angle_diff = (wind_dir - cf_bearing + 180) % 360 - 180
    wind_out = wind_mph * math.cos(math.radians(angle_diff))

    return {
        "temp_f":          round(np.random.uniform(55, 92), 1),
        "wind_mph":        round(wind_mph, 1),
        "wind_dir_deg":    round(wind_dir, 1),
        "wind_out_to_cf":  round(wind_out, 1),
        "precip_chance":   int(np.random.uniform(0, 30)),
        "condition":       "Partly Cloudy",
        "humidity":        int(np.random.uniform(30, 75)),
        "is_dome":         False,
    }


if __name__ == "__main__":
    print("Testing weather fetch for Wrigley Field (CHC)...")
    w = fetch_game_weather("CHC", "2025-04-15T19:05:00Z")
    for k, v in w.items():
        print(f"  {k}: {v}")
