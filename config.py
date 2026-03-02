"""
config.py — shared constants, park factors, team mappings
"""

# Park factors (runs, 100 = neutral, >100 = hitter-friendly)
# Source: FanGraphs 2023 park factors
PARK_FACTORS = {
    "COL": 115,  # Coors Field
    "BOS": 105,  # Fenway
    "CIN": 104,  # Great American Ball Park
    "TEX": 103,  # Globe Life
    "MIL": 102,  # American Family Field
    "PHI": 102,  # Citizens Bank
    "NYY": 101,  # Yankee Stadium
    "HOU": 100,
    "LAD": 99,
    "ATL": 99,
    "CHC": 99,
    "STL": 99,
    "NYM": 98,
    "SF":  98,
    "TB":  98,
    "CLE": 98,
    "MIN": 97,
    "DET": 97,
    "BAL": 97,
    "MIA": 96,
    "SD":  96,
    "SEA": 96,
    "OAK": 96,
    "PIT": 95,
    "LAA": 95,
    "TOR": 95,
    "CWS": 95,
    "KC":  95,
    "WSH": 94,
    "ARI": 94,
}

# Outdoor parks (weather applies)
OUTDOOR_PARKS = {
    "BOS", "CHC", "CIN", "CLE", "COL", "DET", "HOU", "KC",
    "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK", "PHI",
    "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH"
}

# MLB Stats API team abbreviation → team ID
TEAM_IDS = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CWS": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
    "HOU": 117, "KC":  118, "LAA": 108, "LAD": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SD":  135, "SF":  137, "SEA": 136,
    "STL": 138, "TB":  139, "TEX": 140, "TOR": 141, "WSH": 120,
}

TEAM_ID_TO_ABB = {v: k for k, v in TEAM_IDS.items()}

# Venue → team abbreviation
VENUE_TO_TEAM = {
    "Chase Field": "ARI",
    "Truist Park": "ATL",
    "Oriole Park at Camden Yards": "BAL",
    "Fenway Park": "BOS",
    "Wrigley Field": "CHC",
    "Guaranteed Rate Field": "CWS",
    "Great American Ball Park": "CIN",
    "Progressive Field": "CLE",
    "Coors Field": "COL",
    "Comerica Park": "DET",
    "Minute Maid Park": "HOU",
    "Kauffman Stadium": "KC",
    "Angel Stadium": "LAA",
    "Dodger Stadium": "LAD",
    "loanDepot park": "MIA",
    "American Family Field": "MIL",
    "Target Field": "MIN",
    "Citi Field": "NYM",
    "Yankee Stadium": "NYY",
    "Oakland Coliseum": "OAK",
    "Citizens Bank Park": "PHI",
    "PNC Park": "PIT",
    "Petco Park": "SD",
    "Oracle Park": "SF",
    "T-Mobile Park": "SEA",
    "Busch Stadium": "STL",
    "Tropicana Field": "TB",
    "Globe Life Field": "TEX",
    "Rogers Centre": "TOR",
    "Nationals Park": "WSH",
}

# The Odds API sport key
ODDS_SPORT_KEY = "baseball_mlb"
ODDS_REGIONS = "us"
ODDS_MARKETS = "h2h,spreads,totals"

# Model thresholds
EDGE_THRESHOLD = 0.04        # 4% minimum edge to flag a bet
MIN_KELLY_FRACTION = 0.01    # 1% minimum Kelly bet size to display
MAX_KELLY_FRACTION = 0.10    # Cap Kelly at 10% of bankroll

# Training seasons
TRAINING_SEASONS = [2022, 2023, 2024]
VALIDATION_SEASON = 2024

# Rolling windows
ROLLING_STARTS = 5     # pitcher rolling window (starts)
ROLLING_GAMES = 10     # team form rolling window (games)
BULLPEN_DAYS = 7       # bullpen ERA window
FATIGUE_DAYS = [1, 2, 3]  # days back to check pitcher usage
