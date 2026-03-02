# MLB Prediction Model

A full-stack MLB betting prediction system with three separate models (moneyline, run line, O/U), Statcast feature engineering, walk-forward backtesting, and automated daily predictions via GitHub Actions.

---

## Architecture

```
Feature Pipeline → [Moneyline Model]  → Win probability %
                 → [Run Line Model]   → Cover probability %
                 → [O/U Model]        → Predicted total + over/under %
                 
All three outputs → Edge Calculator vs. Vegas lines
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
# Edit .env and add:
# ODDS_API_KEY=...     (from https://the-odds-api.com — free tier)
# WEATHER_API_KEY=...  (from https://www.weatherapi.com — free tier)
```

### 3. Build data + train models (first-time setup)

```bash
python setup_and_train.py
```

This runs all 26 steps:
- Fetches 2022–2024 FanGraphs and Statcast data
- Builds the historical feature matrix (~7,000 rows)
- Trains and calibrates all three models
- Runs walk-forward backtest on 2024 season
- Generates today's predictions

**Note:** The full setup takes 30–60 minutes due to Statcast data download volume.

### 4. Run daily predictions manually

```bash
python predict/daily_predictions.py
# Or for a specific date:
python predict/daily_predictions.py --date 2025-04-15
```

### 5. Track yesterday's results

```bash
python predict/results_tracker.py
```

---

## File Structure

```
mlb-prediction-model/
├── config.py                        # Park factors, team IDs, constants
├── requirements.txt
├── setup_and_train.py               # One-shot setup runner
│
├── data/
│   ├── fetch_schedule.py            # MLB Stats API (free, no auth)
│   ├── fetch_fangraphs.py           # FanGraphs via pybaseball
│   ├── fetch_statcast.py            # Baseball Savant via pybaseball
│   ├── fetch_odds.py                # The Odds API (moneyline/RL/O/U)
│   ├── fetch_weather.py             # WeatherAPI for outdoor parks
│   ├── raw/                         # Daily raw data cache
│   └── processed/                   # Built feature matrices + model inputs
│       ├── fangraphs/{season}/
│       ├── statcast/{season}/
│       └── training_data.csv
│
├── features/
│   ├── pitcher_features.py          # SP: ERA, FIP, xFIP, K%, rolling window
│   ├── bullpen_features.py          # BP: 7-day ERA, fatigue, closer avail.
│   ├── team_form_features.py        # W/L, run diff, H/A splits, vs. LHP/RHP
│   └── build_feature_matrix.py      # Assembles all features → one row/game
│
├── models/
│   ├── calibrate.py                 # Isotonic calibration utilities
│   ├── train_ou.py                  # XGBRegressor for total runs
│   ├── train_moneyline.py           # XGBClassifier for home win
│   ├── train_runline.py             # XGBClassifier for home covers -1.5
│   ├── *.pkl                        # Saved trained models (after training)
│   └── plots/                       # Calibration and residual plots
│
├── predict/
│   ├── daily_predictions.py         # Main daily runner
│   ├── edge_calculator.py           # Edge % and Kelly sizing vs. Vegas
│   ├── results_tracker.py           # Nightly accuracy logging
│   └── output/                      # Daily prediction CSVs, JSONs, reports
│
├── backtest/
│   ├── backtest.py                  # Walk-forward backtester
│   └── results/                     # Backtest output and plots
│
└── .github/workflows/
    ├── daily_predict.yml            # 10am ET daily predictions
    └── monthly_retrain.yml          # 1st of month model retrain
```

---

## Data Sources

| Source | What it provides | Cost |
|---|---|---|
| MLB Stats API (`statsapi.mlb.com`) | Schedules, scores, rosters, probable pitchers | Free, no auth |
| FanGraphs (via `pybaseball`) | ERA, FIP, xFIP, K%, BB%, wRC+, wOBA | Free |
| Baseball Savant (via `pybaseball`) | Statcast: xwOBA, barrel%, exit velocity, hard-hit% | Free |
| The Odds API | Moneyline, run line, O/U lines | Free tier: 500 req/month |
| WeatherAPI | Temperature, wind speed/direction | Free tier: 1M calls/month |

---

## Features

### Starting Pitcher (strongest signal)
- `era`, `fip`, `xfip`, `k_pct`, `bb_pct`, `whip` — season-to-date
- `rolling_era`, `rolling_fip`, `rolling_k_pct` — last 5 starts
- `days_rest` — days since last outing
- `statcast_xwoba`, `statcast_barrel_pct`, `statcast_hard_hit_pct` — quality of contact allowed

### Bullpen
- `bp_era_7d`, `bp_fip_7d` — 7-day bullpen ERA/FIP
- `bp_fatigued_1d/2d/3d` — pitcher appearance counts by days back
- `bp_closer_available` — flag: closer hasn't pitched in 2 days
- `bp_hl_usage_pct` — high-leverage reliever usage rate

### Team Form
- `wins_last3/5/10`, `win_pct_last10` — rolling win records
- `rd_last10` — run differential last 10 games
- `avg_runs_scored`, `avg_runs_allowed` — last 10 games
- `ha_win_pct`, `ha_rd` — home/away split win% and run diff
- `vs_hand_win_pct` — win % vs. LHP or RHP (based on today's SP)
- `rest_days` — days since last game

### Statcast (Team)
- `xwoba`, `barrel_pct`, `hard_hit_pct`, `avg_ev` — batting quality of contact
- Park factor (100 = neutral, >100 = hitter-friendly)

### Weather (outdoor parks)
- `temp_f`, `wind_mph`
- `wind_out_to_cf` — wind component blowing toward center field (positive = hitter-friendly)
- `precip_chance`

---

## Models

All three models use **XGBoost** with a **date-based train/validation split** (no random leakage). Classifiers are wrapped with **isotonic calibration** to fix probability overconfidence.

| Model | Type | Target | Val Metric | Typical performance |
|---|---|---|---|---|
| O/U | XGBRegressor | Total runs | MAE | ~2.5–2.8 runs MAE |
| Moneyline | XGBClassifier + calibration | Home win (0/1) | Brier score, log loss | ~0.25 Brier |
| Run line | XGBClassifier + calibration | Home covers -1.5 | Brier score | ~0.25 Brier |

---

## Edge Calculator

Converts model probabilities to betting edge using:

```python
edge = model_probability - fair_implied_probability
# where fair_implied removes the bookmaker vig from the raw lines

kelly = (b * p - q) / b   # Kelly criterion bet sizing
# b = decimal odds - 1
# p = model probability
# q = 1 - p
```

Games with **edge ≥ 4%** are flagged. Kelly fraction is capped at 10% and used at 25% (quarter-Kelly) for conservative sizing.

---

## Automation (GitHub Actions)

**Daily predictions** run at 10am ET each day:
1. Fetch schedule + probable pitchers
2. Fetch current Vegas lines
3. Build features and run all three models
4. Commit `predict/output/YYYY-MM-DD_predictions.csv` and `_report.txt`

**Nightly results tracking** logs prediction accuracy to `predict/output/accuracy_log.csv`.

**Monthly retraining** on the 1st of each month pulls fresh FanGraphs + Statcast data and retrains all three models.

### Setup GitHub Secrets
In your repo: **Settings → Secrets → Actions**
- `ODDS_API_KEY`
- `WEATHER_API_KEY`

---

## Tips

**Start with the O/U model** — it's the most tractable. If your predicted total is within 0.5 runs of Vegas on average, the model is working.

**Expect 53–56% accuracy on run line** — the market is efficient. Even professional bettors rarely exceed 58%. Profitability comes from edge size, not just win rate.

**Watch for model drift** — pitcher performance in April vs. August is different. The monthly retrain handles this, but you can also track rolling accuracy in `accuracy_log.csv`.

**Calibration > accuracy** — a model that says 60% and is right 60% of the time is more valuable than one that says 80% and is only right 60% of the time. Always check your Brier score.
