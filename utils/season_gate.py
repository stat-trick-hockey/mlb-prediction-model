"""
utils/season_gate.py
Checks whether today falls within the MLB regular season window.
Prevents unnecessary API calls during spring training and offseason.

MLB regular season typically runs:
  - Late March / Early April → late September / early October
  - Postseason runs through late October
"""

from datetime import date


# Approximate regular season windows by year
# Update each year when the schedule is released
SEASON_WINDOWS = {
    2024: (date(2024, 3, 20), date(2024, 10, 1)),
    2025: (date(2025, 3, 27), date(2025, 9, 28)),
    2026: (date(2026, 4, 1),  date(2026, 9, 30)),  # placeholder
}

# Include postseason through end of October
POSTSEASON_EXTRA_DAYS = 30


def is_regular_season(check_date: date = None, include_postseason: bool = True) -> bool:
    """
    Returns True if check_date falls within the MLB regular season.
    Defaults to today.
    """
    if check_date is None:
        check_date = date.today()

    year = check_date.year
    window = SEASON_WINDOWS.get(year)

    if window is None:
        # Fallback: assume April 1 – September 30 for unknown years
        start = date(year, 4, 1)
        end   = date(year, 9, 30)
    else:
        start, end = window

    if include_postseason:
        from datetime import timedelta
        end = end + timedelta(days=POSTSEASON_EXTRA_DAYS)

    return start <= check_date <= end


def is_spring_training(check_date: date = None) -> bool:
    """Returns True if we're currently in spring training."""
    if check_date is None:
        check_date = date.today()

    year = check_date.year
    # Spring training typically runs mid-Feb through late March
    st_start = date(year, 2, 15)
    window = SEASON_WINDOWS.get(year)
    st_end = window[0] if window else date(year, 4, 1)

    return st_start <= check_date < st_end


def get_season_status(check_date: date = None) -> str:
    """Returns a human-readable status string."""
    if check_date is None:
        check_date = date.today()

    if is_regular_season(check_date, include_postseason=False):
        return "regular_season"
    elif is_regular_season(check_date, include_postseason=True):
        return "postseason"
    elif is_spring_training(check_date):
        return "spring_training"
    else:
        return "offseason"


def should_run_predictions(check_date: date = None) -> tuple:
    """
    Returns (should_run: bool, reason: str).
    Use this as the gate at the top of daily_predictions.py.
    """
    if check_date is None:
        check_date = date.today()

    status = get_season_status(check_date)

    if status in ("regular_season", "postseason"):
        return True, f"In {status} — running predictions"
    elif status == "spring_training":
        return False, "Spring training — skipping predictions (no regular season games)"
    else:
        return False, f"Offseason ({check_date}) — skipping predictions"


if __name__ == "__main__":
    today = date.today()
    should_run, reason = should_run_predictions(today)
    status = get_season_status(today)

    print(f"Date:   {today}")
    print(f"Status: {status}")
    print(f"Run:    {should_run}")
    print(f"Reason: {reason}")
