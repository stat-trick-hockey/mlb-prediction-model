"""
data/barrel_calc.py
Computes barrel classification from raw launch_speed and launch_angle
when the 'barrel' column is missing from Statcast data.

MLB barrel definition:
  - Exit velocity >= 98 mph
  - Launch angle within an EV-dependent window
  - Each +1 mph above 98 widens the angle window by ~1 deg each side
  - Maximum window: 8 to 50 degrees (reached at ~116 mph)

Reference: https://www.mlb.com/glossary/statcast/barrel
"""

import numpy as np
import pandas as pd


def compute_barrel(launch_speed: pd.Series, launch_angle: pd.Series) -> pd.Series:
    """
    Compute barrel flag (0/1) from exit velocity and launch angle.
    Returns a boolean Series aligned with the input.

    Args:
        launch_speed: exit velocity in mph
        launch_angle: launch angle in degrees

    Returns:
        pd.Series of bool (True = barrel)
    """
    ev = launch_speed.astype(float)
    la = launch_angle.astype(float)

    # Must meet minimum EV threshold
    min_ev = 98.0

    # Angle window at minimum EV (98 mph): 26 to 30 degrees
    base_low  = 26.0
    base_high = 30.0

    # Each mph above 98 widens window by 1 deg on each side
    # Capped at EV=116: window becomes 8 to 50 degrees
    ev_above_min = (ev - min_ev).clip(lower=0, upper=18)  # 18 = 116 - 98

    angle_low  = (base_low  - ev_above_min).clip(lower=8)
    angle_high = (base_high + ev_above_min).clip(upper=50)

    barrel = (ev >= min_ev) & (la >= angle_low) & (la <= angle_high)

    # NaN inputs → not a barrel
    barrel = barrel & ev.notna() & la.notna()

    return barrel.astype(float)  # return as float so .mean() gives barrel %


def ensure_barrel_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures a 'barrel' column exists in a Statcast DataFrame.
    If already present, returns as-is.
    If missing but launch_speed and launch_angle exist, computes it.
    Otherwise fills with NaN.

    Args:
        df: raw Statcast DataFrame

    Returns:
        DataFrame with 'barrel' column guaranteed to exist
    """
    if "barrel" in df.columns:
        # Already present — normalize to float (sometimes comes as int or bool)
        df["barrel"] = pd.to_numeric(df["barrel"], errors="coerce")
        return df

    if "launch_speed" in df.columns and "launch_angle" in df.columns:
        print("  'barrel' column missing — computing from launch_speed and launch_angle")
        df = df.copy()
        df["barrel"] = compute_barrel(df["launch_speed"], df["launch_angle"])
        # Set to NaN where either input is NaN
        mask = df["launch_speed"].isna() | df["launch_angle"].isna()
        df.loc[mask, "barrel"] = np.nan
        return df

    # Can't compute — fill with NaN
    print("  WARNING: Cannot compute barrel — launch_speed or launch_angle missing")
    df = df.copy()
    df["barrel"] = np.nan
    return df


def barrel_summary(df: pd.DataFrame) -> dict:
    """
    Quick summary of barrel stats from a Statcast DataFrame.
    Useful for debugging / sanity checking.
    """
    if "barrel" not in df.columns:
        df = ensure_barrel_column(df)

    batted = df[df["type"] == "X"] if "type" in df.columns else df
    total  = len(batted)
    barrels = batted["barrel"].sum()
    barrel_pct = batted["barrel"].mean()

    return {
        "total_batted_balls": total,
        "barrels":            int(barrels) if not pd.isna(barrels) else 0,
        "barrel_pct":         round(barrel_pct, 4) if not pd.isna(barrel_pct) else None,
    }


if __name__ == "__main__":
    # Unit test with known values
    test_cases = [
        # (ev, la, expected_barrel)
        (105, 28, True),   # Classic barrel
        (98,  28, True),   # Minimum EV, center of base window
        (98,  25, False),  # Minimum EV, just below base window
        (116, 10, True),   # Max EV, wide window
        (116, 7,  False),  # Max EV, just below min angle
        (90,  28, False),  # Below minimum EV
        (100, 50, False),  # EV ok but angle too high
        (100, 0,  False),  # EV ok but angle too low (groundball)
    ]

    ev_series = pd.Series([t[0] for t in test_cases], dtype=float)
    la_series = pd.Series([t[1] for t in test_cases], dtype=float)
    results   = compute_barrel(ev_series, la_series)

    print("Barrel computation unit tests:")
    all_pass = True
    for i, (ev, la, expected) in enumerate(test_cases):
        result = bool(results[i])
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_pass = False
        print(f"  {status}  EV={ev} LA={la:>3}  expected={expected}  got={result}")

    print(f"\n{'All tests passed ✓' if all_pass else 'Some tests failed ✗'}")
