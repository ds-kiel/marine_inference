"""
Convert raw partner-supplied sensor data into the feature DataFrame that
preprocessing.build_input() expects.

Partners supply only what their instruments measure:
    date, Sal, Temp, Oxygen  (10-minute or higher-resolution CSV)

This module adds the derived features so they never need to know about them:
    hour_sin, hour_cos  — diurnal cycle
    day_sin,  day_cos   — seasonal/annual cycle

Pipeline
--------
  1. Parse and validate the raw CSV (or DataFrame)
  2. Resample to 10-minute grid
  3. Quality filter: mark physically implausible values as NaN
  4. Forward-interpolate short gaps (≤ 1 hour = 6 steps)
  5. Fill longer gaps with training-set column means from scaler_stats.json
  6. Add cyclical time features from the index timestamps
  7. Order columns to match the training layout
  8. Return the most recent 1008 rows as a DataFrame with 'date' column

The returned DataFrame is ready to pass directly to predict().
"""

import numpy as np
import pandas as pd

RESAMPLE_FREQ = "10min"
INTERP_LIMIT  = 6        # forward-only, 6 steps = 1 hour
SEQ_LEN       = 1008     # 7 days at 10-minute resolution

# Physical plausibility windows — same as data_prep.py
_PLAUSIBILITY = {
    "Temp":   (-5.0,  35.0),
    "Sal":    ( 0.0,  40.0),
    "Oxygen": ( 0.0,  20.0),
}

# Final column order matching the NoWind training layout
FEATURE_ORDER = ["hour_sin", "hour_cos", "day_sin", "day_cos", "Sal", "Temp", "Oxygen"]


def _parse_input(source, required_cols, name="input"):
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)

    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}. Got: {list(df.columns)}")

    date_col = next(
        (c for c in df.columns if c.lower() in ("date", "timestamp", "time", "datetime")),
        None,
    )
    if date_col is None:
        raise ValueError(f"{name}: no date/timestamp column found.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = "date"

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _quality_filter(df):
    for col, (lo, hi) in _PLAUSIBILITY.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    return df


def _resample_and_interpolate(df):
    df = df.resample(RESAMPLE_FREQ).asfreq()
    df = df.interpolate(method="time", limit=INTERP_LIMIT, limit_direction="forward")
    return df


def _fill_remaining_gaps(df, stats):
    """Fallback: fill any surviving NaNs with training-set column means."""
    for feat, mean_val in zip(stats["feature_names"], stats["mean"]):
        if feat in df.columns and df[feat].isna().any():
            df[feat] = df[feat].fillna(float(mean_val))
    return df


def _add_cyclical_time_features(df):
    t = df.index
    df["hour_sin"] = np.sin(2 * np.pi * t.hour       / 24)
    df["hour_cos"] = np.cos(2 * np.pi * t.hour       / 24)
    df["day_sin"]  = np.sin(2 * np.pi * t.dayofyear  / 365)
    df["day_cos"]  = np.cos(2 * np.pi * t.dayofyear  / 365)
    return df


def prepare_features(station_input, stats):
    """
    Build the 7-feature input DataFrame from raw station observations.

    Parameters
    ----------
    station_input : str | pd.DataFrame
        CSV path or DataFrame with columns: date, Sal, Temp, Oxygen
        Must cover at least 7 continuous days at 10-minute (or finer) resolution.
        Short gaps (≤ 1 hour) are interpolated; longer gaps use training-set means.
    stats : dict
        Scaler stats from load_scaler_stats(). Used for gap-filling fallback.

    Returns
    -------
    pd.DataFrame
        Exactly 1008 rows with columns [date, hour_sin, hour_cos, day_sin, day_cos,
        Sal, Temp, Oxygen] — ready to pass to predict().
    """
    df = _parse_input(station_input, ["Sal", "Temp", "Oxygen"])
    df = _quality_filter(df)
    df = _resample_and_interpolate(df)
    df = _add_cyclical_time_features(df)
    df = _fill_remaining_gaps(df, stats)

    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns after feature engineering: {missing}")

    df = df[FEATURE_ORDER]

    if len(df) < SEQ_LEN:
        raise ValueError(
            f"After resampling, only {len(df)} rows available. "
            f"Need at least {SEQ_LEN} (7 days at 10-minute resolution)."
        )

    # Take the most recent 1008 rows
    df = df.iloc[-SEQ_LEN:]
    return df.reset_index()  # restores 'date' as a column for build_input()
