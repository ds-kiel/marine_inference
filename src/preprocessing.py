import json
import numpy as np
import pandas as pd


# Columns the model expects after feature engineering
FEATURE_COLUMNS = [
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "Sal", "Temp", "Oxygen",
]
REQUIRED_COLUMNS = ["date"] + FEATURE_COLUMNS

# Raw columns partners must supply — feature engineering is done by raw_to_features.py
RAW_INPUT_COLUMNS = ["date", "Sal", "Temp", "Oxygen"]

SEQ_LEN = 1008  # 7 days × 24 h × 6 steps/h at 10-minute resolution


def load_scaler_stats(stats_path: str) -> dict:
    with open(stats_path) as f:
        stats = json.load(f)
    return {
        "mean": np.array(stats["mean"], dtype=np.float32),
        "scale": np.array(stats["scale"], dtype=np.float32),
        "feature_names": stats["feature_names"],
    }


def build_input(df: pd.DataFrame, stats: dict) -> tuple:
    """
    Validates and normalizes a 1008-row input DataFrame.

    Accepts either:
    - A fully-featured DataFrame with all 7 feature columns (from raw_to_features.py)
    - Or a pre-built feature CSV in the same format

    Returns
    -------
    x_enc : np.ndarray  shape (1, 1008, 7)  — StandardScaler-normalized features
    x_mark : np.ndarray shape (1, 1008, 4)  — time marks (unused by model)
    forecast_start : pd.Timestamp           — first timestamp of the prediction window
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")

    if len(df) != SEQ_LEN:
        raise ValueError(
            f"Expected exactly {SEQ_LEN} rows (7 days of 10-minute data), got {len(df)}"
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # verify 10-minute cadence (allow ±30s jitter)
    deltas = df["date"].diff().dropna()
    if not (deltas.between(pd.Timedelta("9min30s"), pd.Timedelta("10min30s"))).all():
        raise ValueError("Input timestamps are not evenly 10-minute-spaced.")

    features = df[FEATURE_COLUMNS].values.astype(np.float32)  # (1008, 7)
    features_norm = (features - stats["mean"]) / stats["scale"]

    forecast_start = df["date"].iloc[-1] + pd.Timedelta("10min")

    return (
        features_norm[np.newaxis, :, :],        # (1, 1008, 7)
        np.zeros((1, SEQ_LEN, 4), dtype=np.float32),
        forecast_start,
    )


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])
