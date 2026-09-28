import numpy as np
import pandas as pd


def inverse_transform_oxygen(pred_norm: np.ndarray, stats: dict) -> np.ndarray:
    """
    Reverse the StandardScaler transform for the Oxygen column only.
    pred_norm : (pred_len,) or (1, pred_len, 1)
    Returns   : (pred_len,) in original units (mg/L)
    """
    oxy_mean = stats["mean"][-1]
    oxy_std = stats["scale"][-1]
    flat = pred_norm.reshape(-1)
    return flat * oxy_std + oxy_mean


def build_output_dataframe(
    pred_mg_per_l: np.ndarray,
    forecast_start: pd.Timestamp,
    station_id: str = "Bocknis_Eck_2",
) -> pd.DataFrame:
    """
    Wraps raw predictions into a DataFrame with timestamps.

    pred_mg_per_l  : (432,) array of oxygen values in mg/L
    forecast_start : timestamp of the first predicted step (10-minute resolution)
    """
    pred_len = len(pred_mg_per_l)
    timestamps = pd.date_range(forecast_start, periods=pred_len, freq="10min")
    return pd.DataFrame({
        "timestamp": timestamps,
        "station_id": station_id,
        "oxygen_mg_per_l": pred_mg_per_l,
    })
