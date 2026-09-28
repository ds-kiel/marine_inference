"""
Public API for running SageFormer oxygen forecasts.

Preferred entry point (raw sensor data):
    bundle = load_model("config/model_config.yaml", "checkpoints/bocknis_eck_3day.pth")
    result = predict_from_raw(bundle, "path/to/raw_station.csv")

Alternative (pre-built feature CSV, for advanced use):
    result = predict(bundle, "path/to/feature_csv.csv")

Raw station CSV must have columns: date, Sal, Temp, Oxygen
All other features (hour_sin/cos, day_sin/cos) are derived internally.
"""

import os
import types

import numpy as np
import pandas as pd
import torch
import yaml

from sageformer.model import Model
from src.preprocessing import build_input, load_scaler_stats, read_csv
from src.postprocessing import build_output_dataframe, inverse_transform_oxygen
from src.raw_to_features import prepare_features


def _config_to_namespace(cfg: dict) -> types.SimpleNamespace:
    m = cfg["model"]
    return types.SimpleNamespace(
        task_name=m["task_name"],
        seq_len=m["seq_len"],
        label_len=m["label_len"],
        pred_len=m["pred_len"],
        enc_in=m["enc_in"],
        c_out=m["c_out"],
        d_model=m["d_model"],
        d_ff=m["d_ff"],
        n_heads=m["n_heads"],
        e_layers=m["e_layers"],
        d_layers=m["d_layers"],
        dropout=m["dropout"],
        cls_len=m["cls_len"],
        graph_depth=m["graph_depth"],
        knn=m["knn"],
        embed_dim=m["embed_dim"],
        factor=m["factor"],
        activation=m["activation"],
        output_attention=m["output_attention"],
    )


def load_model(config_path: str, checkpoint_path: str, device: str = "cpu") -> dict:
    """
    Load SageFormer and return a bundle with the model and scaler stats.

    Parameters
    ----------
    config_path     : path to config/model_config.yaml
    checkpoint_path : path to checkpoints/bocknis_eck_3day.pth
    device          : "cpu" or "cuda"

    Returns
    -------
    dict with keys: "model", "stats", "config", "device"
    """
    config_dir = os.path.dirname(os.path.abspath(config_path))

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    stats_path = os.path.normpath(
        os.path.join(config_dir, "..", cfg["artifacts"]["scaler_stats"])
    )
    stats = load_scaler_stats(stats_path)

    args = _config_to_namespace(cfg)
    model = Model(args).float()
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return {"model": model, "stats": stats, "config": cfg, "device": device}


def predict_from_raw(
    bundle: dict,
    station_input,
    station_id: str = "Bocknis_Eck_2",
) -> pd.DataFrame:
    """
    Run a 3-day oxygen forecast from raw sensor observations.

    Partners supply only what their instruments record.  This function handles
    all feature engineering (cyclical time encodings, gap filling, normalisation)
    before passing data to the model.

    Parameters
    ----------
    bundle        : return value of load_model()
    station_input : path to a CSV  OR  a pandas DataFrame with columns:
                    date, Sal, Temp, Oxygen
                    Must cover at least 7 continuous days at 10-minute resolution.
    station_id    : identifier written into the output DataFrame

    Returns
    -------
    DataFrame with columns: timestamp, station_id, oxygen_mg_per_l
    432 rows — one per predicted 10-minute step over the 3-day horizon.
    """
    stats = bundle["stats"]
    df = prepare_features(station_input, stats)
    return predict(bundle, df, station_id=station_id)


def predict(
    bundle: dict,
    input_data,
    station_id: str = "Bocknis_Eck_2",
) -> pd.DataFrame:
    """
    Run a 3-day oxygen forecast from a pre-built 7-feature DataFrame or CSV.

    Use predict_from_raw() instead unless you have a specific reason to manage
    feature engineering yourself.

    Parameters
    ----------
    bundle     : return value of load_model()
    input_data : path to a feature CSV  OR  a DataFrame with 1008 rows and
                 columns [date, hour_sin, hour_cos, day_sin, day_cos, Sal, Temp, Oxygen]
    station_id : identifier written into the output DataFrame

    Returns
    -------
    DataFrame with columns: timestamp, station_id, oxygen_mg_per_l
    """
    if isinstance(input_data, str):
        df = read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError("input_data must be a file path (str) or a pandas DataFrame")

    stats  = bundle["stats"]
    device = bundle["device"]
    model  = bundle["model"]

    x_enc, _, forecast_start = build_input(df, stats)
    x_enc_t = torch.from_numpy(x_enc).float().to(device)

    # SageFormer's graph_constructor adds small random noise per forward pass.
    # Seed here so results are reproducible across calls.
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(x_enc_t)  # (1, 432, 7)

    pred_norm = out[0, :, -1].cpu().numpy()  # MS mode: last column = Oxygen
    pred_oxygen = inverse_transform_oxygen(pred_norm, stats)
    return build_output_dataframe(pred_oxygen, forecast_start, station_id)
